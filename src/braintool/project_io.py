"""Сохранение и загрузка разметки (сессия 7).

Файл `*.bpmarkup` — zip с двумя частями:
- `markup.json` — пайплайн, группы (папки, настройки), по каждому кадру: выдержка,
  на которой найдены маски, предупреждение, маски (номера, «принято», линия отреза,
  точки контура) и ключи растров;
- `masks.npz` — растры масок (упакованы по биту, сжаты).

Сами фото в файл не копируются — только пути к папкам групп. Если папку перенесли,
при загрузке она ищется рядом с файлом разметки по имени (частый случай — вся рабочая
папка лаборатории переписана на другой диск/компьютер).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .imaging import choose_mask_exposure, contrast_stretch_to_uint8, load_image, scan_group_folder
from .models import GroupConfig, MaskMode, ShotReview, SpecimenMask

FORMAT_VERSION = 1
FILE_SUFFIX = ".bpmarkup"


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(bool).ravel())


def _unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    return np.unpackbits(packed, count=n).astype(bool).reshape(shape)


def save_markup(
    path: Path, mode: MaskMode, groups: list[GroupConfig], reviews: list[ShotReview], current_index: int,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    shots = []
    for ri, review in enumerate(reviews):
        masks = []
        for mi, m in enumerate(review.masks):
            key = f"r{ri}_m{mi}"
            arrays[key] = _pack(m.mask)
            entry = {
                "animal_index": m.animal_index,
                "slice_index": m.slice_index,
                "accepted": bool(m.accepted),
                "mask": key,
                "cut_line": [list(p) for p in m.cut_line] if m.cut_line is not None else None,
                "rostral_anchor": list(m.rostral_anchor) if m.rostral_anchor is not None else None,
                "polygon": [list(p) for p in m.polygon] if m.polygon else None,
                "source_blob": None,
            }
            if m.source_blob is not None:
                arrays[key + "_src"] = _pack(m.source_blob)
                entry["source_blob"] = key + "_src"
            masks.append(entry)
        shots.append({
            "group": review.shot.group.name,
            "shot_key": review.shot.shot_key,
            "exposure": review.shot.chosen_exposure,
            "shape": list(review.image.shape[:2]),
            "warning": review.warning,
            "masks": masks,
        })
    meta = {
        "format": "BrainPhotoTool markup",
        "version": FORMAT_VERSION,
        "mode": mode.value,
        "current_index": current_index,
        "groups": [
            {
                "name": g.name, "folder": str(g.folder), "cols": g.cols, "rows": g.rows,
                "is_control": g.is_control, "condition": g.condition,
            }
            for g in groups
        ],
        "shots": shots,
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("markup.json", json.dumps(meta, ensure_ascii=False, indent=1))
        zf.writestr("masks.npz", buf.getvalue())
    # запись через временный файл — сбой посреди сохранения не портит прошлую версию
    tmp.replace(path)


@dataclass
class LoadedMarkup:
    mode: MaskMode
    groups: list[GroupConfig]
    reviews: list[ShotReview]
    current_index: int
    problems: list[str]


def _resolve_folder(saved: str, markup_path: Path) -> Path | None:
    folder = Path(saved)
    if folder.is_dir():
        return folder
    # папку перенесли вместе с файлом разметки — ищем по хвосту пути рядом с файлом
    parts = folder.parts
    for n in range(1, min(4, len(parts)) + 1):
        candidate = markup_path.parent.joinpath(*parts[-n:])
        if candidate.is_dir():
            return candidate
    return None


def load_markup(path: Path) -> LoadedMarkup:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("markup.json").decode("utf-8"))
        arrays = np.load(io.BytesIO(zf.read("masks.npz")))
        arrays = {k: arrays[k] for k in arrays.files}
    if meta.get("version", 0) > FORMAT_VERSION:
        raise ValueError("Файл разметки сохранён более новой версией программы — обновите программу.")
    mode = MaskMode(meta["mode"])
    problems: list[str] = []

    groups: dict[str, GroupConfig] = {}
    shots_by_group: dict[str, dict] = {}
    for g in meta["groups"]:
        folder = _resolve_folder(g["folder"], path)
        if folder is None:
            problems.append(f"Группа «{g['name']}»: папка не найдена ({g['folder']}) — её кадры пропущены.")
            continue
        group = GroupConfig(
            name=g["name"], folder=folder, mode=mode, cols=g["cols"], rows=g.get("rows"),
            is_control=g.get("is_control", False), condition=g.get("condition", ""),
        )
        groups[group.name] = group
        shots_by_group[group.name] = {s.shot_key: s for s in scan_group_folder(group)}

    reviews: list[ShotReview] = []
    for s in meta["shots"]:
        group = groups.get(s["group"])
        if group is None:
            continue
        shot = shots_by_group[s["group"]].get(s["shot_key"])
        if shot is None:
            problems.append(f"Группа «{s['group']}», кадр «{s['shot_key']}»: файлов больше нет — пропущен.")
            continue
        exposure = s.get("exposure")
        if exposure not in shot.exposure_files:
            exposure = choose_mask_exposure(shot)
        shot.chosen_exposure = exposure
        try:
            image = load_image(shot.exposure_files[exposure])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Группа «{s['group']}», кадр «{s['shot_key']}»: не открывается ({exc}) — пропущен.")
            continue
        shape = tuple(s["shape"])
        if image.shape[:2] != shape:
            problems.append(
                f"Группа «{s['group']}», кадр «{s['shot_key']}»: размер фото изменился "
                "с момента сохранения — разметка к нему не подходит, кадр пропущен."
            )
            continue
        masks = []
        for e in s["masks"]:
            masks.append(SpecimenMask(
                animal_index=e["animal_index"],
                slice_index=e["slice_index"],
                mask=_unpack(arrays[e["mask"]], shape),
                accepted=e["accepted"],
                cut_line=tuple(tuple(p) for p in e["cut_line"]) if e.get("cut_line") else None,
                source_blob=_unpack(arrays[e["source_blob"]], shape) if e.get("source_blob") else None,
                rostral_anchor=tuple(e["rostral_anchor"]) if e.get("rostral_anchor") else None,
                polygon=[tuple(p) for p in e["polygon"]] if e.get("polygon") else None,
            ))
        reviews.append(ShotReview(
            shot=shot, image=image, display_image=contrast_stretch_to_uint8(image),
            masks=masks, warning=s.get("warning"),
        ))
    current = min(int(meta.get("current_index", 0)), max(len(reviews) - 1, 0))
    return LoadedMarkup(mode=mode, groups=list(groups.values()), reviews=reviews,
                        current_index=current, problems=problems)


# ---------- имена файлов (сессия 8) ----------
# Раньше диалоги сохранения всегда предлагали одно и то же имя («разметка_срезы»,
# «результаты.xlsx»), а автосохранение было одним файлом на пайплайн — новый проект
# молча затирал прошлый. Теперь в имени группы + дата-время, а занятое имя не
# предлагается никогда.

AUTOSAVE_DIR = Path.home() / "BrainPhotoTool автосохранение"
AUTOSAVE_KEEP = 20  # сколько последних автосохранений на пайплайн хранить
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _pipeline_word(mode: MaskMode) -> str:
    return "срезы" if mode == MaskMode.WHOLE_BLOB else "эпителий"


def _safe(text: str, limit: int = 60) -> str:
    text = _BAD_CHARS.sub("_", text).strip(" .")
    return text[:limit].rstrip(" .") or "без_названия"


def file_stem(prefix: str, mode: MaskMode, group_names: list[str]) -> str:
    """«разметка_срезы_<группа>[ и ещё N]_2026-09-24_14-05»."""
    parts = [prefix, _pipeline_word(mode)]
    if group_names:
        first = _safe(group_names[0])
        parts.append(first if len(group_names) == 1 else f"{first} и ещё {len(group_names) - 1}")
    parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    return "_".join(parts)


def unique_path(path: Path) -> Path:
    """Тот же путь, а если файл уже есть — «имя (2).ext», «имя (3).ext»…"""
    path = Path(path)
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        n += 1
    return candidate


def new_autosave_path(mode: MaskMode, group_names: list[str]) -> Path:
    """Своё автосохранение для каждого нового проекта (прошлые не затираются)."""
    return unique_path(AUTOSAVE_DIR / (file_stem("автосохранение", mode, group_names) + FILE_SUFFIX))


def list_autosaves(mode: MaskMode) -> list[Path]:
    """Автосохранения пайплайна, от последнего к первому (включая старое
    «автосохранение_срезы.bpmarkup» из сессии 7)."""
    if not AUTOSAVE_DIR.is_dir():
        return []
    files = AUTOSAVE_DIR.glob(f"автосохранение_{_pipeline_word(mode)}*{FILE_SUFFIX}")
    return sorted((p for p in files if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def prune_autosaves(mode: MaskMode, keep_also: Path | None = None) -> None:
    """Удалить старые автосохранения сверх AUTOSAVE_KEEP (текущее не трогается)."""
    for old in list_autosaves(mode)[AUTOSAVE_KEEP:]:
        if keep_also is not None and old == keep_also:
            continue
        try:
            old.unlink()
        except OSError:
            pass
