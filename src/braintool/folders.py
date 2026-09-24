"""Какие папки брать в проект, когда пользователь добавил/перетащил папку (сессия 8).

Данные в лаборатории лежат так: `<исследование>/<группа>/{голова, череп и мозг, срезы}`.
Пользователь может указать любую из ступенек:
- саму нужную подпапку («срезы» / «череп и мозг») — берётся она, группа = имя папки выше;
- папку группы — внутри ищется нужная подпапка пайплайна;
- папку исследования — берутся все группы внутри, у которых есть нужная подпапка;
- любую папку прямо с фото (своя раскладка) — берётся как есть.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .imaging import IMAGE_EXTENSIONS
from .models import MaskMode

# как называется подпапка с фото для каждого пайплайна (для сообщений)
SUBFOLDER_TITLES = {
    MaskMode.WHOLE_BLOB: "срезы",
    MaskMode.ROSTRAL_CUT: "череп и мозг",
}

# на сколько уровней вниз искать группы (исследование → группа → подпапка)
_MAX_DEPTH = 2


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().replace("ё", "е")).strip()


def is_pipeline_subfolder(folder: Path, mode: MaskMode) -> bool:
    """Папка с фото нужного пайплайна: «срезы» (и «Срезы 2», «срез») для срезов,
    «череп и мозг» (и «черепа и мозги») для эпителия."""
    name = _norm(folder.name)
    if mode == MaskMode.WHOLE_BLOB:
        return name.startswith("срез")
    return name.startswith("череп")


def _has_images(folder: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir())
    except OSError:
        return False


def _subdirs(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


@dataclass
class FoundFolders:
    groups: list[tuple[str, Path]] = field(default_factory=list)  # (название группы, папка с фото)
    problems: list[str] = field(default_factory=list)


def _find(folder: Path, mode: MaskMode, depth: int) -> list[tuple[str, Path]]:
    if is_pipeline_subfolder(folder, mode):
        return [(folder.parent.name or folder.name, folder)]
    subs = _subdirs(folder)
    own = [s for s in subs if is_pipeline_subfolder(s, mode)]
    if own:
        # несколько подходящих подпапок («срезы», «срезы 2») — каждая своей группой
        if len(own) == 1:
            return [(folder.name, own[0])]
        return [(f"{folder.name} — {s.name}", s) for s in own]
    # папка прямо с фото (своя раскладка) — только если её выбрали саму: при спуске
    # вглубь так подхватились бы «голова»/«срезы» у группы без нужной подпапки
    if depth == 0 and _has_images(folder):
        return [(folder.name, folder)]
    if depth >= _MAX_DEPTH:
        return []
    found: list[tuple[str, Path]] = []
    for s in subs:
        found.extend(_find(s, mode, depth + 1))
    return found


def find_group_folders(paths: list[Path], mode: MaskMode) -> FoundFolders:
    """Разворачивает выбранные/перетащенные папки в список групп (название, папка с фото)."""
    result = FoundFolders()
    seen: set[Path] = set()
    sub = SUBFOLDER_TITLES[mode]
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            result.problems.append(f"«{path.name}» — не папка, пропущено.")
            continue
        found = _find(path, mode, 0)
        if not found:
            result.problems.append(
                f"В папке «{path.name}» нет подпапки «{sub}» и нет фото — ничего не добавлено."
            )
            continue
        for name, folder in found:
            key = folder.resolve()
            if key in seen:
                continue
            seen.add(key)
            result.groups.append((name, folder))
    return result


def data_dir_for(groups) -> Path:
    """Папка, где по умолчанию предлагать сохранять разметку/таблицы: общая папка
    исследования для всех групп (у `<исследование>/<группа>/срезы` — `<исследование>`)."""
    tops = []
    for g in groups:
        folder = Path(g.folder)
        group_dir = folder.parent if is_pipeline_subfolder(folder, g.mode) else folder
        tops.append(str(group_dir.parent))
    if not tops:
        return Path.home()
    try:
        common = Path(os.path.commonpath(tops))
    except ValueError:  # разные диски Windows
        common = Path(tops[0])
    return common if common.is_dir() else Path.home()
