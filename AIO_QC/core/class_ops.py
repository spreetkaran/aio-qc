# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Class removal / re-ordering (port of remove_class.py, GUI-driven).

Flow:  plan = plan_removal(...)  ->  (dry run: show plan)  ->  apply_plan(plan, backup=...)
       -> verify_after(...)
LabelMe: shapes with removed labels are dropped.
YOLO:    rows of removed ids dropped, remaining ids remapped to the new order, and the
         class map file (dataset.yaml nc/names, or classes.txt) rewritten.
Images are never modified. Empty label files are preserved.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import AnnotationError, DatasetType
from .io import atomic_write_text

YAML_SPECIAL = set(":#[]{}&,*!|>'\"%@`")


@dataclass
class RemovalPlan:
    dataset_type: DatasetType
    remove: set                               # labels (LabelMe) or old ids (YOLO)
    old_names: list[str] = field(default_factory=list)
    new_names: list[str] = field(default_factory=list)
    id_map: dict[int, int | None] = field(default_factory=dict)
    affected: dict[str, Counter] = field(default_factory=dict)   # key -> Counter(removed labels / ids)
    remapped_files: set[str] = field(default_factory=set)
    unknown_ids: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    class_file: Path | None = None

    @property
    def rows_removed(self) -> int:
        return sum(sum(c.values()) for c in self.affected.values())

    @property
    def files_to_write(self) -> set[str]:
        return set(self.affected) | self.remapped_files


def build_id_map(old_names, remove_ids, new_names) -> dict[int, int | None]:
    remaining = [n for i, n in enumerate(old_names) if i not in remove_ids]
    if sorted(new_names) != sorted(remaining):
        raise ValueError("New order must contain every remaining class exactly once")
    to_new = {n: i for i, n in enumerate(new_names)}
    return {i: (None if i in remove_ids else to_new[n]) for i, n in enumerate(old_names)}


def plan_removal(adapter, samples, remove, new_names=None, progress=None) -> RemovalPlan:
    """Pre-scan everything; no file is touched."""
    plan = RemovalPlan(adapter.type, set(remove))
    if adapter.type == DatasetType.LABELME:
        for n, s in enumerate(samples):
            try:
                c = Counter(str(sh.get("label", "")) for sh in adapter.load(s).shapes)
            except AnnotationError as exc:
                plan.errors.append(f"{s.name}: {exc}")
                continue
            hit = Counter({k: v for k, v in c.items() if k in plan.remove})
            if hit:
                plan.affected[s.key] = hit
            if progress:
                progress(n + 1, len(samples))
        return plan

    plan.old_names = list(adapter.class_map.names)
    plan.class_file = adapter.class_map.source
    if not plan.old_names:
        raise ValueError("YOLO class map is empty — load classes.txt or add dataset.yaml first")
    plan.new_names = list(new_names) if new_names is not None else \
        [n for i, n in enumerate(plan.old_names) if i not in plan.remove]
    plan.id_map = build_id_map(plan.old_names, plan.remove, plan.new_names)
    for n, s in enumerate(samples):
        if s.ann_path is None or not s.ann_path.is_file():
            continue
        try:
            ann = adapter.load(s)
        except AnnotationError as exc:
            plan.errors.append(f"{s.name}: {exc}")
            continue
        hit = Counter()
        for sh in ann.shapes:
            cid = sh.get("_class_id")
            if cid is None:
                plan.errors.append(f"{s.name}: line {sh['_line']}: {sh.get('_error')}")
            elif cid not in plan.id_map:
                plan.unknown_ids[cid] += 1
            elif plan.id_map[cid] is None:
                hit[cid] += 1
            elif plan.id_map[cid] != cid:
                plan.remapped_files.add(s.key)
        if hit:
            plan.affected[s.key] = hit
        if progress:
            progress(n + 1, len(samples))
    if plan.unknown_ids:
        plan.errors.append("Label files contain class ids outside the class map: "
                           + ", ".join(f"{k} (×{v})" for k, v in sorted(plan.unknown_ids.items()))
                           + " — fix them first (Validation / Label Ambiguities)")
    return plan


def backup_files(root: Path, paths, progress=None) -> str:
    """Store the originals in the compact backup store (.aio_qc/backups.sqlite). -> description for the summary."""
    from .backups import BackupStore
    store = BackupStore(root)
    paths = [Path(p) for p in paths]
    op = store.begin("class_removal", "Class removal " + time.strftime("%Y-%m-%d %H:%M"))
    try:
        for i, p in enumerate(paths):
            store.save(op, p)
            if progress and i % 200 == 0:
                progress(i, len(paths))
    finally:
        store.close()
    return f"{len(paths):,} file(s) in .aio_qc/backups.sqlite (Tools ▸ Storage… to export or delete)"


def _rewrite_yolo_file(path: Path, id_map) -> None:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        new = id_map[int(float(parts[0]))]
        if new is None:
            continue
        parts[0] = str(new)
        out.append(" ".join(parts))
    atomic_write_text(path, "\n".join(out) + ("\n" if out else ""))


def write_class_file(path: Path, names: list[str]) -> None:
    """dataset.yaml: replace nc/names blocks, keep everything else. txt: one name per line."""
    if path.suffix.lower() not in (".yaml", ".yml"):
        atomic_write_text(path, "\n".join(names) + "\n")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    root_key = lambda l, k: l.startswith(f"{k}:")
    block = ["names:"]
    for i, n in enumerate(names):
        safe = '"' + n.replace('"', '\\"') + '"' if any(ch in n for ch in YAML_SPECIAL) else n
        block.append(f"  {i}: {safe}")
    out, i, wrote_nc, wrote_names = [], 0, False, False
    while i < len(lines):
        l = lines[i]
        if root_key(l, "nc"):
            out.append(f"nc: {len(names)}")
            wrote_nc = True
        elif root_key(l, "names"):
            out.extend(block)
            wrote_names = True
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t", "-"))):
                if not lines[i].strip() and i + 1 < len(lines) and not lines[i + 1].startswith((" ", "\t", "-")):
                    break
                i += 1
            continue
        else:
            out.append(l)
        i += 1
    if not wrote_names:
        out.extend(block)
    if not wrote_nc:
        out.append(f"nc: {len(names)}")
    atomic_write_text(path, "\n".join(out) + "\n")


def apply_plan(adapter, samples, plan: RemovalPlan, backup: bool, progress=None) -> tuple[str | None, list[str]]:
    by_key = {s.key: s for s in samples}
    keys = sorted(plan.files_to_write)
    files = [by_key[k].ann_path for k in keys if k in by_key]
    if plan.dataset_type != DatasetType.LABELME and plan.class_file:
        files.append(plan.class_file)
    backup_dir = backup_files(adapter.root, files) if backup else None
    errors = []
    for n, k in enumerate(keys):
        s = by_key.get(k)
        try:
            if plan.dataset_type == DatasetType.LABELME:
                ann = adapter.load(s)
                ann.shapes = [sh for sh in ann.shapes if str(sh.get("label", "")) not in plan.remove]
                adapter.save(s, ann)
            else:
                _rewrite_yolo_file(s.ann_path, plan.id_map)
        except Exception as exc:
            errors.append(f"{s.name if s else k}: {exc}")
        if progress:
            progress(n + 1, len(keys))
    if plan.dataset_type != DatasetType.LABELME:
        if plan.class_file is None:
            errors.append("No class map file to update")
        else:
            write_class_file(plan.class_file, plan.new_names)
    return backup_dir, errors


def verify_after(adapter, samples, plan: RemovalPlan) -> list[str]:
    """Post-scan: no removed class left, every YOLO id within the new map."""
    problems = []
    for s in samples:
        try:
            ann = adapter.load(s)
        except AnnotationError as exc:
            problems.append(f"{s.name}: {exc}")
            continue
        for sh in ann.shapes:
            if plan.dataset_type == DatasetType.LABELME:
                if str(sh.get("label", "")) in plan.remove:
                    problems.append(f"{s.name}: {sh.get('label')!r} still present")
            else:
                cid = sh.get("_class_id")
                if cid is not None and not 0 <= cid < len(plan.new_names):
                    problems.append(f"{s.name}: id {cid} outside 0-{len(plan.new_names) - 1}")
    return problems
