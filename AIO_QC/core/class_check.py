# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
YOLO class-list verification: do the class ids used in the label files fit the class names?

    scan = scan_class_ids(label_files)          # threaded, reads only the first token of each line
    verdict = verify_classes(names, scan)       # "mismatch" | "ok" (+ human-readable details)

A mismatch (a label uses id >= number of names) means the class list cannot be the right one for this
dataset. A match is necessary but not sufficient (a list in the wrong ORDER still matches), so the UI asks the
user to confirm that the file is the final one — remembered per dataset until the file changes.
"""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassIdScan:
    files: int = 0                         # label files read
    lines: int = 0                         # label lines with a class id
    counts: Counter = field(default_factory=Counter)      # id -> lines
    examples: dict = field(default_factory=dict)          # id -> first file using it
    bad: int = 0                           # lines whose first token is not an integer
    unreadable: int = 0

    @property
    def max_id(self) -> int:
        return max(self.counts) if self.counts else -1


def _scan_file(path: Path):
    counts, bad = Counter(), 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                tok = line.split(maxsplit=1)
                if not tok:
                    continue
                try:
                    v = float(tok[0])
                    if not v.is_integer() or v < 0:
                        raise ValueError
                    counts[int(v)] += 1
                except ValueError:
                    bad += 1
    except FileNotFoundError:
        return Counter(), 0                 # image without a label file = background image
    except OSError:
        return None, 0
    return counts, bad


def scan_class_ids(label_files, workers: int | None = None, progress=None) -> ClassIdScan:
    """Threaded (I/O bound): ~100k small label files in a few seconds on an SSD."""
    files = [Path(p) for p in label_files if p is not None]
    out = ClassIdScan()
    workers = workers or min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (path, (counts, bad)) in enumerate(zip(files, pool.map(_scan_file, files, chunksize=64)), 1):
            if counts is None:
                out.unreadable += 1
                continue
            out.files += 1
            out.bad += bad
            for cid, n in counts.items():
                out.counts[cid] += n
                out.lines += n
                out.examples.setdefault(cid, path)
            if progress is not None and i % 500 == 0:
                progress(i, len(files))
    return out


@dataclass
class ClassVerdict:
    kind: str                     # "mismatch" | "ok" | "no_labels"
    title: str
    details: str
    missing_ids: list = field(default_factory=list)    # ids used in labels but without a name
    unused_ids: list = field(default_factory=list)     # names never used in labels


def _range_text(ids: list[int], limit: int = 12) -> str:
    ids = sorted(ids)
    if not ids:
        return ""
    runs, start, prev = [], ids[0], ids[0]
    for v in ids[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        runs.append(f"{start}" if start == prev else f"{start}–{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(runs[:limit]) + (" …" if len(runs) > limit else "")


def verify_classes(names: list[str], scan: ClassIdScan, source_name: str = "the class list") -> ClassVerdict:
    n = len(names)
    used = sorted(scan.counts)
    if not used:
        return ClassVerdict("no_labels", "No labels to check",
                            f"No label lines were found, so {source_name} cannot be checked against the data.")
    missing = [i for i in used if i >= n]
    unused = [i for i in range(n) if i not in scan.counts]
    if missing:
        lines = sum(scan.counts[i] for i in missing)
        ex = ", ".join(sorted({Path(scan.examples[i]).name for i in missing})[:3])
        return ClassVerdict(
            "mismatch", "Classes don't match the labels",
            f"The labels use class id{'s' if len(missing) > 1 else ''} {_range_text(missing)}, but {source_name} "
            f"has only {n} name{'s' if n != 1 else ''} (ids 0–{n - 1}).\n"
            f"{lines:,} label line(s) would have no class name — e.g. in {ex}.",
            missing_ids=missing, unused_ids=unused)
    detail = (f"{source_name}: {n} classes · the labels use {len(used)} of them "
              f"(ids {_range_text(used)}) in {scan.lines:,} lines across {scan.files:,} files.")
    if unused:
        detail += (f"\nNever used in the labels: {_range_text(unused)} "
                   f"({', '.join(names[i] for i in unused[:4])}{' …' if len(unused) > 4 else ''}).")
    if scan.bad:
        detail += f"\n{scan.bad:,} line(s) have a non-integer class id (see Validation)."
    return ClassVerdict("ok", "Classes fit the labels", detail, unused_ids=unused)
