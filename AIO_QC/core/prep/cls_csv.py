# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Classification labels from a table (Qt-free): a CSV / TSV / TXT that says which class each image has.

Layouts read (found by themselves; every choice can be changed — sniff() says what it found):
    one column          file,label            (Hugging Face metadata.csv: file_name,label; Kaggle; most tools)
    several in one      file,labels           "cat;dog" / "cat|dog" / "cat dog"  (multi-label)
    one-hot             filename,cat,dog,…    1 / 0 per class   (Roboflow _classes.csv, multi-label exports)
    plain list          path label            (ImageNet / Caffe train.txt: "n01440764/x.JPEG 0", space separated)
    JSON Lines          {"file_name": …, "label": …} per line   (Hugging Face imagefolder metadata.jsonl; a list of
                        classes in "labels" is several classes)
A split column (train / val / test) is used when there is one. Numbers as classes are turned into names with a class
list (classes.txt, one name per line — line 1 = class 0), else kept as written.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field

FILE_HINTS = ("file_name", "filename", "file", "image", "image_name", "img", "path", "image_path", "name", "id")
LABEL_HINTS = ("label", "labels", "class", "classes", "category", "categories", "tag", "tags", "target", "y")
SPLIT_HINTS = ("split", "subset", "set", "fold")
IMG_RE = re.compile(r"\.(jpe?g|png|bmp|tiff?|webp|gif)$", re.I)
TRUE = {"1", "1.0", "true", "yes", "y", "t"}
FALSE = {"0", "0.0", "false", "no", "n", "f", ""}


@dataclass
class Spec:
    file_col: int = 0
    mode: str = "column"                 # column | onehot
    label_col: int = 1
    onehot_cols: list = field(default_factory=list)
    sep: str = "auto"                    # several classes in one cell: auto | ; | , | | | space | none
    split_col: int = -1                  # -1: none
    names: list = field(default_factory=list)      # class names for numbers (line 1 = class 0)
    delimiter: str = ","
    header: bool = True


def _jsonl_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """JSON Lines -> (header = every key in first-seen order, rows); a list value -> "a;b" (several classes)."""
    import json
    objs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except ValueError:
            continue
        if isinstance(o, dict):
            objs.append(o)
    keys: list = []
    for o in objs:
        for k in o:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, tuple)):
            return ";".join(str(x) for x in v)
        return "" if v is None else str(v)
    return [str(k) for k in keys], [[cell(o.get(k)) for k in keys] for o in objs]


def read_rows(path) -> tuple[list[str], list[list[str]], str, bool]:
    """(header, rows, delimiter, has a header). Plain "path label" lists are split on whitespace; JSON Lines
    (metadata.jsonl) are read by key."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    if str(path).lower().endswith((".jsonl", ".ndjson")) or text.lstrip().startswith("{"):
        head, rows = _jsonl_rows(text)
        return head, rows, ";", True
    sample = text[:65536]
    lines = [ln for ln in sample.splitlines() if ln.strip()][:50]
    if lines and not any(c in lines[0] for c in ",;\t|") and all(len(ln.split()) >= 2 for ln in lines[:20]):
        rows = [ln.split() for ln in text.splitlines() if ln.strip()]
        delim = " "
    else:
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    if not rows:
        return [], [], delim, False
    first = rows[0]
    header = not any(IMG_RE.search(c.strip()) for c in first) and not all(_num(c) for c in first[1:] if c) and \
        any(c.strip().lower() in FILE_HINTS + LABEL_HINTS + SPLIT_HINTS or not _num(c) for c in first)
    if delim == " " and len(first) == 2 and IMG_RE.search(first[0]):
        header = False
    if header:
        return [c.strip() for c in first], rows[1:], delim, True
    return [f"column {i + 1}" for i in range(max(len(r) for r in rows))], rows, delim, False


def _num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def sniff(path) -> tuple[Spec, list[str], list[list[str]], str]:
    """The layout found, the header, the first rows (for a preview) and a one-line description."""
    head, rows, delim, has_header = read_rows(path)
    spec = Spec(delimiter=delim, header=has_header)
    if not head:
        return spec, head, rows, "empty"
    low = [h.lower() for h in head]
    sample = rows[:500]

    def col(i):
        return [r[i].strip() if i < len(r) else "" for r in sample]
    # the file column: a known name, else the one with image names
    spec.file_col = next((i for i, h in enumerate(low) if h in FILE_HINTS), None)
    if spec.file_col is None:
        spec.file_col = max(range(len(head)), key=lambda i: sum(bool(IMG_RE.search(v)) for v in col(i)))
    spec.split_col = next((i for i, h in enumerate(low) if h in SPLIT_HINTS and i != spec.file_col), -1)
    others = [i for i in range(len(head)) if i not in (spec.file_col, spec.split_col)]
    binary = [i for i in others if all(v.lower() in TRUE | FALSE for v in col(i))]
    named = next((i for i in others if low[i] in LABEL_HINTS), None)
    if named is None and len(binary) >= 2 and len(binary) >= len(others) - 1:
        spec.mode, spec.onehot_cols = "onehot", binary
        desc = f"one column per class (1 / 0): {len(binary)} classes"
    else:
        spec.mode, spec.label_col = "column", named if named is not None else (others[0] if others else 0)
        vals = col(spec.label_col)
        seps = [s for s in (";", "|", ",") if any(s in v for v in vals)]
        spec.sep = seps[0] if seps else ("space" if any(" " in v.strip() for v in vals) and delim != " " else "none")
        desc = f"classes in “{head[spec.label_col]}”" + ("" if spec.sep == "none" else
                                                          f" (several per image, separated by “{spec.sep}”)")
    desc = f"files in “{head[spec.file_col]}” · " + desc + (f" · split in “{head[spec.split_col]}”"
                                                             if spec.split_col >= 0 else "")
    return spec, head, rows[:20], desc


def _split_cell(v: str, sep: str) -> list[str]:
    v = v.strip()
    if not v:
        return []
    if sep == "none":
        return [v]
    if sep == "space":
        return v.split()
    if sep == "auto":
        for s in (";", "|", ","):
            if s in v:
                return [x.strip() for x in v.split(s) if x.strip()]
        return [v]
    return [x.strip() for x in v.split(sep) if x.strip()]


def read_table(path, spec: Spec | None = None, notes=None):
    """-> (Records with .tags (and .split), class order). Records come from core.prep.importers."""
    from .importers import Record
    head, rows, _d, _h = read_rows(path)
    if spec is None:
        spec = sniff(path)[0]
    order, recs, bad = [], {}, 0

    def name(v: str) -> str:
        v = v.strip()
        if spec.names and re.fullmatch(r"-?\d+(\.0)?", v):
            i = int(float(v))
            return spec.names[i] if 0 <= i < len(spec.names) else v
        return v
    for r in rows:
        if spec.file_col >= len(r) or not r[spec.file_col].strip():
            bad += 1
            continue
        ref = r[spec.file_col].strip()
        if spec.mode == "onehot":
            tags = [head[i].strip() for i in spec.onehot_cols if i < len(r) and r[i].strip().lower() in TRUE]
        else:
            tags = [name(t) for t in _split_cell(r[spec.label_col] if spec.label_col < len(r) else "", spec.sep)]
        rec = recs.get(ref)
        if rec is None:
            rec = recs[ref] = Record(ref)
            rec.split = r[spec.split_col].strip().lower() if 0 <= spec.split_col < len(r) else ""
        for t in tags:
            if t and t not in rec.tags:
                rec.tags.append(t)
            if t and t not in order:
                order.append(t)
    if spec.mode == "onehot":
        order = [head[i].strip() for i in spec.onehot_cols]
    elif spec.names:
        order = [n for n in spec.names if n in order] + [o for o in order if o not in spec.names]
    if notes is not None and bad:
        notes.append(f"{bad:,} row(s) without a file name were skipped")
    return list(recs.values()), order


def read_names(path) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    from pathlib import Path
    if path.lower().endswith((".yaml", ".yml")):
        from AIO_QC.core.class_map import parse_yaml_class_map
        return list(parse_yaml_class_map(Path(path)).names)
    from AIO_QC.core.class_map import load_classes_file
    return list(load_classes_file(path))
