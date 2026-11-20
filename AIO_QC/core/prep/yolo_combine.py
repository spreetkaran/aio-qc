# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Combine & Flatten for YOLO (Qt-free): one or more YOLO datasets (any nesting: images/ + labels/ mirrored, with or
without train / val / test) → one dataset with one images/ and labels/ folder per split (or one for all).

    · class lists merged by name (letter case and spaces ignored): the first dataset's order is kept, classes only
      the others have are added at the end; every label file is rewritten with the new ids (only where they change)
    · names made unique with the naming engine (core.naming: folder prefix by default — camA/img001 and camB/img001
      both survive), verified, listed in renamed.csv
    · exact duplicate images (the same bytes) added once — the plan says whether their labels agree
    · hard links where possible (no extra disk space), else copies; the sources are never changed
    · every file logged (RunLog): Preparation ▸ Undo removes exactly what the run created
"""
from __future__ import annotations

import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import common as C

SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


@dataclass
class Options:
    sources: list
    dst: str
    keep_splits: bool = True               # train / val / test of the sources kept (else all in one)
    link: str = "hardlink"                 # hardlink | copy
    dedupe: bool = True
    background: bool = True                # images without a label file: kept (empty label)
    naming: dict = field(default_factory=dict)     # core.naming.Scheme as a dict (default: the folder prefix)
    workers: int = max(4, min(32, (os.cpu_count() or 4) * 2))


@dataclass
class Plan:
    opts: Options
    classes: list = field(default_factory=list)          # merged, in id order
    remap: dict = field(default_factory=dict)            # source root -> {old id: new id}
    files: list = field(default_factory=list)            # [{img, label, split, dst_img, dst_lab, src}]
    dups: list = field(default_factory=list)             # [(path, kept path, same labels?)]
    stats: Counter = field(default_factory=Counter)
    details: dict = field(default_factory=lambda: defaultdict(list))
    renamed: list = field(default_factory=list)          # [(original rel, new rel)]

    @property
    def actions(self):
        return self.files

    def summary_rows(self):
        s = self.stats
        rows = [("Images → the combined dataset", s["images"]),
                ("    with a label file", s["labelled"]), ("    without (background, empty label)", s["background"]),
                ("Classes in the combined list", len(self.classes)),
                ("    label files rewritten with new class ids", s["remapped"]),
                ("Renamed so no two files share a name", s["renamed"]),
                ("Exact duplicate images, added once", s["dups"]),
                ("    of them with DIFFERENT labels (review)", s["dups_diff"]),
                ("Label files without their image (not added)", s["orphan_labels"])]
        if s["unknown_ids"]:
            rows.append(("Lines with a class id the source's list does not have (kept as is)", s["unknown_ids"]))
        return rows


def source_classes(root: Path) -> list[str]:
    from AIO_QC.core.class_map import find_class_map
    labels = tuple(p for p in root.rglob("labels") if p.is_dir())
    return list(find_class_map(root, labels).names)


def _key(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def merge_classes(lists) -> tuple[list, list]:
    """[[names of dataset 1], [names of 2], …] → (merged names, [{old id: new id} per dataset])."""
    merged, at = [], {}
    maps = []
    for names in lists:
        m = {}
        for i, n in enumerate(names):
            k = _key(n)
            if k not in at:
                at[k] = len(merged)
                merged.append(str(n).strip())
            m[i] = at[k]
        maps.append(m)
    return merged, maps


def _pairs(root: Path):
    """(image, label or None, split) for every image under an images/ folder; label files without an image."""
    out, labels_seen, all_labels = [], set(), set()
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith(".") and d not in (C.LOG_DIR,)]
        parts = Path(dp).relative_to(root).parts
        low = [p.lower() for p in parts]
        if "labels" in low:
            for f in fns:
                if f.lower().endswith(".txt") and f.lower() not in ("classes.txt",):
                    all_labels.add(os.path.normcase(os.path.join(dp, f)))
        if "images" not in low:
            continue
        i = len(low) - 1 - low[::-1].index("images")
        split = next((SPLITS[p] for p in low if p in SPLITS), "")
        lab_dir = Path(root, *parts[:i], "labels", *parts[i + 1:])
        for f in fns:
            if os.path.splitext(f)[1].lower() not in C.IMAGE_EXTS:
                continue
            lab = lab_dir / (os.path.splitext(f)[0] + ".txt")
            has = lab.is_file()
            if has:
                labels_seen.add(os.path.normcase(str(lab)))
            out.append((Path(dp, f), lab if has else None, split))
    return out, all_labels - labels_seen


def _rel_for_naming(root: Path, img: Path, many: bool) -> str:
    """camA/day1/x.jpg — the folders that tell files apart (images/ and the split dropped; the source's name first
    when several sources are combined)."""
    parts = [p for p in img.relative_to(root).parts[:-1] if p.lower() != "images" and p.lower() not in SPLITS]
    if many:
        parts = [root.name] + parts
    return "/".join(parts + [img.name])


def plan(opts: Options, progress=None, cancel=None) -> Plan:
    from AIO_QC.core import naming as N
    P = Plan(opts)
    dst = Path(opts.dst).resolve()
    srcs = [Path(s).resolve() for s in opts.sources]
    for s in srcs:
        if s == dst or dst in s.parents or s in dst.parents:
            raise ValueError("The combined folder must be outside the source folders (and not contain them).")
    say = progress or (lambda *_: None)
    lists = [source_classes(s) for s in srcs]
    P.classes, maps = merge_classes(lists)
    P.remap = {str(s): m for s, m in zip(srcs, maps)}
    for s, names, m in zip(srcs, lists, maps):
        if any(i != j for i, j in m.items()):
            P.details["Class ids that change"].append(
                f"{s.name}: " + ", ".join(f"{names[i]} {i}→{j}" for i, j in m.items() if i != j))
        if not names:
            P.details["Sources without a class list (ids kept)"].append(s.name)
    entries = []
    for s in srcs:
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        say(0, 0, f"Scanning {s.name}")
        pairs, orphans = _pairs(s)
        P.stats["orphan_labels"] += len(orphans)
        for o in sorted(orphans)[:200]:
            P.details["Label files without their image"].append(o)
        for img, lab, split in pairs:
            if lab is None and not opts.background:
                continue
            entries.append({"src": s, "img": img, "label": lab, "split": split if opts.keep_splits else ""})
    if opts.keep_splits and any(e["split"] for e in entries):
        for e in entries:                          # a source without splits joins train
            e["split"] = e["split"] or "train"
    # exact duplicates: the first copy is kept
    if opts.dedupe and entries:
        say(0, 0, "Finding duplicate images")
        groups = C.exact_groups(list(range(len(entries))), lambda i: _size(entries[i]["img"]),
                                lambda i: str(entries[i]["img"]), None, opts.workers, cancel,
                                lambda d, t: say(d, t, "Comparing same-size images"))
        drop = set()
        for g in groups:
            g.sort()
            keep = entries[g[0]]
            for i in g[1:]:
                same = _labels_of(entries[i], P) == _labels_of(keep, P)
                P.dups.append((str(entries[i]["img"]), str(keep["img"]), same))
                P.stats["dups"] += 1
                P.stats["dups_diff"] += not same
                P.details["Duplicates" + ("" if same else " with DIFFERENT labels — review")].append(
                    f"{entries[i]['img']}  =  {keep['img']}")
                drop.add(i)
        entries = [e for i, e in enumerate(entries) if i not in drop]
    # names: unique per split (or across all), the naming engine, verified
    many = len(srcs) > 1
    items = [N.Item(i, _rel_for_naming(e["src"], e["img"], many), split=e["split"], path=str(e["img"]),
                    extra_exts=(".txt",)) for i, e in enumerate(entries)]
    sc = N.Scheme.from_dict(opts.naming) if opts.naming else N.Scheme(preset="folder")
    root_len = len(str(dst)) + len("/train/images")
    NP = N.plan(items, sc, flat=True, root_len=root_len)
    if not NP.ok:
        NP = N.resolve(NP, root_len=root_len)
    for it in NP.items:
        e = entries[it.key]
        stem = NP.new[it.key]
        base = dst / e["split"] if e["split"] else dst
        e["dst_img"] = base / "images" / (stem + e["img"].suffix)
        e["dst_lab"] = base / "labels" / (stem + ".txt")
        if stem != e["img"].stem or it.folders:
            P.renamed.append((it.rel, (e["split"] + "/" if e["split"] else "") + stem + e["img"].suffix))
        P.stats["renamed"] += stem != e["img"].stem
        P.stats["labelled" if e["label"] else "background"] += 1
        m = P.remap[str(e["src"])]
        P.stats["remapped"] += bool(e["label"]) and any(i != j for i, j in m.items())
    P.files = entries
    P.stats["images"] = len(entries)
    P.details["Renamed (original → new)"] = [f"{a}  →  {b}" for a, b in P.renamed if a.rsplit("/", 1)[-1] !=
                                              b.rsplit("/", 1)[-1]]
    P.details["Combined class list"] = [f"{i}: {n}" for i, n in enumerate(P.classes)]
    return P


def _size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return None


def _labels_of(e, P) -> list:
    if e["label"] is None:
        return []
    return sorted(remap_text(e["label"].read_text(encoding="utf-8", errors="replace"), P.remap[str(e["src"])])[0]
                  .splitlines())


def remap_text(text: str, m: dict) -> tuple[str, int]:
    """Rewrite the class id of every line. -> (text, lines whose id the map does not have)."""
    if not m or all(i == j for i, j in m.items()):
        return text, 0
    out, unknown = [], 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            out.append(line)
            continue
        if cid in m:
            parts[0] = str(m[cid])
        else:
            unknown += 1
        out.append(" ".join(parts))
    return "\n".join(out) + ("\n" if out else ""), unknown


def run(P: Plan, progress=None, cancel=None) -> dict:
    o = P.opts
    dst = Path(o.dst)
    dst.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(dst / C.LOG_DIR, "combine_yolo", keep_originals=False)
    done, errors = Counter(), []
    n = len(P.files)
    try:
        for k, e in enumerate(P.files, 1):
            if cancel is not None and cancel.is_set():
                raise C.Cancelled()
            try:
                for t in (e["dst_img"], e["dst_lab"]):
                    if t.exists():
                        raise FileExistsError(f"{t} already exists")
                e["dst_img"].parent.mkdir(parents=True, exist_ok=True)
                e["dst_lab"].parent.mkdir(parents=True, exist_ok=True)
                how = "copy"
                if o.link == "hardlink":
                    try:
                        os.link(e["img"], e["dst_img"])
                        how = "link"
                    except OSError:
                        shutil.copy2(e["img"], e["dst_img"])
                else:
                    shutil.copy2(e["img"], e["dst_img"])
                log.created(e["dst_img"], how, str(e["img"]), "image")
                text = e["label"].read_text(encoding="utf-8", errors="replace") if e["label"] else ""
                text, unknown = remap_text(text, P.remap[str(e["src"])])
                P.stats["unknown_ids"] += unknown
                C.write_atomic(str(e["dst_lab"]), text.encode("utf-8"))
                log.created(e["dst_lab"], "class ids remapped" if e["label"] else "empty (background)",
                            str(e["label"] or ""), "label")
                done["images"] += 1
                done[how] += 1
            except OSError as exc:
                log.error("combine", str(e["img"]), str(e["dst_img"]), exc)
                errors.append(f"{e['img']}: {exc}")
            if progress is not None and (k % 25 == 0 or k == n):
                progress(k, n, "Linking / copying files")
        splits = sorted({e["split"] for e in P.files})
        from AIO_QC.core.yolo_convert import dataset_yaml
        for name, text in (("dataset.yaml", dataset_yaml(dst, splits or [""], P.classes, "det",
                                                          ", ".join(Path(s).name for s in o.sources))),
                           ("classes.txt", "\n".join(P.classes) + "\n")):
            t = dst / name
            if not t.exists():
                C.write_atomic(str(t), text.encode("utf-8"))
                log.created(t, "combined class list", "", "meta")
        if P.renamed:
            import csv
            t = dst / "renamed.csv"
            with open(t, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["original", "new"])
                w.writerows(P.renamed)
            log.created(t, "every renamed file", "", "meta")
    except C.Cancelled:
        errors.append("stopped by you — what was done so far is in the log (and can be undone)")
    finally:
        log.close()
    return {"done": done, "errors": errors, "log": str(log.csv_path), "undo": str(log.jsonl_path)}
