# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Extend Model (Qt-free): fine-tune a model you trained with more images and / or new classes — without it forgetting
what it already knew.

    rows, names, old_of = plan_classes(base_names, new_names, actions)      the class plan
    P = ExtendPlan(base=…, base_names=…, names=names, old_of=old_of, old_yaml=…, new_yaml=…)
    tree = build_tree(P, out)                                                the training copy (hard links)
    P.save(run / "aioqc" / "extend.json")                                    lineage + resume / CLI

The class plan: the base model's classes keep their ids (locked: its scoring head is enlarged, not reset — its rows
are copied, core.train_runner.expand_head), the new dataset's classes are matched by name (letter case and spaces
ignored) and the rest are appended. Per class: keep / rename / merge into another / drop.

The training copy mixes the old training data (when you have it — strongly recommended) and the new images: images
are hard-linked (copied when a link is not possible), label files are rewritten only where an id changes. The old
validation images stay a split of their own (val_old) so "did it forget?" is measured on exactly the same images as
before; the new ones are val_new. A new training image that is an exact copy of an old validation image is left out
(it would leak the answer).

Old classes on new images: images labelled only with the new classes still contain cars, people … — unlabelled, the
model would learn them as background and forget them. The base model pre-labels them (core.train_runner, where torch
is) before training; with "review" on, training waits until you have looked at them (the copy is a normal YOLO
dataset: open it in AIO QC).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

KEEP, NEW, RENAME, MERGE, DROP = "keep", "new", "rename", "merge", "drop"
FORGET_LIMIT = 0.03            # an old class losing more than 3 mAP points (50-95) is flagged


def norm(name: str) -> str:
    return " ".join(str(name).split()).casefold()


@dataclass
class ClassRow:
    name: str                      # the name in the extended model
    old_id: int | None             # in the base model (None: a new class)
    new_id: int | None             # in the extended model (None: dropped)
    action: str                    # keep | new | rename | merge | drop
    source: str = ""               # the dataset label it comes from ("" = only in the base model)
    old_images: int = 0
    new_images: int = 0
    note: str = ""


def plan_classes(base_names, new_names, actions=None, counts_old=None, counts_new=None):
    """The class plan. actions: {dataset label: ("rename", new name) | ("merge", existing name) | ("drop",)}.
    -> (rows, names of the extended model in id order, old_of {new id: old id}, label_map {dataset label: new id
    or None}). Base classes are never re-numbered."""
    actions = actions or {}
    counts_old, counts_new = counts_old or {}, counts_new or {}
    names = [str(n) for n in base_names]
    at = {norm(n): i for i, n in enumerate(names)}
    rows = [ClassRow(n, i, i, KEEP, old_images=int(counts_old.get(n, 0))) for i, n in enumerate(names)]
    label_map = {}
    order = [x for x in new_names if (actions.get(x) or (None,))[0] != MERGE] + \
        [x for x in new_names if (actions.get(x) or (None,))[0] == MERGE]      # merges after their target exists
    for lab in order:
        act = actions.get(lab) or ()
        kind = act[0] if act else None
        if kind == DROP:
            label_map[lab] = None
            rows.append(ClassRow(lab, None, None, DROP, lab, new_images=int(counts_new.get(lab, 0)),
                                 note="not trained"))
            continue
        target = act[1] if kind in (RENAME, MERGE) and len(act) > 1 and str(act[1]).strip() else lab
        k = norm(target)
        if k in at:
            i = at[k]
            label_map[lab] = i
            r = rows[i] if i < len(base_names) else next(x for x in rows if x.new_id == i)
            r.new_images += int(counts_new.get(lab, 0))
            if norm(lab) != norm(r.name):
                rows.append(ClassRow(lab, None, i, MERGE, lab, new_images=0, note=f"merged into {r.name}"))
            elif not r.source:
                r.source = lab
            continue
        i = len(names)
        names.append(str(target).strip())
        at[k] = i
        label_map[lab] = i
        rows.append(ClassRow(names[i], None, i, NEW if kind is None else RENAME, lab,
                             new_images=int(counts_new.get(lab, 0)),
                             note="" if kind is None else f"from “{lab}”"))
    old_of = {i: i for i in range(len(base_names))}
    return rows, names, old_of, label_map


@dataclass
class ExtendPlan:
    base: str                                   # the model to extend (.pt)
    base_names: list
    names: list                                 # the extended model's classes, in id order
    old_of: dict                                # new id -> old id (rows of the scoring head copied)
    label_map: dict                             # new dataset label -> new id (None = dropped)
    new_yaml: str                               # the new images (a YOLO dataset.yaml)
    old_yaml: str = ""                          # the base model's training data (recommended)
    goal: str = "classes"                       # images (more images, same classes) | classes (new classes too)
    prelabel: bool = True                       # the base model labels its classes on the new images
    prelabel_conf: float = 0.5
    review: bool = True                         # training waits until the pre-labels are reviewed
    two_stage: bool = True                      # 1: head only (backbone frozen), 2: everything at a low LR
    stage1_epochs: int = 5
    freeze: int = 10
    stage2_lr: float = 0.001
    forget_limit: float = FORGET_LIMIT
    created: float = field(default_factory=time.time)
    parent: dict = field(default_factory=dict)  # lineage: the base model's run, its scores

    @property
    def new_classes(self) -> list:
        return self.names[len(self.base_names):]

    def save(self, path) -> None:
        d = asdict(self)
        d["old_of"] = {str(k): v for k, v in self.old_of.items()}
        Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ExtendPlan":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["old_of"] = {int(k): int(v) for k, v in (d.get("old_of") or {}).items()}
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


# ====================================================================================================== the copy
def is_cls_folder(path) -> bool:
    return bool(path) and Path(path).is_dir()


def cls_images(root, split: str) -> list[tuple[Path, str]]:
    """A classification folder's images of a split, with their class: [(image, class folder)]."""
    from .dataset import IMAGE_EXTENSIONS, class_folder_splits
    d = dict(class_folder_splits(Path(root))).get(split)
    if d is None and split == "val":
        d = dict(class_folder_splits(Path(root))).get("test")
    if d is None:
        return []
    out = []
    for c in sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith(".")):
        for dp, dn, fns in os.walk(c):
            dn[:] = [x for x in dn if not x.startswith(".")]
            for f in sorted(fns):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    out.append((Path(dp, f), c.name))
    return out


def _yaml_images(yaml_path, split: str) -> list[Path]:
    from .training import _images_of, _load_yaml
    yp = Path(yaml_path)
    if yp.is_dir():                                    # a classification folder
        return [p for p, _c in cls_images(yp, split)]
    d = _load_yaml(yp)
    root = Path(str(d.get("path") or yp.parent))
    if not root.is_absolute():
        root = (yp.parent / root).resolve()
    if not root.exists():
        root = yp.parent
    entry = d.get(split)
    if not entry:                                  # no train: / val: lines — the usual folders
        entry = next((str(c) for c in (root / split / "images", root / "images" / split) if c.is_dir()), None)
        if entry is None:
            return []
    imgs, _probs = _images_of(entry, root)
    return sorted(imgs)


def yaml_names(yaml_path) -> list:
    """The classes of a dataset.yaml — or of a classification folder (train/'s class folders, in name order)."""
    if is_cls_folder(yaml_path):
        from .dataset import class_folder_splits
        tr = dict(class_folder_splits(Path(yaml_path))).get("train")
        return sorted(d.name for d in tr.iterdir() if d.is_dir() and not d.name.startswith(".")) if tr else []
    from .training import _load_yaml, _names
    return _names(_load_yaml(Path(yaml_path)))


def remap_lines(text: str, m: dict) -> tuple[str, Counter, int]:
    """Class ids through `m` (old id -> new id or None = line dropped). -> (text, Counter(new id), lines dropped)."""
    out, seen, dropped = [], Counter(), 0
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        try:
            cid = int(float(p[0]))
        except ValueError:
            continue
        new = m.get(cid, cid) if m else cid
        if new is None:
            dropped += 1
            continue
        p[0] = str(new)
        out.append(" ".join(p))
        seen[new] += 1
    return "\n".join(out) + ("\n" if out else ""), seen, dropped


def _place(src: Path, dst: Path, how: Counter):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        from .yolo_convert import _jpeg_truncated
        if _jpeg_truncated(src):                   # trainers repair such JPEGs in place: never through a link
            shutil.copy2(src, dst)
            how["copied"] += 1
            how["copied_bytes"] += src.stat().st_size
            return
    try:
        os.link(src, dst)
        how["linked"] += 1
    except OSError:
        shutil.copy2(src, dst)
        how["copied"] += 1
        how["copied_bytes"] += src.stat().st_size


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def _digest(p: Path) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _folder_tag(folder: Path) -> str:
    import hashlib
    return f"{folder.name}_{hashlib.blake2b(str(folder).encode(), digest_size=3).hexdigest()}"


def estimate(P: ExtendPlan) -> dict:
    """How many images, and how much disk space if hard links are not possible (another drive)."""
    out = {"old_train": 0, "old_val": 0, "new_train": 0, "new_val": 0, "bytes": 0}
    for tag, yaml in (("old", P.old_yaml), ("new", P.new_yaml)):
        if not yaml:
            continue
        for sp in ("train", "val"):
            imgs = _yaml_images(yaml, sp)
            out[f"{tag}_{sp}"] = len(imgs)
            out["bytes"] += sum(max(0, _size(i)) for i in imgs[:2000]) * (len(imgs) / max(1, min(2000, len(imgs))))
    out["bytes"] = int(out["bytes"])
    return out


def build_cls_tree(P: ExtendPlan, out, progress=None, cancel=None) -> dict:
    """Classifiers: <out>/train/<class>/{old,new}_<folder>/… and val/ (old + new), val_old.txt / val_new.txt
    ("path<TAB>class") for the before / after check. Hard links; a class renamed / merged lands in its new folder;
    a class left out is not copied."""
    out = Path(out)
    st = {"images": Counter(), "labels_rewritten": 0, "labels_linked": 0, "lines_dropped": 0, "leak_skipped": 0,
          "how": Counter(), "objects": Counter(), "new_train_labels": []}
    at = {norm(n): n for n in P.names}

    def target(src, cls):
        if src == "new" and cls in P.label_map:
            i = P.label_map[cls]
            return None if i is None else P.names[i]
        return at.get(norm(cls))
    jobs = []
    for src, root in (("old", P.old_yaml), ("new", P.new_yaml)):
        if not root:
            continue
        for sp in ("train", "val"):
            jobs += [(src, sp, img, cls) for img, cls in cls_images(root, sp)]
    old_val = {}
    for src, sp, img, _c in jobs:
        if src == "old" and sp == "val":
            old_val.setdefault(_size(img), []).append(img)
    lists = {"val_old": [], "val_new": []}
    total = len(jobs)
    for n, (src, sp, img, cls) in enumerate(jobs, 1):
        if cancel is not None and cancel.is_set():
            st["cancelled"] = True
            break
        if progress is not None and (n % 200 == 0 or n == total):
            progress(n, total)
        name = target(src, cls)
        if name is None:
            st["lines_dropped"] += 1
            continue
        if src == "new" and sp == "train" and _size(img) in old_val and \
                any(_digest(o) == _digest(img) for o in old_val[_size(img)]):
            st["leak_skipped"] += 1
            continue
        safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in name)
        dst = out / sp / safe / f"{src}_{_folder_tag(img.parent)}" / img.name
        if dst.exists():
            continue
        _place(img, dst, st["how"])
        st["images"][f"{src}_{sp if sp == 'train' else 'val_' + src}"] += 1
        st["objects"][name] += 1
        if sp == "val":
            lists[f"val_{src}"].append(f"{dst.relative_to(out).as_posix()}\t{name}")
    for k, v in lists.items():
        if v:
            (out / f"{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    for name in P.names:                                   # every class needs its train folder (ids by name)
        (out / "train" / "".join("_" if ch in '\\/:*?"<>|' else ch for ch in name)).mkdir(parents=True,
                                                                                         exist_ok=True)
    return st


def build_tree(P: ExtendPlan, out, progress=None, cancel=None) -> dict:
    """The training copy: images/{train,val_old,val_new}/{old,new}/… + labels/… + data.yaml (val = both),
    val_old.yaml, val_new.yaml. -> stats."""
    from .training import _label_path
    out = Path(out)
    st = {"images": Counter(), "labels_rewritten": 0, "labels_linked": 0, "lines_dropped": 0, "leak_skipped": 0,
          "how": Counter(), "objects": Counter(), "new_train_labels": []}
    old_map = {}                                  # old dataset ids -> new ids (by name; base ids are locked)
    if P.old_yaml:
        on = yaml_names(P.old_yaml)
        at = {norm(n): i for i, n in enumerate(P.names)}
        old_map = {i: at.get(norm(n)) for i, n in enumerate(on)}
    by_name = {norm(x): j for j, x in enumerate(P.names)}
    new_map = {i: (P.label_map[n] if n in P.label_map else by_name.get(norm(n)))
               for i, n in enumerate(yaml_names(P.new_yaml))}
    jobs = []
    if P.old_yaml:
        jobs += [("old", "train", "train", i) for i in _yaml_images(P.old_yaml, "train")]
        jobs += [("old", "val", "val_old", i) for i in _yaml_images(P.old_yaml, "val")]
    jobs += [("new", "train", "train", i) for i in _yaml_images(P.new_yaml, "train")]
    jobs += [("new", "val", "val_new", i) for i in _yaml_images(P.new_yaml, "val")]
    # leakage: a new training image that is an old validation image
    old_val = {}
    for src, _sp, dst_split, img in jobs:
        if dst_split == "val_old":
            old_val.setdefault(_size(img), []).append(img)
    total = len(jobs)
    for n, (src, _sp, dst_split, img) in enumerate(jobs, 1):
        if cancel is not None and cancel.is_set():
            st["cancelled"] = True
            break
        if progress is not None and (n % 200 == 0 or n == total):
            progress(n, total)
        if src == "new" and dst_split == "train" and _size(img) in old_val:
            d = _digest(img)
            if any(_digest(o) == d for o in old_val[_size(img)]):
                st["leak_skipped"] += 1
                continue
        base = Path(src) / _folder_tag(img.parent)                             # a folder per source folder
        di = out / "images" / dst_split / base / img.name
        dl = out / "labels" / dst_split / base / (img.stem + ".txt")
        if di.exists():
            continue
        _place(img, di, st["how"])
        st["images"][f"{src}_{dst_split}"] += 1
        lab = _label_path(img)
        m = old_map if src == "old" else new_map
        text = lab.read_text(encoding="utf-8", errors="replace") if lab.is_file() else ""
        if m and any(k != v for k, v in m.items()):
            text, seen, dropped = remap_lines(text, m)
            st["lines_dropped"] += dropped
            dl.parent.mkdir(parents=True, exist_ok=True)
            dl.write_text(text, encoding="utf-8")
            st["labels_rewritten"] += 1
        else:
            _t, seen, _d = remap_lines(text, {})
            if lab.is_file():
                dl.parent.mkdir(parents=True, exist_ok=True)
                _place(lab, dl, Counter())
                st["labels_linked"] += 1
            else:
                dl.parent.mkdir(parents=True, exist_ok=True)
                dl.write_text("", encoding="utf-8")
        st["objects"].update({P.names[k] if k < len(P.names) else str(k): v for k, v in seen.items()})
        if src == "new" and dst_split == "train":
            st["new_train_labels"].append([str(di), str(dl)])
    _write_yamls(P, out)
    return st


def _write_yamls(P: ExtendPlan, out: Path):
    def y(train, val):
        lines = [f"path: {json.dumps(str(out.resolve()).replace(chr(92), '/'))}", f"train: {train}",
                 "val: " + (val if isinstance(val, str) else "[" + ", ".join(val) + "]"),
                 "", f"nc: {len(P.names)}", "names:"] + [f"  {i}: {json.dumps(n, ensure_ascii=False)}"
                                                       for i, n in enumerate(P.names)]
        return "\n".join(lines) + "\n"
    vals = [v for v in ("images/val_old", "images/val_new") if (out / v).is_dir()]
    (out / "data.yaml").write_text(y("images/train", vals or "images/train"), encoding="utf-8")
    for v in vals:
        (out / f"{Path(v).name}.yaml").write_text(y("images/train", v), encoding="utf-8")


# ====================================================================================================== results
def compare(before: dict, after: dict, base_names, limit: float = FORGET_LIMIT) -> dict:
    """Per-class scores before (the base model) and after (the extended one), by class name.
    before / after: {"val_old": [{"name", "map", "map50", …}], "val_new": […]}. -> {rows, forgot, learned}."""
    rows, forgot = [], []
    for split in ("val_old", "val_new"):
        b = {norm(r["name"]): r for r in before.get(split) or []}
        for r in after.get(split) or []:
            k = norm(r["name"])
            old = b.get(k)
            d = None if old is None or old.get("map") is None or r.get("map") is None else r["map"] - old["map"]
            row = {"split": split, "name": r["name"], "before": None if old is None else old.get("map"),
                   "after": r.get("map"), "delta": d, "old_class": k in {norm(n) for n in base_names}}
            rows.append(row)
            if d is not None and row["old_class"] and d < -limit:
                forgot.append(row)
    learned = [r for r in rows if not r["old_class"] and (r["after"] or 0) > 0]
    return {"rows": rows, "forgot": forgot, "learned": learned}


def history_line(P: ExtendPlan, run, result: dict | None) -> dict:
    """One line of the model's Improve history (lineage)."""
    return {"when": time.time(), "run": str(run), "from": P.base, "goal": P.goal, "new_classes": P.new_classes,
            "classes": len(P.names), "forgot": [r["name"] for r in (result or {}).get("forgot", [])]}


def base_info(pt, python: str = "", timeout: float = 180) -> dict:
    """What is known about the model to extend: its classes, and — when it was trained here — its run, its
    training data and its scores. Read from the run (no torch needed); else asked of the environment's Python."""
    pt = Path(pt)
    info = {"names": [], "dataset_yaml": "", "run": "", "metrics": {}, "epoch_time": None, "train_images": None}
    run = pt.parent.parent if pt.parent.name == "weights" else None
    spec = run / "aioqc" / "spec.json" if run is not None else None
    if spec is not None and spec.is_file():
        try:
            d = json.loads(spec.read_text(encoding="utf-8"))
            info.update(run=str(run), dataset_yaml=str((d.get("extend") or {}).get("tree") and
                                                         Path(d["extend"]["tree"]) / "data.yaml" or
                                                         d.get("dataset_yaml") or ""))
            info["names"] = list((d.get("extend") or {}).get("names") or d.get("names") or [])
            st = json.loads((run / "aioqc" / "status.json").read_text(encoding="utf-8"))
            info.update(metrics=st.get("metrics") or {}, epoch_time=st.get("epoch_time"),
                        train_images=st.get("train_images"))
        except (OSError, ValueError):
            pass
    if not info["names"] and python:
        import subprocess
        code = ("import json,sys\nfrom ultralytics import YOLO\nm=YOLO(sys.argv[1])\n"
                "print(json.dumps([m.names[i] for i in range(len(m.names))]))")
        try:
            r = subprocess.run([python, "-c", code, str(pt)], capture_output=True, text=True, timeout=timeout)
            info["names"] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else []
            if r.returncode != 0:
                info["error"] = (r.stderr or "")[-600:]
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
            info["error"] = str(exc)
    return info


def time_estimate(info: dict, images: int) -> str:
    """'about 3 min per epoch' from the base run's own speed, scaled to the number of training images."""
    et, n = info.get("epoch_time"), info.get("train_images")
    if not et or not n or not images:
        return ""
    sec = float(et) * images / max(1, int(n))
    return f"about {sec / 60:.0f} min per epoch" if sec >= 90 else f"about {sec:.0f} s per epoch"
