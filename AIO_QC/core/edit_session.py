# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
External-editor session: the editor works on COPIES; AIO QC writes the real files.

    <root>/.aio_qc/edit_sessions/<ts>/          work folder the editor opens (hidden -> never discovered)
        0001_<name>.json / .txt                  annotation copies (numbered in flag order)
        0001_<name>.jpg                          image: hard link (no extra space) -> symlink -> copy
        labels.txt / classes.txt                 class list for the editor (labelme --labels / labelImg)
        session.json                             manifest (crash recovery, undo)
    <root>/.aio_qc/backups.sqlite                originals, saved before the first write-back (core.backups)

Space: the work copies never carry imageData (the editor gets the image file — or, for JSONs whose only copy of the
image is embedded, the embedded image decoded once into the work folder) and the editor runs without image data, so a
session costs a few KB per file. Whether the written-back JSON embeds the image is AIO QC's choice (`image_data`:
"none" = imageData null, "embed" = keep the file's embedded image or embed the image file the way the selected
labelme version does), identical for labelme 5 and 6 (same key; labelme 6 needs the key present, so null is written).

Why copies: both labelme versions rewrite `imagePath` relative to where the JSON was opened from and decide
on their own whether to embed the image (5.x yes, 6.x no); labelImg numbers YOLO classes by its own list order
and rewrites classes.txt on every save. Writing back through AIO QC keeps imagePath, flags, version and unknown
keys, applies the chosen image-data policy, validates labels against the class list, keeps untouched YOLO rows
byte-for-byte, refuses to overwrite files that changed outside the session (conflict), and can undo.

Change detection is poll-based (mtime + size, stable for two polls) so it works for every editor, OS and
file system, including network drives and editors that replace files instead of rewriting them.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backups import BackupStore
from .dataset import DatasetType
from .editors import LABELIMG, LABELME
from .io import atomic_write_text, load_json
from .labelme_image import KEY as IMG_KEY, encode_b64, image_file_of, sniff_ext

SESSIONS = "edit_sessions"
STATUS = ("waiting", "fixed", "unchanged", "rejected", "conflict", "error")


class Rejected(Exception):
    """The edited file cannot be written back (shown to the user; the editor stays open to fix it)."""


def _sig(path: Path) -> str | None:
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return None


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "file"


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard link (same volume, no extra space, no admin rights) -> symlink -> copy."""
    for how, fn in (("hardlink", os.link), ("symlink", os.symlink)):
        try:
            fn(src, dst)
            return how
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)
    return "copy"


def _canon(d):
    """JSON form: a digest compared after the manifest is reloaded must not differ only by tuple vs list."""
    return json.loads(json.dumps(d))


def _digest_labelme(shapes: list, flags: dict | None = None) -> list:
    d = [(s.get("label"), s.get("shape_type") or "polygon",
          [(round(float(p[0]), 2), round(float(p[1]), 2)) for p in s.get("points") or []]) for s in shapes]
    if flags is not None:                              # classification with LabelMe flags: the flags ARE the label
        d.append(("__flags__", sorted(str(k) for k, v in flags.items() if v is True)))
    return _canon(d)


def _digest_yolo(shapes: list) -> list:
    return _canon([(s["label"], s["shape_type"], [(round(p[0], 6), round(p[1], 6)) for p in s["points"]])
                   for s in shapes])


def yolo_task(shape_lists: list[list[dict]]) -> str:
    """detect | segment | obb | mixed — from the valid shapes of the files in the session."""
    rect = poly = poly4 = 0
    for shapes in shape_lists:
        for s in shapes:
            if s.get("shape_type") == "rectangle":
                rect += 1
            elif s.get("shape_type") == "polygon":
                poly += 1
                poly4 += len(s.get("points") or []) == 4
    if poly == 0:
        return "detect"
    if rect == 0:
        return "obb" if poly4 == poly else "segment"
    return "mixed"


@dataclass
class Entry:
    key: str
    name: str                      # display name
    mirror: str                    # annotation file name in the work folder
    image: str | None = None       # image file name in the work folder
    orig_sig: str | None = None    # original annotation file (mtime|size) as last seen / written by us
    orig_existed: bool = True
    base: list = field(default_factory=list)       # digest of the original shapes (unchanged detection)
    last_mirror_sig: str | None = None
    pending_sig: str | None = None
    status: str = "waiting"
    message: str = ""
    prev_review: list | None = None                # [status, note] before the session (undo)
    backup: str | None = None                      # "db" (backups.sqlite) or a legacy relative path
    writes: int = 0
    hidden: list | None = None                     # label filter: indices of the shapes kept out of the editor
    hidden_stale: bool = False                     # the original changed outside: recompute `hidden` from labels


@dataclass
class Update:
    """What to write for one entry: `apply(ann)` mutates the full (non-light) annotation."""
    entry: Entry
    apply: object
    summary: str


class EditSession:
    def __init__(self, root: Path, work: Path, editor: str, dataset_type: str, class_names: list[str],
                 image_data: str, source_title: str, task: str = ""):
        self.root, self.work = Path(root), Path(work)
        self.editor, self.dataset_type = editor, dataset_type
        self.class_names = list(class_names)
        self.image_data = image_data          # none | embed  (LabelMe datasets; "keep" in sessions from ≤ 3.3)
        self.source_title = source_title
        self.source_version = ""
        self.backup_op: int | None = None     # operation id in backups.sqlite
        self.task = task                      # YOLO: detect | segment | obb | mixed
        self.flags_matter = False             # LabelMe-flag classification: flag changes are edits
        self.only_labels: list[str] | None = None   # show only these labels in the editor (others kept as they are)
        self._pending_hidden: dict[str, list] = {}
        self.entries: list[Entry] = []
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        self.active = True
        self.backup_dir = self.root / ".aio_qc" / "backups" / f"edit_{self.work.name}"
        self.stray: set[str] = set()

    # ------------------------------------------------------------------ persistence
    def save(self):
        data = {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k not in ("entries", "root", "work", "backup_dir", "stray")}
        data["entries"] = [asdict(e) for e in self.entries]
        atomic_write_text(self.work / "session.json", json.dumps(data, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, work: Path, root: Path) -> "EditSession":
        d = json.loads((Path(work) / "session.json").read_text(encoding="utf-8"))
        s = cls(root, work, d["editor"], d["dataset_type"], d["class_names"], d["image_data"],
                d["source_title"], d.get("task", ""))
        s.created, s.active = d.get("created", ""), d.get("active", True)
        s.flags_matter = d.get("flags_matter", False)
        s.source_version = d.get("source_version", "")
        s.backup_op = d.get("backup_op")
        s.only_labels = d.get("only_labels")
        known = set(Entry.__dataclass_fields__)
        s.entries = [Entry(**{k: v for k, v in e.items() if k in known}) for e in d["entries"]]
        return s

    @staticmethod
    def unfinished(root: Path) -> list[Path]:
        base = Path(root) / ".aio_qc" / SESSIONS
        out = []
        if base.is_dir():
            for d in sorted(base.iterdir()):
                try:
                    if json.loads((d / "session.json").read_text(encoding="utf-8")).get("active"):
                        out.append(d)
                except (OSError, ValueError):
                    continue
        return out

    def editable(self, shape: dict) -> bool:
        """YOLO shapes the editor may touch: a box / polygon whose class id has a name. Everything else (invalid
        lines, ids without a name) is hidden from the editor and written back unchanged."""
        if shape.get("shape_type") not in ("rectangle", "polygon"):
            return False
        cid = shape.get("_class_id")
        return cid is not None and (not self.class_names or cid < len(self.class_names))

    def shown(self, shape: dict) -> bool:
        """Label filter: does this shape go to the editor? (No filter: every shape.)"""
        return self.only_labels is None or shape.get("label") in self.only_labels

    def hidden_of(self, e: Entry, shapes: list) -> list[int]:
        """Indices (in `shapes`, the current original) of the shapes kept out of the editor."""
        if self.only_labels is None:
            return []
        if e.hidden is not None and not e.hidden_stale and all(i < len(shapes) for i in e.hidden):
            return list(e.hidden)
        return [i for i, x in enumerate(shapes) if not self.shown(x)]

    @staticmethod
    def merge_hidden(shapes: list, hidden: list[int], edited: list) -> tuple[list, list[int]]:
        """Put the edited (shown) shapes back between the hidden ones: hidden shapes keep their order, the edited
        block goes where the first shown shape was. Returns (shapes, new hidden indices)."""
        hid = set(hidden)
        out, new_hidden, placed = [], [], False
        for i, x in enumerate(shapes):
            if i in hid:
                new_hidden.append(len(out))
                out.append(x)
            elif not placed:
                out.extend(edited)
                placed = True
        if not placed:
            out.extend(edited)
        return out, new_hidden

    # ------------------------------------------------------------------ create
    @classmethod
    def create(cls, root: Path, samples, adapter, editor: str, class_names: list[str], image_data: str,
               source_title: str, review=None, source_version: str = "5.0.0",
               label_hints: list[str] | None = None, only_labels: list[str] | None = None) -> "EditSession":
        """Build the work folder. `adapter` is the BASE (unmapped) adapter: files are edited with their real labels."""
        root = Path(root)
        work = root / ".aio_qc" / SESSIONS / time.strftime("%Y%m%d_%H%M%S")
        n = 1
        while work.exists():
            work = work.with_name(f"{work.name}_{n}")
            n += 1
        work.mkdir(parents=True)
        is_yolo = adapter.type in (DatasetType.YOLO, DatasetType.YOLO_SPLIT)
        s = cls(root, work, editor, "yolo" if is_yolo else "labelme", class_names, image_data, source_title)
        s.source_version = source_version
        s.flags_matter = getattr(adapter.type, "value", "") == "cls_labelme"
        s.only_labels = sorted(set(only_labels)) if only_labels is not None else None
        loaded = []
        for sample in samples:
            try:
                ann = adapter.load(sample)
            except Exception as exc:
                loaded.append((sample, None, str(exc)))
                continue
            loaded.append((sample, ann, None))
        if is_yolo:
            s.task = yolo_task([[x for x in a.shapes if s.editable(x)] for _s, a, _e in loaded if a is not None])
            if editor == LABELIMG and s.task != "detect":
                shutil.rmtree(work, ignore_errors=True)
                raise ValueError(f"labelImg edits boxes only, but these files contain {s.task} shapes. "
                                 "Use labelme instead.")
        for i, (sample, ann, err) in enumerate(loaded, 1):
            stem = f"{i:04d}_{_safe(Path(sample.name).stem)}"
            e = Entry(key=sample.key, name=sample.display if hasattr(sample, "display") else sample.name,
                      mirror=stem + (".txt" if editor == LABELIMG else ".json"))
            if review is not None:
                r = review.get(sample.key)
                e.prev_review = [r["status"], r.get("note")] if r else None
            s.entries.append(e)
            if err is not None:
                e.status, e.message = "error", f"could not read the annotation: {err}"
                continue
            try:
                s._prepare_entry(e, sample, ann, adapter, stem, source_version)
            except Exception as exc:                           # one bad file never breaks the session
                e.status, e.message = "error", str(exc)
        if not is_yolo or editor == LABELME:
            names = class_names or sorted(set(label_hints or []))      # hints: suggestions, not validated
            (work / "labels.txt").write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        if editor == LABELIMG:
            (work / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
        for e in s.entries:
            e.last_mirror_sig = _sig(work / e.mirror)
        s.save()
        return s

    def _prepare_entry(self, e: Entry, sample, ann, adapter, stem: str, source_version: str):
        work = self.work
        img = ann.image_path or sample.image_path
        img = Path(img) if img else None
        if img is not None and img.is_file():
            e.image = stem + img.suffix.lower()
            link_or_copy(img, work / e.image)
        e.orig_existed = bool(sample.ann_path and Path(sample.ann_path).is_file())
        e.orig_sig = _sig(Path(sample.ann_path)) if sample.ann_path else None
        if self.dataset_type == "labelme":
            data = {k: v for k, v in ann.raw.items() if k != IMG_KEY}     # never copy the embedded image
            embedded = ann.raw.get(IMG_KEY)
            if e.image is None:
                if not embedded:
                    raise ValueError("image file not found and no embedded image data")
                blob = base64.b64decode(embedded)                         # the JSON is the only copy of the image
                e.image = stem + sniff_ext(blob)
                (work / e.image).write_bytes(blob)
            data["imagePath"] = e.image
            data[IMG_KEY] = None
            shapes = ann.shapes
            if self.only_labels is not None:
                e.hidden = self.hidden_of(e, shapes)
                hid = set(e.hidden)
                shapes = [x for i, x in enumerate(shapes) if i not in hid]
                data["shapes"] = [{k: v for k, v in x.items() if not str(k).startswith("_")} for x in shapes]
                if e.hidden:
                    e.message = f"{len(e.hidden)} shape(s) of other labels hidden in the editor and kept unchanged"
            (work / e.mirror).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            e.base = _digest_labelme(shapes, (ann.raw.get("flags") or {}) if self.flags_matter else None)
            return
        # ---- YOLO
        if e.image is None:
            raise ValueError("image file not found")
        size = adapter.image_size(sample, ann)
        if not size:
            raise ValueError("cannot read the image size")
        W, H = size
        valid = [x for x in ann.shapes if self.editable(x)]
        bad = len(ann.shapes) - len(valid)
        if self.only_labels is not None:
            e.hidden = [i for i, x in enumerate(ann.shapes) if self.editable(x) and not self.shown(x)]
            hid = set(e.hidden)
            valid = [x for i, x in enumerate(ann.shapes) if self.editable(x) and i not in hid]
        if self.editor == LABELIMG:
            lines = []
            for x in valid:
                cid = x.get("_class_id")
                (x1, y1), (x2, y2) = x["points"][0], x["points"][1]
                lines.append(f"{cid} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {abs(x2 - x1):.6f} {abs(y2 - y1):.6f}")
            (work / e.mirror).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            shapes = [{"label": x["label"], "points": [[p[0] * W, p[1] * H] for p in x["points"]], "group_id": None,
                       "description": "", "shape_type": x["shape_type"], "flags": {}} for x in valid]
            data = {"version": source_version, "flags": {}, "shapes": shapes, "imagePath": e.image,
                    "imageData": None, "imageHeight": H, "imageWidth": W}
            (work / e.mirror).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        e.base = _digest_yolo(valid)
        notes = []
        if bad:
            notes.append(f"{bad} line(s) with an invalid format or a class id without a name are not shown in the "
                         "editor and are kept unchanged")
        if e.hidden:
            notes.append(f"{len(e.hidden)} shape(s) of other labels hidden in the editor and kept unchanged")
        e.message = "; ".join(notes)

    # ------------------------------------------------------------------ polling
    def poll(self) -> list[Entry]:
        """Entries whose work file changed and has been stable for two polls (never half-written files)."""
        ready = []
        for e in self.entries:
            if e.status == "error" and e.last_mirror_sig is None:
                continue
            sig = _sig(self.work / e.mirror)
            if sig is None or sig == e.last_mirror_sig:
                e.pending_sig = None
                continue
            if e.pending_sig == sig:
                ready.append(e)
            else:
                e.pending_sig = sig
        if self.editor == LABELIMG:          # labelImg can be switched to Pascal VOC by the user: warn once
            for p in self.work.glob("*.xml"):
                self.stray.add(p.name)
        return ready

    def mark_seen(self, e: Entry):
        e.last_mirror_sig = _sig(self.work / e.mirror)
        e.pending_sig = None

    # ------------------------------------------------------------------ build an update
    def build_update(self, e: Entry, adapter, sample) -> Update | None:
        """None = unchanged. Raises Rejected (bad content) or ValueError (conflict / unreadable)."""
        if sample.ann_path is not None and _sig(Path(sample.ann_path)) != e.orig_sig:
            raise ValueError("conflict: the original file changed outside this session")
        if self.dataset_type == "labelme":
            return self._update_labelme(e, sample)
        return self._update_yolo(e, adapter, sample)

    def _check_labels(self, labels):
        if not self.class_names:
            return
        unknown = sorted({l for l in labels if l not in self.class_names})
        if unknown:
            raise Rejected(f"label(s) not in the class list: {', '.join(unknown)}")

    def _update_labelme(self, e: Entry, sample=None) -> Update | None:
        try:
            data = load_json(self.work / e.mirror)
        except Exception as exc:
            raise ValueError(f"could not read the edited file yet ({exc})")
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise Rejected("the edited file has no 'shapes' list")
        problems = []
        for i, s in enumerate(shapes):
            pts = s.get("points") or []
            st = s.get("shape_type") or "polygon"
            if not str(s.get("label") or "").strip():
                problems.append(f"shape {i} has no label")
            need = {"polygon": 3, "rectangle": 2, "circle": 2, "line": 2, "linestrip": 2, "point": 1}.get(st, 1)
            if st != "mask" and len(pts) < need:
                problems.append(f"shape {i} ({st}) has {len(pts)} point(s), needs {need}")
        if problems:
            raise Rejected("; ".join(problems[:4]))
        self._check_labels([s.get("label") for s in shapes])
        flags = data.get("flags")
        new_digest = _digest_labelme(shapes, (flags or {}) if self.flags_matter else None)
        policy = self.image_data
        if new_digest == e.base:
            return None                    # annotations untouched: never rewrite the file (not even for policy)
        shapes = copy.deepcopy(shapes)
        flags_matter = self.flags_matter
        version = self.source_version or "5"
        ann_path = Path(sample.ann_path) if sample is not None and sample.ann_path else None
        session = self

        def apply(ann):
            if session.only_labels is not None:
                merged, hidden = session.merge_hidden(ann.shapes, session.hidden_of(e, ann.shapes), shapes)
                ann.shapes = merged
                session._pending_hidden[e.key] = hidden
            else:
                ann.shapes = shapes
            if isinstance(flags, dict):
                ann.raw["flags"] = flags
                if flags_matter:              # classification adapters write the flags from the tags
                    ann.tags = [str(k) for k, v in flags.items() if v is True]
            img = Path(ann.image_path) if getattr(ann, "image_path", None) else None
            if (img is None or not img.is_file()) and ann_path is not None:
                img = image_file_of(ann_path, ann.raw)
            has_file = img is not None and img.is_file()
            if policy == "none":
                if has_file:                               # never drop the ONLY copy of an image
                    ann.raw[IMG_KEY] = None                # key kept: labelme 6 requires it
            elif policy == "embed" and not ann.raw.get(IMG_KEY) and has_file:
                ann.raw[IMG_KEY] = encode_b64(img, version)
            if IMG_KEY not in ann.raw:
                ann.raw[IMG_KEY] = None
            return True

        before = len(e.base)
        return Update(e, apply, f"{before} → {len(shapes)} shape(s)" if before != len(shapes) else "shapes edited")

    def _yolo_new_shapes(self, e: Entry) -> list[dict]:
        """Edited shapes, normalised, converted to what this YOLO dataset can store."""
        out, notes = [], []
        if self.editor == LABELIMG:
            try:
                text = (self.work / e.mirror).read_text(encoding="utf-8")
                editor_names = (self.work / "classes.txt").read_text(encoding="utf-8").strip("\n").split("\n")
            except OSError as exc:
                raise ValueError(f"could not read the edited file yet ({exc})")
            for ln, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                except (ValueError, IndexError):
                    raise Rejected(f"line {ln} is not a YOLO box")
                if not 0 <= cid < len(editor_names):
                    raise Rejected(f"line {ln}: class id {cid} is unknown")
                label = editor_names[cid]
                out.append({"label": label, "shape_type": "rectangle",
                            "points": [[cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2]]})
            self._check_labels([s["label"] for s in out])
            return out
        try:
            data = load_json(self.work / e.mirror)
        except Exception as exc:
            raise ValueError(f"could not read the edited file yet ({exc})")
        W, H = float(data.get("imageWidth") or 0), float(data.get("imageHeight") or 0)
        if not (W and H):
            raise Rejected("the edited file has no image size")
        task = self.task
        for i, s in enumerate(data.get("shapes") or []):
            st = s.get("shape_type") or "polygon"
            pts = [[float(p[0]) / W, float(p[1]) / H] for p in s.get("points") or []]
            label = s.get("label")
            if st == "rectangle" and len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                if task in ("segment", "obb"):
                    out.append({"label": label, "shape_type": "polygon",
                                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
                else:
                    out.append({"label": label, "shape_type": "rectangle", "points": [[x1, y1], [x2, y2]]})
            elif st == "polygon" and len(pts) >= 3:
                if task == "detect":
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    out.append({"label": label, "shape_type": "rectangle",
                                "points": [[min(xs), min(ys)], [max(xs), max(ys)]]})
                    notes.append("polygon → box")
                elif task == "obb" and len(pts) != 4:
                    raise Rejected(f"shape {i}: oriented boxes need exactly 4 points (has {len(pts)})")
                else:
                    out.append({"label": label, "shape_type": "polygon", "points": pts})
            else:
                raise Rejected(f"shape {i}: a {st} cannot be stored in a YOLO {task} dataset "
                               "(use rectangles" + (" or polygons)" if task != "detect" else ")"))
        for s in out:
            s["points"] = [[min(max(v, 0.0), 1.0) for v in p] for p in s["points"]]    # clamp into the image
        self._check_labels([s["label"] for s in out])
        e.message = "; ".join(sorted(set(notes)))
        return out

    def _update_yolo(self, e: Entry, adapter, sample) -> Update | None:
        new = self._yolo_new_shapes(e)
        digest = _digest_yolo(new)
        if digest == e.base:
            return None

        def apply(ann):
            hid = set(self.hidden_of(e, ann.shapes))
            idx = [i for i, s in enumerate(ann.shapes) if self.editable(s)]
            editable = [ann.shapes[i] for i in idx]
            hid_in_editable = [j for j, i in enumerate(idx) if i in hid]
            originals = [s for j, s in enumerate(editable) if j not in set(hid_in_editable)]
            invalid = [s for s in ann.shapes if not self.editable(s)]
            used, result = set(), []
            for s in new:
                match = None
                for j, o in enumerate(originals):          # same geometry -> keep the original row (lossless)
                    if j in used or o["shape_type"] != s["shape_type"] or len(o["points"]) != len(s["points"]):
                        continue
                    if all(abs(a[0] - b[0]) < 2e-5 and abs(a[1] - b[1]) < 2e-5 for a, b in zip(o["points"], s["points"])):
                        match = j
                        break
                if match is not None:
                    used.add(match)
                    o = dict(originals[match])
                    o["label"] = s["label"]                # relabel keeps the original geometry tokens
                    result.append(o)
                else:
                    result.append({"label": s["label"], "points": s["points"], "shape_type": s["shape_type"],
                                   "group_id": None, "flags": {}, "_row": None, "_class_id": None,
                                   "_geom_dirty": True, "_new": True})
            if self.only_labels is not None:       # hidden labels keep their place, the edited block goes in between
                result, self._pending_hidden[e.key] = self.merge_hidden(editable, hid_in_editable, result)
            ann.shapes = result + invalid                  # invalid lines stay (Validation reports them)
            return True

        before = len(e.base)
        return Update(e, apply, f"{before} → {len(new)} shape(s)" if before != len(new) else "shapes edited")

    # ------------------------------------------------------------------ write bookkeeping
    def backup_original(self, e: Entry, sample):
        """Store the original once per session, before the first write-back (compressed, imageData by reference)."""
        if e.backup is not None or sample.ann_path is None or not Path(sample.ann_path).is_file():
            return
        store = BackupStore(self.root)
        try:
            if self.backup_op is None:
                self.backup_op = store.begin("edit", f"Fix in {self.source_title} · {self.work.name}")
            store.save(self.backup_op, sample.ann_path)
        finally:
            store.close()
        e.backup = "db"
        self.save()

    def after_write(self, e: Entry, sample, summary: str):
        if e.key in self._pending_hidden:
            e.hidden, e.hidden_stale = self._pending_hidden.pop(e.key), False
        e.orig_sig = _sig(Path(sample.ann_path))
        e.status, e.writes = "fixed", e.writes + 1
        e.message = summary + (f" · {e.message}" if e.message and e.message not in summary else "")
        self.mark_seen(e)
        if self.dataset_type == "labelme":
            try:
                d = load_json(sample.ann_path)
                hid = set(e.hidden or []) if self.only_labels is not None else set()
                shapes = [x for i, x in enumerate(d.get("shapes") or []) if i not in hid]
                e.base = _digest_labelme(shapes, (d.get("flags") or {}) if self.flags_matter else None)
            except Exception:
                pass
        else:
            e.base = _digest_yolo(self._yolo_new_shapes(e))
        self.save()

    def resolve_conflict(self, e: Entry, sample, use_editor: bool):
        """use_editor=True: accept the outside change as the new base and write the editor's version on the next
        sync. False: keep the file as it is now and ignore the editor's copy until it is saved again."""
        e.orig_sig = _sig(Path(sample.ann_path)) if sample.ann_path else None
        e.hidden_stale = self.only_labels is not None      # the hidden shapes are re-read from the new original
        if use_editor:
            e.last_mirror_sig = None            # forces a re-sync of the current editor copy
            e.pending_sig = None
            e.status, e.message = "waiting", "conflict resolved: the editor's version will be written"
        else:
            self.mark_seen(e)
            e.status, e.message = "unchanged", "conflict resolved: kept the file as it is"
        self.save()

    def undo(self) -> list[tuple[str, list | None]]:
        """Restore every written file (backup copy, or remove a file the session created). Returns
        [(key, review status before the session)] for the caller to restore review marks and refresh."""
        restored = []
        store = BackupStore(self.root)
        legacy_op = None
        for e, backup in self.undo_plan():
            target = self._target_of(e)
            if target is None:
                continue
            if e.backup == "db" and self.backup_op is not None:
                store.restore(self.backup_op, target)
            elif backup is not None and backup.is_file():
                tmp = target.with_name(target.name + ".aioqc_undo")
                shutil.copy2(backup, tmp)
                os.replace(tmp, target)
            elif backup is not None:                    # legacy folder already compacted into backups.sqlite
                if legacy_op is None:
                    legacy_op = next((o["id"] for o in store.ops() if o["title"] == self.backup_dir.name), -1)
                if legacy_op < 0 or store.restore(legacy_op, target) == "missing":
                    continue
            elif not e.orig_existed and target.is_file():
                target.unlink()
            e.writes, e.status, e.message = 0, "waiting", "undone"
            e.hidden_stale = self.only_labels is not None
            e.orig_sig = _sig(target)
            restored.append((e.key, e.prev_review))
        store.close()
        self.save()
        return restored

    def _target_of(self, e: Entry) -> Path | None:
        return self._targets.get(e.key) if hasattr(self, "_targets") else None

    def bind_targets(self, samples_by_key: dict):
        """Annotation path per entry (needed by undo)."""
        self._targets = {k: Path(s.ann_path) for k, s in samples_by_key.items() if getattr(s, "ann_path", None)}

    def undo_plan(self) -> list[tuple[Entry, Path | None]]:
        """(entry, backup file or None = the file did not exist before) for every written entry."""
        out = []
        for e in self.entries:
            if e.writes:
                out.append((e, None if not e.backup or e.backup == "db" else self.backup_dir / e.backup))
        return out

    def counts(self) -> dict:
        c = {k: 0 for k in STATUS}
        for e in self.entries:
            c[e.status] = c.get(e.status, 0) + 1
        return c

    def backed_up(self) -> int:
        return sum(1 for e in self.entries if e.backup)

    def finish(self, remove_work: bool = True):
        """End the session. The work copies (and the editor's own files) are deleted; the slim manifest stays for
        history until the storage retention removes it."""
        self.active = False
        for e in self.entries:                          # the manifest does not need the shape digests any more
            e.base = []
        self.save()
        if remove_work:
            for p in list(self.work.iterdir()):
                if p.name == "session.json":
                    continue
                try:
                    shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
                except OSError:
                    pass
