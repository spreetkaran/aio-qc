# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Session = one opened dataset. The single place where annotations are written,
so the viewer, the label index and every QC dialog stay in sync.
"""
from __future__ import annotations

import os

import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QProgressDialog, QWidget

from AIO_QC.core import Annotation, AnnotationError, DatasetType, Sample, Trash
from AIO_QC.core.cache import DatasetCache
from AIO_QC.core.mapping import MappedAdapter
from AIO_QC.core.state import ReviewStore, load_label_map, save_label_map
from AIO_QC.core.diag import ignored as _ignored


class Session(QObject):
    annotationsChanged = pyqtSignal(list)   # keys whose annotation content changed
    bulkChanged = pyqtSignal(object, str, int)   # (Backup, what, files) after a bulk change that can be undone
    trashed = pyqtSignal(object)                 # a trash record (untrash() puts it back) after files were moved
    samplesRemoved = pyqtSignal(list)       # keys no longer in the dataset
    samplesAdded = pyqtSignal(list)         # keys that appeared (live updates)
    ownPaths = pyqtSignal(list)             # files AIO QC itself moved / wrote outside the normal save path
    issuesChanged = pyqtSignal()            # a tool published findings (Issues Inbox)
    indexChanged = pyqtSignal()

    mappingChanged = pyqtSignal()
    reviewChanged = pyqtSignal(list)        # keys
    classUndoChanged = pyqtSignal()         # Change class: undo / redo available
    keysRenamed = pyqtSignal(dict)          # {old key: new key} — images moved / renamed by AIO QC (class folders)

    def __init__(self, root: Path, adapter, user_classes: list[str] | None = None):
        super().__init__()
        self.locked: dict[str, str] = {}      # key -> editor title, while open in an editor session
        self.class_undo: list[dict] = []      # Change class steps (classification datasets)
        self.class_redo: list[dict] = []
        self.root = Path(root)
        self.base_adapter = adapter                      # real labels (writes, class removal)
        self.label_map: dict[str, str] = load_label_map(self.root)
        self.adapter = MappedAdapter(adapter, self.label_map)   # what the viewer and QC checks see
        self.samples: list[Sample] = adapter.discover()
        self.by_key: dict[str, Sample] = {s.key: s for s in self.samples}
        self.review = ReviewStore(self.root)
        try:
            self.cache: DatasetCache | None = DatasetCache(self.root)
        except Exception as exc:                 # read-only disk, locked DB, ... -> run uncached
            logger.warning(f"cache disabled: {exc}")
            self.cache = None
        self.raw_index: dict[str, Counter] = {}    # key -> ORIGINAL label counts
        self.index: dict[str, Counter] = {}        # key -> label counts (after view mapping)
        self._count_pool: dict[tuple, Counter] = {}   # distinct label counts -> the one shared Counter
        self.index_errors: dict[str, str] = {}     # key -> parse error
        self.index_complete = False
        # ---- running aggregates (O(changed) updates instead of O(dataset) recounts)
        self._tot_raw, self._tot, self._files_raw = Counter(), Counter(), Counter()
        self._rv = Counter(self.review.status(s.key) for s in self.samples)      # status -> files (None = unreviewed)
        self.key_by_path: dict[str, str] = {}
        for s in self.samples:
            self._map_paths(s, True)
        self._order: list | None = None             # discovery-order keys, built on the first live insertion
        from AIO_QC.core.issues import IssueStore
        from AIO_QC.core.state import state_dir
        self.issues = IssueStore(self.root, state_dir(self.root))
        self.last_inserted: list = []               # [(row, sample)] of the latest add_samples (for views)
        self.user_classes = user_classes
        self.user_classes_path: Path | None = None
        self.trash = Trash(self.root)
        from AIO_QC.core.history import HistoryLog
        self.history = HistoryLog(self.root)
        from AIO_QC.core.guidelines import GuideStore
        self.guides = GuideStore(self.root)          # labelling guidelines + picture examples (tiny when unused)        # per-file audit trail (core.history)
        self.tool = None                            # name of the tool doing the current writes (history)

    # ------------------------------------------------------------------ info
    @property
    def type(self) -> DatasetType:
        return self.adapter.type

    @property
    def is_yolo(self) -> bool:
        return self.adapter.type in (DatasetType.YOLO, DatasetType.YOLO_SPLIT)

    @property
    def true_classes(self) -> list[str] | None:
        """
        Reference classes for ambiguity / unknown-label checks.
        YOLO: dataset.yaml, else classes.txt (found in the dataset or loaded by the user).
        LabelMe: the loaded classes file.
        """
        if self.is_yolo:
            return self.adapter.class_names or None
        return self.user_classes or None

    @property
    def classes_source(self) -> str:
        if self.is_yolo:
            cm = self.adapter.class_map
            if not cm.names:
                return ""
            name = cm.source.name if cm.source else "classes file"
            return {"yaml": name, "txt": f"{name} (found in dataset)", "user": f"{name} (loaded)"}.get(cm.kind, name)
        return "loaded classes file" if self.user_classes else ""

    @property
    def splits(self) -> list[str]:
        """train / val / test present in the dataset (canonical order)."""
        from AIO_QC.core.dataset import SPLIT_ORDER
        return sorted({s.split for s in self.samples if s.split}, key=lambda x: (SPLIT_ORDER.get(x, 9), x))

    @property
    def folders(self) -> list[str]:
        """Source sub-folders (nesting that is not a split)."""
        from AIO_QC.core.io import natural_key
        return sorted({s.folder for s in self.samples if s.folder}, key=natural_key)

    def label_totals(self, raw: bool = False) -> Counter:
        """Instances per label — kept as running totals (O(labels), not O(files))."""
        return +(self._tot_raw if raw else self._tot)

    def raw_label_files(self) -> Counter:
        """original label -> number of files using it (running)"""
        return +self._files_raw

    def keys_with_review(self, statuses, rank=None) -> list[str]:
        """Keys of the files whose review status is in `statuses`, in dataset order (`rank`: key -> position,
        e.g. the file list's row_of). O(reviewed files) — never a pass over a million unreviewed ones."""
        statuses = set(statuses)
        expected = sum(self._rv.get(st, 0) for st in statuses)
        if not expected:
            return []
        rv, by_key = self.review, self.by_key
        out = []
        for rel, e in rv.data.items():
            if e.get("status") not in statuses:
                continue
            for pre in rv._prefixes:                       # the review store keys are root-relative ("a/b.json")
                k = pre + (rel if pre[-1] == "/" else rel.replace("/", pre[-1]))
                if k in by_key:
                    out.append(k)
                    break
        if len(out) != expected:                           # unusual key form: the exact (slow) way
            return [x.key for x in self.samples if rv.status(x.key) in statuses]
        if rank is not None:
            out.sort(key=rank)
        return out

    def review_totals(self) -> dict:
        """{'ok': n, 'flag': n, 'fixed': n, None: unreviewed} for the files of the dataset (running)."""
        return dict(self._rv)

    def _map_paths(self, s, add: bool):
        for p in (s.ann_path, s.image_path):
            if p is not None:
                if add:
                    self.key_by_path[str(p)] = s.key
                elif self.key_by_path.get(str(p)) == s.key:
                    del self.key_by_path[str(p)]

    def _rebuild_totals(self):
        self._tot_raw, self._tot, self._files_raw = Counter(), Counter(), Counter()
        for c in self.raw_index.values():
            self._tot_raw.update(c)
            self._files_raw.update(c.keys())
        for c in self.index.values():
            self._tot.update(c)

    def label_namespace(self) -> str:
        """Cache namespace: YOLO label names depend on the class map; the same JSONs read as classification (their
        flags) count differently from LabelMe shapes."""
        if self.is_yolo:
            return "|".join(self.base_adapter.class_names)
        return "labelme" if self.type.value == "labelme" else self.type.value

    def as_tool(self, name: str):
        """with session.as_tool("Hand-off batch"): … — writes inside are logged under that tool."""
        from contextlib import contextmanager

        @contextmanager
        def cm():
            prev, self.tool = self.tool, name
            try:
                yield
            finally:
                self.tool = prev
        return cm()

    def log(self, keys, action: str, what: str = "", tool: str | None = None, op=None):
        try:
            self.history.log(keys, tool or self.tool or "AIO QC", action, what, op)
        except Exception as exc:                       # the audit trail must never break a write
            logger.warning(f"history: {exc}")

    def close(self):
        try:
            self.history.close()
        except Exception as _exc:
            _ignored(_exc)
        if self.cache is not None:
            try:
                self.cache.close()
            except Exception as _exc:
                _ignored(_exc)
            self.cache = None

    # ---------------------------------------------------------------- mapping
    def set_label_map(self, mapping: dict[str, str]):
        self.label_map = {k: v for k, v in mapping.items() if v and v != k}
        self.adapter = MappedAdapter(self.base_adapter, self.label_map)
        save_label_map(self.root, self.label_map)
        self.index = {k: self._mapped(c) for k, c in self.raw_index.items()}
        self._rebuild_totals()                       # the mapped view changed as a whole (rare)
        self.mappingChanged.emit()
        self.indexChanged.emit()

    def _shared(self, raw: Counter) -> Counter:
        """One Counter object per distinct label count (most files share a handful of patterns, e.g. {car: 1,
        plate: 1}), label names interned: a million-file index takes a fraction of the memory. Index Counters are
        never changed in place (update_index replaces them), so sharing is safe."""
        sig = tuple(sorted(raw.items()))
        hit = self._count_pool.get(sig)
        if hit is None:
            hit = Counter({sys.intern(str(k)): v for k, v in sig})
            self._count_pool[sig] = hit
        return hit

    def _mapped(self, raw: Counter) -> Counter:
        if not self.label_map:                       # no mapping: the mapped view IS the raw counts (no copy)
            return raw
        out = Counter()
        for label, n in raw.items():
            out[self.adapter.map_label(label)] += n
        return out

    # ----------------------------------------------------------------- review
    def set_review(self, keys, status, note=None):
        keys = [k for k in keys if k in self.by_key]
        for k in keys:
            old = self.review.status(k)
            self.review.set(k, status, note)
            self._rv[old] -= 1
            self._rv[status] += 1
        if keys:
            self.log(keys, "review", {None: "mark cleared", "ok": "marked OK", "flag": "flagged",
                                      "fixed": "marked fixed"}.get(status, str(status)) + (f" — {note}" if note else ""))
            self.reviewChanged.emit(keys)

    def keys_with_label(self, label: str) -> list[str]:
        return [k for k, c in self.index.items() if c.get(label)]

    # ------------------------------------------------------------------ index
    def update_index(self, key: str, raw: Counter | None, error: str | None = None):
        """Index stores ORIGINAL labels; the mapped view is derived from them. Totals follow incrementally."""
        old_raw, old = self.raw_index.get(key), self.index.get(key)
        if old_raw is not None:
            self._tot_raw.subtract(old_raw)
            self._files_raw.subtract(old_raw.keys())
        if old is not None:
            self._tot.subtract(old)
        if raw is not None:
            raw = self._shared(raw)
            self.raw_index[key] = raw
            self.index[key] = m = self._mapped(raw)
            self._tot_raw.update(raw)
            self._files_raw.update(raw.keys())
            self._tot.update(m)
        else:
            self.raw_index.pop(key, None)
            self.index.pop(key, None)
        if error:
            self.index_errors[key] = error
        else:
            self.index_errors.pop(key, None)

    # ------------------------------------------------------------------ edits
    def load(self, key: str) -> Annotation:
        return self.adapter.load(self.by_key[key])

    def modify_many(self, parent: QWidget, jobs: Iterable[tuple[str, Callable[[Annotation], bool]]],
                    title: str = "Applying changes...", raw: bool = False) -> tuple[list[str], list[str]]:
        """
        jobs: (key, fn(annotation) -> changed?). Each file is loaded fresh, edited,
        and atomically written. raw=True edits the real labels (ignores the view mapping).
        Returns (changed_keys, errors).
        """
        adapter = self.base_adapter if raw else self.adapter
        jobs = list(jobs)
        changed, errors = [], []
        locked = [k for k, _fn in jobs if k in self.locked]
        if locked:                                       # open in an external editor: one writer at a time
            errors += [f"{self.by_key[k].name if k in self.by_key else k}: open in {self.locked[k]} — "
                       "edit it there, or end the editor session first" for k in locked]
            jobs = [(k, fn) for k, fn in jobs if k not in self.locked]
        dlg = QProgressDialog(title, "Cancel", 0, len(jobs), parent)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)
        import contextlib
        from AIO_QC.core import provenance
        with (self.base_adapter.batch() if hasattr(self.base_adapter, "batch") else contextlib.nullcontext()), \
                provenance.unless_active("rule", self.tool or title.rstrip(". …").strip()):
            self._modify_loop(jobs, adapter, dlg, changed, errors)    # (COCO: each file written once)
        dlg.setValue(len(jobs))
        if changed:
            what = title.rstrip(". …").strip()
            self.log(changed, "edit", what, tool=self.tool or what)
            self.annotationsChanged.emit(changed)
            self.indexChanged.emit()
        return changed, errors

    def _modify_loop(self, jobs, adapter, dlg, changed, errors):
        for i, (key, fn) in enumerate(jobs):
            if dlg.wasCanceled():
                errors.append("Cancelled by user; remaining files untouched.")
                break
            sample = self.by_key.get(key)
            if sample is None:
                continue
            try:
                ann = adapter.load(sample)
                if fn(ann):
                    adapter.save(sample, ann)            # MappedAdapter.save restores original labels
                    self.update_index(key, ann.label_counts())
                    changed.append(key)
            except (AnnotationError, ValueError, OSError) as exc:
                errors.append(f"{sample.name}: {exc}")
                logger.warning(f"modify failed {key}: {exc}")
            if i % 10 == 0:
                dlg.setValue(i)
                QApplication.processEvents()

    # ------------------------------------------------------------------ dataset membership (trash, live updates)
    def forget_samples(self, keys: list[str]):
        """Samples that are gone (trashed here, or deleted outside AIO QC)."""
        rs = {k for k in keys if k in self.by_key}
        if not rs:
            return
        if len(rs) <= 64:                           # a few (the keyboard flow): each found by binary search in the
            import bisect                           # samples (sorted by order key) + a memmove, not two rebuilt lists
            ok = self.base_adapter.order_key
            for k in rs:
                smp = self.by_key[k]
                if self._order is not None:
                    i = bisect.bisect_left(self._order, ok(smp))
                else:                               # O(log n) order keys computed, never all n
                    i = bisect.bisect_left(self.samples, ok(smp), key=ok)
                if not (i < len(self.samples) and self.samples[i] is smp):
                    i = next((j for j, x in enumerate(self.samples) if x is smp), -1)
                if i < 0:
                    continue
                del self.samples[i]
                if self._order is not None:
                    del self._order[i]
        else:
            if self._order is not None:
                self._order = [o for o, s in zip(self._order, self.samples) if s.key not in rs]
            self.samples = [s for s in self.samples if s.key not in rs]
        for k in rs:
            self.update_index(k, None)              # totals follow
            self._rv[self.review.status(k)] -= 1
            self._map_paths(self.by_key[k], False)
            self.by_key.pop(k, None)
            self.index_errors.pop(k, None)
        self.issues.forget(rs)
        self.samplesRemoved.emit(sorted(rs))
        self.indexChanged.emit()

    def add_samples(self, samples) -> list[str]:
        """Files that appeared outside AIO QC: index them (light load) and announce them."""
        import bisect
        new = [s for s in samples if s.key not in self.by_key]
        ok = self.base_adapter.order_key
        if len(new) > 64 and self._order is None:    # many: discovery-order keys, once (O(n)), then O(log n) each
            self._order = [ok(x) for x in self.samples]
        # ascending key order: every insertion lands at or after the previous one, so the index at insertion time is
        # already the FINAL row — views replay these inserts in the same order and end up identical
        self.last_inserted = []
        for s in sorted(new, key=ok):
            k = ok(s)
            if self._order is not None:
                i = bisect.bisect_right(self._order, k)
                self._order.insert(i, k)
            else:                                    # a few (undo, a file appeared): O(log n) keys, not all n
                i = bisect.bisect_right(self.samples, k, key=ok)
            self.samples.insert(i, s)
            self.last_inserted.append((i, s))
        for s in new:
            self.by_key[s.key] = s
            self._map_paths(s, True)
            self._rv[self.review.status(s.key)] += 1
            try:
                ann = self.base_adapter.load(s, light=True)
                self.update_index(s.key, ann.label_counts())
            except (AnnotationError, ValueError, OSError) as exc:
                self.update_index(s.key, None, str(exc))
        keys = [s.key for s in new]
        if keys:
            self.samplesAdded.emit(keys)
            self.indexChanged.emit()
        return keys

    def paths_of(self, keys) -> list[str]:
        """Annotation + image paths of these samples (own-write bookkeeping for the dataset watcher)."""
        out = []
        for k in keys:
            s = self.by_key.get(k)
            if s is not None:
                out += [str(p) for p in (s.ann_path, s.image_path) if p is not None]
        return out

    # ------------------------------------------------------------------ classification
    @property
    def is_classification(self) -> bool:
        return self.type.is_classification

    def class_of(self, key: str) -> str:
        c = self.index.get(key)
        return next(iter(c), "") if c else ""

    def tags_of(self, key: str) -> list[str]:
        c = self.index.get(key)
        return sorted(c, key=str.lower) if c else []

    # settings: single- / multi-label, your class order
    def cls_settings(self) -> dict:
        if getattr(self, "_cls_settings", None) is None:
            from AIO_QC.core.state import load_cls_settings
            self._cls_settings = load_cls_settings(self.root) if self.is_classification else {}
        return self._cls_settings

    def save_cls_settings(self, **kw):
        from AIO_QC.core.state import save_cls_settings
        d = dict(self.cls_settings(), **kw)
        self._cls_settings = d
        save_cls_settings(self.root, d)

    @property
    def cls_mode(self) -> str:
        """single (one class per image) | multi (any number). Class folders are always single-label."""
        if self.type.value == "cls_folders":
            return "single"
        return self.cls_settings().get("mode") or "single"

    def multi_label_count(self) -> int:
        return sum(1 for c in self.index.values() if len(c) > 1)

    def cls_classes(self) -> list[str]:
        """The class list in your order (Classes ▸ order), then the rest alphabetically."""
        names = set(self.base_adapter.class_names) | {k for c in self.index.values() for k in c} | \
            set(self.true_classes or [])
        order = [n for n in self.cls_settings().get("order") or [] if n in names]
        return order + sorted(names - set(order), key=lambda n: (n.lower(), n))

    def set_class(self, keys: list[str], new_class: str) -> tuple[list[str], list[str]]:
        """Every image gets exactly this class (LabelMe flags rewritten / class folders: the image MOVES)."""
        return self.set_tags(keys, tags=[new_class], what=f"class → {new_class}")

    def set_tags(self, keys, tags=None, add=(), remove=(), what: str = "", record: bool = True, force: bool = False,
                 drop=()):
        """Change images' classes: tags = the full new set, or add / remove some (multi-label). LabelMe flags: the
        flags are rewritten (a backup kept, File History lists it; `drop`: flag names removed from the files, for a
        rename / removal); class folders: the image moves to its class folder (review mark, issues, cached results
        and backups follow). One step of Undo. -> (new keys, errors)"""
        from AIO_QC.core.backups import BackupStore
        folders = self.type.value == "cls_folders"
        done, errors, moved, undo, pairs = [], [], [], [], []
        store, op = None, None
        for k in list(keys):
            smp = self.by_key.get(k)
            if smp is None:
                continue
            if k in self.locked:
                errors.append(f"{smp.name}: open in {self.locked[k]}")
                continue
            old = self.tags_of(k)
            if tags is not None:
                new = [t for t in dict.fromkeys(tags) if t]
            else:
                new = [t for t in old if t not in set(remove)] + [t for t in add if t not in old]
            if folders:
                if len(new) != 1:
                    errors.append(f"{smp.name}: an image in a class folder has exactly one class")
                    continue
                if old == new:
                    continue
                try:
                    nw = self.base_adapter.set_class(smp, new[0])
                except (AnnotationError, OSError) as exc:
                    errors.append(f"{smp.name}: {exc}")
                    continue
                moved.append((smp, nw))
                pairs.append((Path(smp.image_path), Path(nw.image_path)))
                undo.append({"key": nw.key, "back": old[0] if old else "", "fwd": new[0], "old_key": smp.key})
                continue
            if sorted(old) == sorted(new) and not force and not drop:
                continue
            try:
                if store is None:
                    store = BackupStore(self.root)
                    op = store.begin("class", what or "Change class")
                store.save(op, smp.ann_path)
                prev = self.base_adapter.flags_of(smp)
                self.base_adapter.set_tags(smp, new, drop=drop)
                after = self.base_adapter.flags_of(smp)
            except (AnnotationError, OSError, ValueError) as exc:
                errors.append(f"{smp.name}: {exc}")
                continue
            self.update_index(k, Counter({t: 1 for t in new}))
            done.append(k)
            undo.append({"key": k, "flags_before": prev, "flags_after": after})
        if store is not None:
            store.close()
        if drop and hasattr(self.base_adapter, "forget_names"):
            self.base_adapter.forget_names([d for d in drop if not any(d in self.tags_of(k) for k in self.by_key)])
        if moved:
            self._carry_over(moved, pairs)
            done += [n.key for _o, n in moved]
        if done:
            self.log(done, "class", what or "class changed", tool=self.tool or "Change class")
            self.annotationsChanged.emit(done)
            self.indexChanged.emit()
            if record:
                self.class_undo.append({"what": what or "class change", "items": undo, "folders": folders})
                self.class_redo.clear()
                self.classUndoChanged.emit()
        return done, errors

    def _class_step(self, rec, back: bool):
        """Undo (back) or redo one recorded class change."""
        done, errors = [], []
        if rec.get("kind") == "issues":
            return self._issues_step(rec, back)
        if rec["folders"]:
            by_cls = {}
            for it in rec["items"]:
                key = it["key"] if back else it["old_key"]
                target = it["back"] if back else it["fwd"]
                if key in self.by_key and target:
                    by_cls.setdefault(target, []).append(key)
            for cls, keys in by_cls.items():
                d, e = self.set_tags(keys, tags=[cls], what=rec["what"], record=False)
                done += d
                errors += e
            return done, errors
        for it in rec["items"]:
            smp = self.by_key.get(it["key"])
            if smp is None:
                continue
            try:
                self.base_adapter.write_flags(smp, it["flags_before" if back else "flags_after"])
                ann = self.base_adapter.load(smp, light=True)
                self.update_index(it["key"], ann.label_counts())
                done.append(it["key"])
            except (AnnotationError, OSError, ValueError) as exc:
                errors.append(f"{smp.name}: {exc}")
        if done:
            self.log(done, "undo" if back else "class", ("undone: " if back else "redone: ") + rec["what"],
                     tool="Change class")
            self.annotationsChanged.emit(done)
            self.indexChanged.emit()
        return done, errors

    def undo_class(self):
        if not self.class_undo:
            return [], []
        rec = self.class_undo.pop()
        done, errors = self._class_step(rec, True)
        if rec["folders"] and rec.get("kind") != "issues":   # the keys changed back: the redo needs the current ones
            self._refresh_move_keys(rec, back=True)
        self.class_redo.append(rec)
        self.classUndoChanged.emit()
        return done, errors

    def redo_class(self):
        if not self.class_redo:
            return [], []
        rec = self.class_redo.pop()
        done, errors = self._class_step(rec, False)
        if rec["folders"] and rec.get("kind") != "issues":
            self._refresh_move_keys(rec, back=False)
        self.class_undo.append(rec)
        self.classUndoChanged.emit()
        return done, errors

    # ------------------------------------------------------------------ issues folder (class folders)
    def send_to_issues(self, keys, folder, note: str = "", record: bool = True) -> tuple[list[str], list[str]]:
        """Class folders: move images to the issues folder, kept by their class (and split) — they leave the
        dataset; one step of Undo puts them back. -> (keys moved, errors). See core.issues_folder."""
        from AIO_QC.core import issues_folder as IF
        if self.type.value != "cls_folders":
            return [], ["the issues folder is for class-folder datasets"]
        folder = IF.ensure(Path(folder), self.root)
        items, errors, rows, taken = [], [], [], set()
        for k in dict.fromkeys(keys):
            smp = self.by_key.get(k)
            if smp is None or smp.image_path is None:
                continue
            if k in self.locked:
                errors.append(f"{smp.name}: open in {self.locked[k]}")
                continue
            tags = self.tags_of(k)
            cls = tags[0] if tags else ""
            dst = IF.destination(folder, cls, smp.split if self._has_splits() else "", smp.name, taken)
            try:
                IF.move(smp.image_path, dst)
            except OSError as exc:
                errors.append(f"{smp.name}: {exc}")
                continue
            taken.add(str(dst))
            items.append({"key": k, "src": str(smp.image_path), "dst": str(dst), "sample": smp, "class": cls})
            rows.append(("to issues", smp.image_path, dst, cls, smp.split, note, self.tool or "Send to issues"))
        self._last_issues = items
        if not items:
            return [], errors
        moved = [it["key"] for it in items]
        self.ownPaths.emit([p for it in items for p in (it["src"], it["dst"])])
        self.forget_samples(moved)
        IF.log(folder, rows, self.root)
        self.log(moved, "issues", f"set aside in {folder.name}" + (f" — {note}" if note else ""),
                 tool=self.tool or "Send to issues")
        if record:
            n = len(items)
            self.class_undo.append({"what": f"{n:,} image{'s' if n != 1 else ''} to issues", "kind": "issues",
                                    "items": items, "folders": True, "folder": str(folder), "note": note})
            self.class_redo.clear()
            self.classUndoChanged.emit()
        return moved, errors

    def _has_splits(self) -> bool:
        return any(sp for sp, _b in getattr(self.base_adapter, "bases", []) or [])

    def _issues_step(self, rec, back: bool):
        """Undo: the images come back from the issues folder (where nothing new took their place). Redo: set aside
        again."""
        from AIO_QC.core import issues_folder as IF
        folder = Path(rec["folder"])
        if not back:
            keys = [it["key"] for it in rec["items"] if it["key"] in self.by_key]
            moved, errors = self.send_to_issues(keys, folder, rec.get("note", ""), record=False)
            now = {it["key"]: it["dst"] for it in self._last_issues}     # where each one is now (maybe _1)
            for it in rec["items"]:
                if it["key"] in now:
                    it["dst"] = now[it["key"]]
            return moved, errors
        samples, errors, rows = [], [], []
        for it in rec["items"]:
            src, dst = Path(it["src"]), Path(it["dst"])
            if not dst.exists():
                errors.append(f"{src.name}: no longer in the issues folder")
                continue
            if src.exists():
                errors.append(f"{src.name}: a file of that name is back in its class folder")
                continue
            try:
                IF.move(dst, src)
            except OSError as exc:
                errors.append(f"{src.name}: {exc}")
                continue
            samples.append(it["sample"])
            rows.append(("back", dst, src, it.get("class", ""), it["sample"].split, "undo", "Undo"))
        if samples:
            self.ownPaths.emit([p for r in rows for p in (str(r[1]), str(r[2]))])
            self.add_samples(samples)
            IF.log(folder, rows, self.root)
            IF.prune_empty(folder)
            self.log([s.key for s in samples], "undo", "put back from the issues folder", tool="Undo")
        return [s.key for s in samples], errors

    def return_from_issues(self, folder, images=None) -> tuple[list[str], list[str], int]:
        """Put images from the issues folder back into the dataset — into the class folder each is in NOW (a class
        fixed there is fixed on the way back). images: those paths only (None: all). -> (keys, errors, skipped)"""
        from AIO_QC.core import issues_folder as IF
        from AIO_QC.core.dataset import Sample
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        folder = Path(folder)
        imgs = [Path(p) for p in images] if images is not None else IF.contents(folder, IMAGE_EXTENSIONS)
        bases = {sp: Path(b) for sp, b in (getattr(self.base_adapter, "bases", None) or [("", self.root)])}
        plan = IF.return_plan(folder, imgs, bases, self.root)
        skipped = len(imgs) - len(plan)
        new, errors, rows = [], [], []
        for img, dst, cls, split in plan:
            try:
                IF.move(img, dst)
            except OSError as exc:
                errors.append(f"{img.name}: {exc}")
                continue
            smp, _reload = self.base_adapter.sample_for_path(dst)
            if smp is None:
                smp = Sample(key=str(dst), name=dst.name, ann_path=None, image_path=dst, split=split)
            new.append(smp)
            rows.append(("back", img, dst, cls, split, "", self.tool or "Put back"))
        if new:
            self.ownPaths.emit([p for r in rows for p in (str(r[1]), str(r[2]))])
            for c in {r[3] for r in rows if r[3]} - set(self.base_adapter.class_names):
                self.base_adapter._classes = sorted(set(self.base_adapter._classes) | {c}, key=str.lower)
            keys = self.add_samples(new)
            IF.log(folder, rows, self.root)
            IF.prune_empty(folder)
            self.log(keys, "issues", "put back from the issues folder", tool=self.tool or "Put back")
            return keys, errors, skipped
        return [], errors, skipped

    def _refresh_move_keys(self, rec, back: bool):
        """After moving images back / forth, where each one is now (keys are paths)."""
        for it in rec["items"]:
            old_key, new_key = it["old_key"], it["key"]
            cur = old_key if back else new_key
            if cur not in self.by_key:              # moved under another name (_1): find it by its folder
                name = Path(cur).name
                hit = next((k for k in self.by_key if Path(k).name == name and
                            Path(k).parent.name == (it["back"] if back else it["fwd"])), None)
                if hit:
                    if back:
                        it["old_key"] = hit
                    else:
                        it["key"] = hit

    # ------------------------------------------------------------------ split moves
    def move_to_split(self, moves: list[tuple[str, str]], image_of=None) -> tuple[list[str], list[str], list]:
        """Move files to another split folder (dataset layout kept). moves = [(key, target split)].
        Returns (new keys, errors, log) — log = [{"key": new key, "pairs": [[src, dst], …]}] for undo."""
        from AIO_QC.core import split_health as SH
        from AIO_QC.core.dataset import Sample
        if self.shared_ann:
            return [], ["COCO: an image's split is the file it is in — moving it between files is not done here "
                        "(Convert & Split writes a new split)"], []
        dirs = SH.split_dirs(self.root, self.samples)
        spelled = {}                                      # normalised split -> how this dataset's samples spell it
        for x in self.samples:
            spelled.setdefault(SH.norm(x.split), x.split)
        plans, errors = [], []
        for k, dst in moves:
            smp = self.by_key.get(k)
            if smp is None or k in self.locked:
                continue
            img = image_of(smp) if image_of is not None else smp.image_path
            try:
                plans.append((smp, dst, img, SH.plan_paths(self.root, smp, img, dst, dirs)))
            except (ValueError, OSError) as exc:
                errors.append(f"{smp.display}: {exc}")
        moved, log = [], []
        for smp, dst, img, pairs in plans:
            done = []
            try:
                for a, b in pairs:
                    b.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(a, b)
                    done.append((a, b))
            except OSError as exc:
                for a, b in reversed(done):             # all or nothing per file
                    try:
                        os.replace(b, a)
                    except OSError:
                        pass
                errors.append(f"{smp.display}: {exc}")
                continue
            new_of = {str(a): b for a, b in pairs}
            name = dirs[SH.norm(dst)]
            ann = new_of.get(str(smp.ann_path)) if smp.ann_path is not None else None
            if ann is None and smp.ann_path is not None:
                ann = SH.swap_split(self.root, smp.ann_path, smp.split, name) or smp.ann_path
            new_img = new_of.get(str(img)) if img is not None else None
            if ann is not None and str(ann).lower().endswith(".json") and img is not None:
                SH.fix_image_path(Path(ann), Path(new_img or img))
            main = Path(new_of.get(str(smp.key)) or ann or new_img)
            new = Sample(key=str(main), name=main.name, ann_path=Path(ann) if ann is not None else None,
                         image_path=Path(new_img) if smp.image_path is not None and new_img is not None else
                         smp.image_path, split=spelled.get(SH.norm(dst), dst), folder=smp.folder)
            moved.append((smp, new))
            log.append({"key": new.key, "old": smp.key, "split": smp.split,
                        "pairs": [[str(a), str(b)] for a, b in pairs]})
        if moved:
            review = {old.key: self.review.get(old.key) for old, _n in moved}
            for old, new in moved:
                self.issues.rename(old.key, new.key)
            self.forget_samples([old.key for old, _n in moved])
            for old, new in moved:
                e = review.get(old.key)
                self.review.set(old.key, None)
                if e:
                    self.review.set(new.key, e["status"], e.get("note"))
            self.add_samples([n for _o, n in moved])
            self.ownPaths.emit([p for e in log for pr in e["pairs"] for p in pr])
            for old, new in moved:
                self.log([new.key], "move", f"moved {old.split or '—'} → {new.split}", tool=self.tool or "Split health")
            self.indexChanged.emit()
        return [n.key for _o, n in moved], errors, log

    def undo_split_moves(self, log: list) -> tuple[list[str], list[str]]:
        """Put files moved by move_to_split back (the log it returned)."""
        from AIO_QC.core.dataset import Sample
        moved, errors = [], []
        for e in log:
            smp = self.by_key.get(e["key"])
            if smp is None:
                errors.append(f"{Path(e['key']).name}: no longer in the dataset")
                continue
            try:
                for a, b in reversed(e["pairs"]):
                    if Path(a).exists():
                        raise OSError(f"{Path(a).name} exists again at the old place")
                for a, b in reversed(e["pairs"]):
                    Path(a).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(b, a)
            except OSError as exc:
                errors.append(f"{smp.display}: {exc}")
                continue
            back = {b: a for a, b in e["pairs"]}
            ann = back.get(str(smp.ann_path), str(smp.ann_path)) if smp.ann_path is not None else None
            img = back.get(str(smp.image_path), str(smp.image_path)) if smp.image_path is not None else None
            if ann is not None and ann.lower().endswith(".json") and img is not None:
                from AIO_QC.core.split_health import fix_image_path
                fix_image_path(Path(ann), Path(img))
            old = Path(e["old"])
            moved.append((smp, Sample(key=str(old), name=old.name, ann_path=Path(ann) if ann else None,
                                      image_path=Path(img) if img else None, split=e["split"], folder=smp.folder)))
        if moved:
            review = {o.key: self.review.get(o.key) for o, _n in moved}
            for o, n in moved:
                self.issues.rename(o.key, n.key)
            self.forget_samples([o.key for o, _n in moved])
            for o, n in moved:
                r = review.get(o.key)
                self.review.set(o.key, None)
                if r:
                    self.review.set(n.key, r["status"], r.get("note"))
            self.add_samples([n for _o, n in moved])
            self.ownPaths.emit([p for e in log for pr in e["pairs"] for p in pr])
            for o, n in moved:
                self.log([n.key], "move", f"moved back {o.split} → {n.split or '—'}", tool=self.tool or "Split health")
            self.indexChanged.emit()
        return [n.key for _o, n in moved], errors

    # ------------------------------------------------------------------ renames (Name Clashes)
    def _carry_over(self, moved, pairs):
        """Samples replaced by renamed / moved ones: review mark and note, issues, list position, cached results
        and backups follow the new paths."""
        review = {old.key: self.review.get(old.key) for old, _n in moved}
        for old, new in moved:
            self.issues.rename(old.key, new.key)
        self.forget_samples([old.key for old, _n in moved])
        for old, new in moved:
            e = review.get(old.key)
            self.review.set(old.key, None)
            if e:
                self.review.set(new.key, e["status"], e.get("note"))
        self.add_samples([n for _o, n in moved])
        if self.cache is not None:
            try:
                self.cache.rename(pairs)
            except Exception as exc:                       # a cache that cannot follow is rebuilt, never fatal
                logger.warning(f"cache rename: {exc}")
        try:
            from AIO_QC.core.backups import BackupStore
            BackupStore(self.root).rename(pairs)
        except Exception as exc:
            logger.warning(f"backups rename: {exc}")
        self.ownPaths.emit([str(p) for pr in pairs for p in pr])
        self.keysRenamed.emit({old.key: new.key for old, new in moved})

    def _renamed_sample(self, smp, pairs):
        from AIO_QC.core.dataset import Sample
        from AIO_QC.core.split_health import fix_image_path
        new_of = {str(a): b for a, b in pairs}
        ann = new_of.get(str(smp.ann_path), smp.ann_path) if smp.ann_path is not None else None
        img = new_of.get(str(smp.image_path), smp.image_path) if smp.image_path is not None else None
        moved_img = next((b for a, b in pairs if smp.image_path is None and str(a) != str(smp.ann_path)), None)
        if ann is not None and str(ann).lower().endswith(".json") and (img or moved_img) is not None:
            fix_image_path(Path(ann), Path(img or moved_img))
        main = Path(new_of.get(str(smp.key)) or ann or img)
        return Sample(key=str(main), name=main.name, ann_path=Path(ann) if ann is not None else None,
                      image_path=Path(img) if img is not None else None, split=smp.split, folder=smp.folder)

    def rename_samples(self, renames, progress=None, stop=None) -> tuple[list[str], list[str], list]:
        """Give files new names in place: renames = [(key, new stem)]. Image and label file together, imagePath
        fixed, all or nothing per file, nothing overwritten. Returns (new keys, errors, log) — log for undo and
        renamed.csv: [{"key": new, "old": old, "pairs": [[old path, new path], …]}]."""
        from AIO_QC.core import renamer as R
        moved, log, errors, all_pairs = [], [], [], []
        n = len(renames)
        for i, (k, stem) in enumerate(renames):
            if stop is not None and stop():
                break
            if progress is not None and i % 200 == 0:
                progress(i, n)
            smp = self.by_key.get(k)
            if smp is None or k in self.locked:
                continue
            try:
                pairs = R.pairs_for(smp, stem, image_only=self.shared_ann)
                if not pairs:
                    continue
                R.apply(pairs)
            except (ValueError, OSError) as exc:
                errors.append(f"{smp.display}: {exc}")
                continue
            new = self.base_adapter.relocate(smp, pairs[0][1]) if self.shared_ann else \
                self._renamed_sample(smp, pairs)
            moved.append((smp, new))
            all_pairs += pairs
            log.append({"key": new.key, "old": smp.key, "pairs": [[str(a), str(b)] for a, b in pairs]})
        if moved:
            self._carry_over(moved, all_pairs)
            for old, new in moved:
                self.log([new.key], "rename", f"renamed {old.name} → {new.name}", tool=self.tool or "Name Clashes")
            self.indexChanged.emit()
        return [nw.key for _o, nw in moved], errors, log

    def undo_renames(self, log) -> tuple[list[str], list[str]]:
        """Put the names rename_samples gave back (its log)."""
        from AIO_QC.core import renamer as R
        moved, errors, all_pairs = [], [], []
        for e in reversed(log):
            smp = self.by_key.get(e["key"])
            if smp is None:
                errors.append(f"{Path(e['key']).name}: no longer in the dataset")
                continue
            back = [(Path(b), Path(a)) for a, b in e["pairs"]]
            try:
                for b, a in back:
                    if os.path.lexists(a) and os.path.normcase(str(a)) != os.path.normcase(str(b)):
                        raise OSError(f"{a.name} exists again under the old name")
                R.apply(back)
            except OSError as exc:
                errors.append(f"{smp.display}: {exc}")
                continue
            moved.append((smp, self.base_adapter.relocate(smp, back[0][1]) if self.shared_ann else
                          self._renamed_sample(smp, back)))
            all_pairs += back
        if moved:
            self._carry_over(moved, all_pairs)
            for old, new in moved:
                self.log([new.key], "rename", f"name put back {old.name} → {new.name}",
                         tool=self.tool or "Name Clashes")
            self.indexChanged.emit()
        return [nw.key for _o, nw in moved], errors

    # ------------------------------------------------------------------ external editor sessions
    def write_external(self, key: str, fn: Callable[[Annotation], bool]) -> bool:
        """Silent single-file write for editor sync (real labels, atomic, re-indexed, views notified)."""
        sample = self.by_key[key]
        ann = self.base_adapter.load(sample)
        if not fn(ann):
            return False
        from AIO_QC.core import provenance
        with provenance.unless_active("manual", self.tool or "Fix in Editor"):
            self.base_adapter.save(sample, ann)
        self.update_index(key, ann.label_counts())
        self.log([key], "edit", "saved in the external editor, written back", tool=self.tool or "Fix in Editor")
        self.annotationsChanged.emit([key])
        self.indexChanged.emit()
        return True

    def edit_file(self, key: str, fn: Callable[[Annotation], str], expect_sig=None) -> tuple[bytes | None, str]:
        """
        One inline edit: load the file fresh (view mapping applied), check it is unchanged since it was shown
        (`expect_sig` = (mtime_ns, size)), let fn(ann) edit it (it returns a short description or raises
        EditRefused), write it atomically, re-index, notify. -> (bytes before the edit, description).
        Raises EditRefused when the file is locked, changed outside, or the edit is not possible.
        """
        from AIO_QC.core.inline_edit import EditRefused
        if key in self.locked:
            raise EditRefused(f"open in {self.locked[key]} — edit it there, or end that session first")
        sample = self.by_key.get(key)
        if sample is None or sample.ann_path is None:
            raise EditRefused("this file has no annotation file to edit")
        path = Path(sample.ann_path)
        if expect_sig is not None and self.ann_sig(sample) != expect_sig:
            raise EditRefused("the file changed on disk since it was shown — reloaded, please try again")
        if self.shared_ann:
            before = self.base_adapter.unit_bytes(sample)          # COCO: just this image's objects
        else:
            before = path.read_bytes() if path.is_file() else None
        ann = self.adapter.load(sample)
        what = fn(ann)
        from AIO_QC.core import provenance
        with provenance.unless_active("manual", self.tool or "On-image edit"):
            self.adapter.save(sample, ann)
        self.update_index(key, ann.label_counts())
        self.log([key], "edit", what, tool=self.tool or "On-image edit")
        self.annotationsChanged.emit([key])
        self.indexChanged.emit()
        return before, what

    @property
    def shared_ann(self) -> bool:
        """One annotation file for many images (COCO)."""
        return bool(getattr(self.base_adapter, "shared_ann", False))

    def ann_sig(self, sample):
        """What tells "changed since it was shown": the file's signature — for COCO, a hash of the image's part."""
        from AIO_QC.core.inline_edit import file_sig
        if sample is None or sample.ann_path is None:
            return None
        if self.shared_ann:
            try:
                return self.base_adapter.unit_sig(sample)
            except Exception:
                return None
        return file_sig(sample.ann_path)

    def unit_read(self, key: str) -> bytes:
        return self.base_adapter.unit_bytes(self.by_key[key])

    def unit_write(self, key: str, data):
        smp = self.by_key[key]
        self.base_adapter.write_unit(smp, data)
        ann = self.base_adapter.load(smp, light=True)
        self.update_index(key, ann.label_counts())
        self.ownPaths.emit([str(smp.ann_path)])

    def reindex_keys(self, keys: list[str]):
        """Files changed on disk (e.g. undo of an editor session): refresh index + views."""
        keys = [k for k in keys if k in self.by_key]
        for k in keys:
            try:
                ann = self.base_adapter.load(self.by_key[k], light=True)
                self.update_index(k, ann.label_counts())
            except (AnnotationError, ValueError, OSError) as exc:
                self.update_index(k, None, str(exc))
        if keys:
            self.annotationsChanged.emit(keys)
            self.indexChanged.emit()

    def delete_shapes(self, parent, shapes_by_key: dict[str, set[int]]) -> tuple[list[str], list[str], int]:
        """Shape indices are identical in the mapped and raw views (mapping never reorders)."""
        removed = [0]

        def make(idxs):
            def fn(ann):
                n = len(ann.shapes)
                for i in sorted(idxs, reverse=True):
                    if 0 <= i < n:
                        del ann.shapes[i]
                        removed[0] += 1
                return len(ann.shapes) != n
            return fn
        changed, errors = self.modify_many(parent, [(k, make(v)) for k, v in shapes_by_key.items()],
                                           "Deleting shapes...")
        return changed, errors, removed[0]

    def apply_to_shapes(self, parent, targets: dict, op: str, new_label: str | None = None,
                        what: str = "shapes") -> dict:
        """
        Delete / relabel shapes found by a scan, safely: each target is re-identified by its fingerprint in the file
        as it is NOW (core.shape_ops.resolve) — a shape that changed since the scan is skipped, never guessed; every
        rewritten file is backed up first (stats["backup"].restore() = undo). Locked files (editor session) are
        skipped by modify_many. targets: {key: [(shape index, fingerprint), …]}; op: "delete" | "relabel".
        """
        from AIO_QC.core.shape_ops import Backup, resolve
        backup = Backup(self.root, what)
        stats = {"done": 0, "missing": 0, "files": 0, "backup": backup}

        def make(key, tg):
            def fn(ann):
                idx, miss = resolve(ann.shapes, tg)
                stats["missing"] += len(miss)
                if not idx:
                    return False
                sample = self.by_key[key]
                if sample.ann_path is not None:
                    backup.save(sample.ann_path)           # the file is still the original here (saved after fn)
                if op == "delete":
                    for i in sorted(idx, reverse=True):
                        del ann.shapes[i]
                else:
                    for i in idx:
                        ann.shapes[i]["label"] = new_label
                stats["done"] += len(idx)
                return True
            return fn

        title = "Deleting shapes..." if op == "delete" else f"Relabelling shapes to '{new_label}'..."
        changed, errors = self.modify_many(parent, [(k, make(k, tg)) for k, tg in targets.items() if k in self.by_key],
                                           title)
        for k in changed:
            p = self.by_key[k].ann_path
            if p is not None:
                backup.written(p)
        backup.close()                                 # checkpoint: the store stays small on disk
        stats["files"] = len(changed)
        stats["changed"], stats["errors"] = changed, errors
        if changed:
            self.bulkChanged.emit(backup, what, len(changed))
        return stats

    def _frame_wh(self, sample, ann) -> tuple[float, float]:
        """Size that relative boxes refer to: 1×1 for normalised (YOLO) files, else the image size."""
        if ann.normalized:
            return 1.0, 1.0
        if ann.width and ann.height:
            return float(ann.width), float(ann.height)
        size = self.adapter.image_size(sample, ann)
        if not size:
            raise ValueError("image size unknown")
        return float(size[0]), float(size[1])

    def add_boxes(self, parent, items: dict, what: str = "added") -> dict:
        """
        Add new boxes: items = {key: [(label, (x1, y1, x2, y2) RELATIVE to the image), …]}. A box is skipped when
        the file already has a shape of that label covering it (IoU ≥ 0.7) — never added twice. Shaped like the
        file's other shapes (4-point polygon in polygon-only YOLO files). Backed up first (stats["backup"]).
        """
        from AIO_QC.core.inline_edit import new_box
        from AIO_QC.core.sequences import iou
        from AIO_QC.core.shape_ops import Backup
        backup = Backup(self.root, what)
        stats = {"done": 0, "missing": 0, "files": 0, "backup": backup, "skipped": 0, "refused": 0}

        def make(key, boxes):
            def fn(ann):
                sample = self.by_key[key]
                W, H = self._frame_wh(sample, ann)
                have = []
                for sh in ann.shapes:
                    pts = sh.get("points") or []
                    if len(pts) >= 2:
                        xs, ys = [p[0] / W for p in pts], [p[1] / H for p in pts]
                        have.append((sh.get("label"), (min(xs), min(ys), max(xs), max(ys))))
                todo = []
                for label, rb in boxes:
                    if any(l == label and iou(b, rb) >= 0.7 for l, b in have):
                        stats["skipped"] += 1
                        continue
                    if not self.adapter.can_use_label(label):
                        stats["refused"] += 1
                        continue
                    todo.append((label, rb))
                if not todo:
                    return False
                if sample.ann_path is not None and Path(sample.ann_path).is_file():
                    backup.save(sample.ann_path)
                for label, (x1, y1, x2, y2) in todo:
                    new_box(ann, (x1 * W, y1 * H, x2 * W, y2 * H), label, W, H)
                    have.append((label, (x1, y1, x2, y2)))
                    stats["done"] += 1
                return True
            return fn
        changed, errors = self.modify_many(parent, [(k, make(k, v)) for k, v in items.items() if k in self.by_key],
                                           "Adding boxes...")
        for k in changed:
            p = self.by_key[k].ann_path
            if p is not None:
                backup.written(p)
        backup.close()
        stats["files"], stats["changed"], stats["errors"] = len(changed), changed, errors
        if changed:
            self.bulkChanged.emit(backup, what, len(changed))
        return stats

    def set_boxes(self, parent, targets: dict, what: str = "boxfit") -> dict:
        """
        Replace the geometry of boxes: targets = {key: [(shape index, fingerprint, RELATIVE box), …]}. Each shape is
        re-identified by its fingerprint (skipped if it changed since); only rectangles (and 4-point polygons, kept
        as 4 corners) are reshaped. Backed up first.
        """
        from AIO_QC.core.shape_ops import Backup, fingerprint, resolve
        backup = Backup(self.root, what)
        stats = {"done": 0, "missing": 0, "files": 0, "backup": backup}

        def make(key, tg):
            def fn(ann):
                sample = self.by_key[key]
                W, H = self._frame_wh(sample, ann)
                found, miss = resolve(ann.shapes, [(i, fp) for i, fp, _b in tg])
                stats["missing"] += len(miss)
                by_fp = {fp: b for _i, fp, b in tg}
                hit = []
                for j in found:                              # resolve matched by fingerprint: same fp, same box
                    want = by_fp.get(fingerprint(ann.shapes[j]))
                    st = ann.shapes[j].get("shape_type") or "polygon"
                    if want is None or not (st == "rectangle" or (st == "polygon" and len(ann.shapes[j]["points"]) == 4)):
                        stats["missing"] += 1
                        continue
                    hit.append((j, st, want))
                if not hit:
                    return False
                if sample.ann_path is not None:
                    backup.save(sample.ann_path)
                for j, st, (x1, y1, x2, y2) in hit:
                    x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
                    pts = [[x1, y1], [x2, y2]] if st == "rectangle" else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    ann.shapes[j]["points"] = [[round(x, 6 if ann.normalized else 2), round(y, 6 if ann.normalized
                                                                                                 else 2)] for x, y in pts]
                    ann.shapes[j]["_geom_dirty"] = True
                    stats["done"] += 1
                return True
            return fn
        changed, errors = self.modify_many(parent, [(k, make(k, v)) for k, v in targets.items() if k in self.by_key],
                                           "Fitting boxes...")
        for k in changed:
            p = self.by_key[k].ann_path
            if p is not None:
                backup.written(p)
        backup.close()
        stats["files"], stats["changed"], stats["errors"] = len(changed), changed, errors
        if changed:
            self.bulkChanged.emit(backup, what, len(changed))
        return stats

    def apply_fixes(self, parent, fixes: dict, what: str = "fixes") -> dict:
        """
        Accept model findings as edits, all-or-nothing per shape, one backup (stats["backup"].restore() = undo):
            fixes = {key: [op, …]} with op one of
                ("relabel", shape index, expected box (rel), new label)
                ("reshape", shape index, expected box (rel), new box (rel))
                ("add", label, box (rel))
                ("delete", shape index, expected box (rel))
                ("set", shape index, expected box (rel), {"points": [[x, y] (rel)], "type": shape type or None})
                                                     any new outline (e.g. a part fitted into its parent)
        A target shape is re-identified by its box as it was scanned (the index first, then anywhere in the file):
        if it was moved / removed / reshaped since, the op is skipped and counted in stats["missing"].
        Adds are skipped when the file already has that label there (IoU ≥ 0.7). Rectangles are reshaped as
        rectangles, 4-point polygons as their 4 corners; other shape types are never reshaped.
        """
        from AIO_QC.core.inline_edit import new_box
        from AIO_QC.core.sequences import iou
        from AIO_QC.core.shape_ops import Backup
        backup = Backup(self.root, what)
        stats = {"done": 0, "missing": 0, "skipped": 0, "refused": 0, "files": 0, "backup": backup}

        def bbox(sh, W, H):
            pts = sh.get("points") or []
            if len(pts) < 2:
                return None
            xs, ys = [float(p[0]) / W for p in pts], [float(p[1]) / H for p in pts]
            return min(xs), min(ys), max(xs), max(ys)

        def find(ann, idx, want, W, H, used):
            near = lambda b: b is not None and all(abs(b[i] - want[i]) <= 0.004 for i in range(4))
            if 0 <= idx < len(ann.shapes) and idx not in used and near(bbox(ann.shapes[idx], W, H)):
                return idx
            for j, sh in enumerate(ann.shapes):
                if j not in used and near(bbox(sh, W, H)):
                    return j
            return None

        def make(key, ops):
            def fn(ann):
                sample = self.by_key[key]
                W, H = self._frame_wh(sample, ann)
                plan, used = [], set()
                for op in ops:
                    if op[0] in ("relabel", "reshape", "delete", "set"):
                        j = find(ann, op[1], op[2], W, H, used)
                        if j is None:
                            stats["missing"] += 1
                            continue
                        if op[0] == "relabel" and not self.adapter.can_use_label(op[3]):
                            stats["refused"] += 1
                            continue
                        if op[0] == "reshape":
                            st = ann.shapes[j].get("shape_type") or "polygon"
                            if not (st == "rectangle" or (st == "polygon" and len(ann.shapes[j]["points"]) == 4)):
                                stats["refused"] += 1
                                continue
                        used.add(j)
                        plan.append((op[0], j, op[3] if len(op) > 3 else None))
                    elif op[0] == "add":
                        label, rb = op[1], op[2]
                        if not self.adapter.can_use_label(label):
                            stats["refused"] += 1
                            continue
                        if any(sh.get("label") == label and (b := bbox(sh, W, H)) is not None and iou(b, rb) >= 0.7
                               for sh in ann.shapes):
                            stats["skipped"] += 1
                            continue
                        plan.append(("add", label, rb))
                if not plan:
                    return False
                if sample.ann_path is not None and Path(sample.ann_path).is_file():
                    backup.save(sample.ann_path)
                for kind, a, b in sorted(plan, key=lambda x: -x[1] if x[0] == "delete" else 0):
                    if kind == "delete":
                        continue
                    if kind == "relabel":
                        ann.shapes[a]["label"] = b
                    elif kind == "set":
                        nd = 6 if ann.normalized else 2
                        ann.shapes[a]["points"] = [[round(float(x) * W, nd), round(float(y) * H, nd)]
                                                   for x, y in b["points"]]
                        if b.get("type"):
                            ann.shapes[a]["shape_type"] = b["type"]
                        ann.shapes[a]["_geom_dirty"] = True
                    elif kind == "reshape":
                        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
                        st = ann.shapes[a].get("shape_type") or "polygon"
                        pts = [[x1, y1], [x2, y2]] if st == "rectangle" else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        nd = 6 if ann.normalized else 2
                        ann.shapes[a]["points"] = [[round(x, nd), round(y, nd)] for x, y in pts]
                        ann.shapes[a]["_geom_dirty"] = True
                    else:
                        x1, y1, x2, y2 = b
                        new_box(ann, (x1 * W, y1 * H, x2 * W, y2 * H), a, W, H)
                    stats["done"] += 1
                for j in sorted((a for kind, a, _b in plan if kind == "delete"), reverse=True):
                    del ann.shapes[j]                          # last, highest index first: indices stay valid
                    stats["done"] += 1
                return True
            return fn
        changed, errors = self.modify_many(parent, [(k, make(k, v)) for k, v in fixes.items() if k in self.by_key],
                                           "Applying fixes...")
        for k in changed:
            p = self.by_key[k].ann_path
            if p is not None:
                backup.written(p)
        backup.close()
        stats["files"], stats["changed"], stats["errors"] = len(changed), changed, errors
        if changed:
            self.bulkChanged.emit(backup, what, len(changed))
        return stats

    def undo_backup(self, backup) -> tuple[list[str], list[str]]:
        """Restore a Backup's originals (files changed since are left alone) and refresh. -> (keys, skipped paths)"""
        restored, skipped = backup.restore()
        keys = [self.key_by_path[p] for p in restored if p in self.key_by_path]
        self.log(keys, "undo", f"put back the original ({getattr(backup, 'title', '') or 'undo'})",
                 tool=self.tool or "Undo")
        self.reindex_keys(keys)
        return keys, skipped

    def relabel(self, parent, keys: Iterable[str], mapping: dict[str, str]) -> tuple[list[str], list[str], int]:
        count = [0]

        def fn(ann):
            hit = False
            for s in ann.shapes:
                new = mapping.get(s.get("label"))
                if new is not None and new != s.get("label"):
                    s["label"] = new
                    count[0] += 1
                    hit = True
            return hit
        changed, errors = self.modify_many(parent, [(k, fn) for k in keys], "Relabelling...")
        return changed, errors, count[0]

    # --------------------------------------------------------------- deletes
    def trash_samples(self, keys: Iterable[str], include_images: bool = False) -> tuple[list[str], list[str]]:
        """
        Move annotation files (and optionally images) to <root>/.aio_qc_trash/.
        LabelMe: sample disappears. YOLO: sample disappears only if its image is trashed,
        otherwise it becomes a background image.
        """
        keys = list(keys)
        blocked = [k for k in keys if k in self.locked]
        if blocked:
            keys = [k for k in keys if k not in self.locked]
            logger.warning(f"{len(blocked)} file(s) open in an editor session were not trashed")
        removed, changed, errors = [], [], []
        record = {"moves": [], "samples": [], "changed": []}
        if self.shared_ann:                                # COCO: the image leaves the file (its entry kept for undo)
            smps = [self.by_key[k] for k in keys if k in self.by_key]
            record["coco"] = self.base_adapter.remove_samples(smps)
            for s in smps:
                if include_images and s.image_path is not None and Path(s.image_path).exists():
                    try:
                        record["moves"].append((str(s.image_path), str(self.trash.move(s.image_path))))
                    except OSError as exc:
                        errors.append(f"{s.name}: {exc}")
                removed.append(s.key)
                record["samples"].append(s)
            if removed:
                self.forget_samples(removed)
                self.indexChanged.emit()
                self.ownPaths.emit(list({str(s.ann_path) for s in smps}))
            self.log(removed, "trash", "taken out of the COCO file" + (" (images to .aio_qc_trash)" if include_images
                                                                      else ""), tool=self.tool or "Delete")
            self.trashed.emit(record)
            return removed, errors
        for key in keys:
            s = self.by_key.get(key)
            if s is None:
                continue
            try:
                if s.ann_path is not None and s.ann_path.exists():
                    record["moves"].append((str(s.ann_path), str(self.trash.move(s.ann_path))))
                if (include_images or s.ann_path is None) and s.image_path is not None and s.image_path.exists():
                    # class-folder datasets have no annotation file: the image IS the sample
                    record["moves"].append((str(s.image_path), str(self.trash.move(s.image_path))))
            except OSError as exc:
                errors.append(f"{s.name}: {exc}")
                continue
            if self.is_yolo and not include_images:
                self.update_index(key, Counter())
                changed.append(key)
                record["changed"].append(key)
            else:
                removed.append(key)
                record["samples"].append(s)
        if removed:
            self.forget_samples(removed)
        if changed:
            self.annotationsChanged.emit(changed)
        if removed or changed:
            self.indexChanged.emit()
        self.log(removed + changed, "trash", "moved to .aio_qc_trash" + (" (with images)" if include_images else ""),
                 tool=self.tool or "Delete")
        if record["moves"]:
            self.trashed.emit(record)
        return removed + changed, errors

    def untrash(self, record) -> tuple[int, int]:
        """Undo a trash_samples(): move the files back (where nothing new took their place) and show them again.
        -> (files put back, files left in the trash)"""
        import shutil
        back, left = 0, 0
        for src, dest in record.get("moves", []):
            if os.path.exists(dest) and not os.path.exists(src):
                try:
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dest, src)
                    back += 1
                    continue
                except OSError:
                    pass
            left += 1
        if record.get("coco") and self.shared_ann:          # COCO: the entries go back into their file
            self.base_adapter.put_back(record["coco"])
            back += len(record["coco"].get("coco") or [])
        samples = [s for s in record.get("samples", []) if (s.ann_path is None or s.ann_path.exists())
                   and (s.image_path is None or s.image_path.exists())]
        if samples:
            self.add_samples(samples)
        keys = [k for k in record.get("changed", []) if k in self.by_key]
        if keys:
            self.reindex_keys(keys)
        self.log([s.key for s in samples] + keys, "undo", "put back from .aio_qc_trash", tool=self.tool or "Undo")
        return back, left

    def trash_paths(self, paths: Iterable[str]) -> tuple[list[str], list[str]]:
        """For files that are not samples (e.g. YOLO labels without image)."""
        done, errors = [], []
        for p in paths:
            try:
                if Path(p).exists():
                    self.trash.move(p)
                    done.append(p)
            except OSError as exc:
                errors.append(f"{Path(p).name}: {exc}")
        return done, errors
