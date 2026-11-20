# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing <-> the Image QC workspace. Everything is read on the UI thread and handed to workers as plain data
(lists / dicts): no Qt object or session is touched from a worker.

    snapshot()            the open dataset as Testing's default test data: AIO QC's own file list, each image's
                          annotation file and the dataset's class names (None + reason when not usable)
    write_labels(paths, fn)  a label fix from Testing's editor goes through AIO QC's write path: the original is
                          backed up first (History ▸ revert / Backups undo), the write is logged in the file's
                          history, the index is refreshed, a flagged file becomes "fixed", and the image shown in
                          Image QC reloads - the edit is visible in AIO QC at once
    send_to_review(paths, note)  flags images in AIO QC's review queue (Issues / Review Grid / file filter)
"""
from __future__ import annotations

import os

from loguru import logger
from AIO_QC.core.diag import ignored as _ignored

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class QCBridge:
    def __init__(self, image_window):
        self.win = image_window

    # ------------------------------------------------------------------ reading
    @property
    def session(self):
        return getattr(self.win, "session", None) if self.win is not None else None

    def unavailable(self) -> str | None:
        s = self.session
        if s is None:
            return "No dataset is open in Image QC."
        if not s.samples:
            return "The open dataset has no images."
        return None

    def snapshot(self) -> dict | None:
        """UI thread: everything a worker needs to load the open dataset as test data."""
        if self.unavailable():
            return None
        s = self.session
        if s.is_classification:            # (image, annotation, classes) per file; the class list in its order
            pairs = [(str(smp.image_path) if smp.image_path is not None else "",
                      str(smp.ann_path) if smp.ann_path is not None else "", list(s.tags_of(smp.key)))
                     for smp in s.samples]
            return dict(root=str(s.root), fmt="classify", pairs=pairs, cls_type=s.type,
                        names={i: n for i, n in enumerate(s.cls_classes())} or None, label=s.root.name)
        # (image, annotation) per file as plain strings. LabelMe files are keyed by their JSON; an image not
        # resolved yet ("") is found by the worker with AIO QC's rules (imagePath, then a same-name image)
        pairs = [(str(smp.image_path) if smp.image_path is not None else "",
                  str(smp.ann_path) if smp.ann_path is not None else "") for smp in s.samples]
        names = None
        if s.is_yolo:
            names = {i: n for i, n in enumerate(s.base_adapter.class_names or [])} or None
        elif s.true_classes:
            names = {i: n for i, n in enumerate(s.true_classes)}
        return dict(root=str(s.root), fmt="yolo" if s.is_yolo else "labelme", pairs=pairs, names=names,
                    label=s.root.name)

    def is_this(self, data) -> bool:
        s = self.session
        return bool(data is not None and getattr(data, "qc", False) and s is not None and
                    os.path.normcase(os.path.abspath(data.root)) == os.path.normcase(os.path.abspath(str(s.root))))

    def keys_for(self, paths) -> list[str]:
        s = self.session
        if s is None:
            return []
        out = []
        for p in paths:
            cands = p if isinstance(p, (tuple, list)) else (p,)
            for c in cands:
                if not c:
                    continue
                k = s.key_by_path.get(str(c)) or s.key_by_path.get(os.path.abspath(str(c)))
                if k is None and str(c).lower().endswith(tuple(IMAGE_EXTS)):      # LabelMe: keyed by its JSON
                    j = os.path.splitext(str(c))[0] + ".json"
                    k = s.key_by_path.get(j) or s.key_by_path.get(os.path.abspath(j))
                if k:
                    out.append(k)
                    break
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ writing
    def write_labels(self, paths, fn, what="labels fixed in Testing"):
        """Back up `paths`, run fn() (which writes them), then tell AIO QC. Returns fn()'s result."""
        from AIO_QC.core.shape_ops import Backup
        from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG
        s = self.session
        if s is None:
            return fn()
        paths = [str(p) for p in paths if p]
        backup = Backup(s.root, "testing", "Testing — label fixes")
        for p in paths:
            if os.path.isfile(p):
                backup.save(p)
        s.ownPaths.emit(paths)                       # our own write: not "changed outside AIO QC"
        try:
            res = fn()
        finally:
            for p in paths:
                if os.path.isfile(p):
                    backup.written(p)
            backup.close()
        keys = self.keys_for(paths)
        if keys:
            s.log(keys, "edit", what, tool="Testing", op=backup.op)
            s.reindex_keys(keys)
            flagged = [k for k in keys if s.review.status(k) == REVIEW_FLAG]
            if flagged:
                s.set_review(flagged, REVIEW_FIXED, "Fixed in Testing")
            cur = getattr(self.win, "current_key", None)
            if cur in keys and hasattr(self.win, "reload_current_annotation"):
                try:
                    self.win.reload_current_annotation()
                except Exception as exc:             # the write succeeded; a refresh problem is only logged
                    logger.warning(f"[testing] refresh after label fix: {exc}")
        return res

    def set_class(self, paths, new_class):
        """A classification dataset open in Image QC: give one image another class the way AIO QC does it (class
        folders: the image moves; LabelMe flags rewritten; backups, File History and Undo included).
        paths: (annotation, image). Returns the image's path afterwards. Raises ValueError."""
        s = self.session
        keys = self.keys_for([paths])
        if s is None or not keys:
            raise ValueError("The image is not in the dataset open in Image QC")
        done, errors = s.set_tags(keys[:1], tags=[new_class], what=f"class → {new_class} (Testing)")
        if errors:
            raise ValueError(errors[0])
        k = done[0] if done else keys[0]
        cur = getattr(self.win, "current_key", None)
        if cur in (k, keys[0]) and hasattr(self.win, "reload_current_annotation") and k in s.by_key:
            try:
                self.win.reload_current_annotation()
            except Exception as exc:                 # the change succeeded; a refresh problem is only logged
                logger.warning(f"[testing] refresh after class change: {exc}")
        smp = s.by_key.get(k)
        return str(smp.image_path) if smp is not None and smp.image_path is not None else None

    # ------------------------------------------------------------------ review
    def send_to_review(self, image_paths, note: str) -> int:
        """Flag these images in AIO QC's review queue. Returns how many were in the open dataset."""
        from AIO_QC.core.state import REVIEW_FLAG
        s = self.session
        keys = self.keys_for(image_paths)
        if s is None or not keys:
            return 0
        s.set_review(keys, REVIEW_FLAG, note)
        try:
            s.log(keys, "review", f"sent to review from Testing — {note}", tool="Testing")
        except Exception as _exc:
            _ignored(_exc)
        dock = getattr(self.win, "file_dock", None)
        if dock is not None and hasattr(dock, "set_key_filter"):
            try:
                dock.set_key_filter(keys, f"Testing: {note}")        # the file list shows just these
            except Exception as exc:
                logger.debug(f"[testing] file filter: {exc}")
        return len(keys)
