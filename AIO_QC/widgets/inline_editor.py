# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
InlineEditor — fixes made directly on the image (canvas gestures -> safe file writes).

Every change: the file is re-read fresh, checked unchanged since it was shown and the target shape re-identified,
the original is stored once in the backup store, the edit is written atomically, the index / views refresh, and a
byte-exact undo step is recorded (Edit ▸ Undo / Redo, Ctrl+Z / Ctrl+Shift+Z). Works for LabelMe and YOLO (detect,
segment, OBB) and every shape type; classification datasets have no shapes (use Change Class).
A flagged file that is edited becomes 🔧 Fixed (“corrected — needs a look”), like Fix in Editor.
"""
from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QInputDialog

from AIO_QC.core.inline_edit import (
    EditHistory, EditRefused, Step, delete, file_sig, new_box, relabel, same_shape, set_points, stype,
)
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG


class InlineEditor(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.history = EditHistory()
        self.sig = None                       # signature of the shown annotation file
        self._store = None
        self._op = None
        self._backed: set[str] = set()
        self._last_label = None
        c = app.canvas
        c.shapeEdited.connect(self.on_shape_edited)
        c.shapeDeleteRequested.connect(self.on_delete)
        c.relabelRequested.connect(self.on_relabel)
        c.boxDrawn.connect(self.on_box)
        c.editRefused.connect(lambda msg: self._say(f"✋ {msg}", warn=True))
        c.shapeSelected.connect(self._hint)

    # ------------------------------------------------------------------ state
    @property
    def session(self):
        return self.app.session

    def reset(self):
        """A different dataset: history and backups belong to the old one."""
        self.history = EditHistory()
        if self._store is not None:
            self._store.close()
        self._store, self._op, self._backed = None, None, set()
        self._update_actions()

    def refresh(self):
        """After a file is shown or reloaded: may it be edited, and remember its on-disk signature."""
        s, key, c = self.session, self.app.current_key, self.app.canvas
        note = ""
        sample = s.by_key.get(key) if s is not None and key else None
        if sample is None or c.pixmap_item.pixmap().isNull():
            note = "no image shown"
        elif s.is_classification:
            note = "classification dataset: images have classes, not shapes — use Change Class (C)"
        elif sample.ann_path is None:
            note = "this image has no annotation file"
        elif key in s.locked:
            note = f"open in {s.locked[key]} — edit it there"
        self.sig = file_sig(sample.ann_path) if sample is not None and sample.ann_path is not None else None
        c.set_editable(not note, note)

    def classes(self) -> list[str]:
        """Relabel targets in a stable order: the dataset's class list, else the labels alphabetically."""
        s = self.session
        names = list(s.true_classes or [])
        if not names:
            names = sorted(s.label_totals(), key=str.lower)
        return names

    def _wh(self):
        pix = self.app.canvas.pixmap_item.pixmap()
        return float(pix.width()), float(pix.height())

    # ------------------------------------------------------------------ commit
    def _backup(self, key: str):
        if key in self._backed:
            return
        from AIO_QC.core.backups import BackupStore
        sample = self.session.by_key[key]
        if self._store is None:
            self._store = BackupStore(self.session.root)
        if self._op is None:
            self._op = self._store.begin("inline", "Inline edits " + time.strftime("%Y-%m-%d %H:%M"))
        if sample.ann_path is not None and Path(sample.ann_path).is_file():
            self._store.save(self._op, sample.ann_path)
        self._backed.add(key)

    def commit(self, fn, select_after=None) -> bool:
        s, key = self.session, self.app.current_key
        if s is None or key is None:
            return False
        try:
            self._backup(key)
            before, what = s.edit_file(key, fn, expect_sig=self.sig)
        except EditRefused as exc:
            self._say(f"✋ Not changed: {exc}", warn=True)
            if "changed on disk" in str(exc):
                self.app.reload_current_annotation()
            else:
                self.app.load_canvas_shapes()                # put the preview back where the file is
            return False
        except Exception as exc:                             # unreadable file, YOLO class map, …
            logger.warning(f"[inline edit] {key}: {exc}")
            self._say(f"✋ Not changed: {exc}", warn=True)
            self.app.load_canvas_shapes()
            return False
        path = str(s.by_key[key].ann_path)
        after_sig = file_sig(path)
        self.history.push(Step(key, path, before, Path(path).read_bytes() if Path(path).is_file() else None,
                               after_sig, what))
        self.sig = after_sig
        logger.info(f"[inline edit] {Path(path).name}: {what}")
        if s.review.status(key) == REVIEW_FLAG:
            s.set_review([key], REVIEW_FIXED, f"Fixed inline · {time.strftime('%Y-%m-%d %H:%M')} · {what}")
        self._update_actions()
        self._say(f"✓ {what} — Ctrl+Z to undo")
        if select_after is not None:
            QTimer.singleShot(0, lambda: self.app.canvas.select_shape(select_after))
        return True

    # ------------------------------------------------------------------ gestures
    def _shown(self, idx: int) -> dict | None:
        ann = self.app.current_ann
        return ann.shapes[idx] if ann is not None and 0 <= idx < len(ann.shapes) else None

    def _guard(self, ann, idx, shown):
        W, H = self._wh()
        if not same_shape(ann, idx, shown, W, H):
            raise EditRefused("that shape changed since it was shown — reloaded, please try again")

    def on_shape_edited(self, idx: int, pts: list):
        shown = self._shown(idx)
        if shown is None:
            return
        shown = {"label": shown.get("label"), "shape_type": stype(shown), "points": shown.get("points")}
        W, H = self._wh()

        def fn(ann):
            self._guard(ann, idx, shown)
            old = len(ann.shapes[idx].get("points") or [])
            set_points(ann, idx, pts, W, H)
            verb = ("added a vertex to" if len(pts) > old else "removed a vertex from" if len(pts) < old
                    else "reshaped")
            return f"{verb} #{idx} {shown['label']}"
        self.commit(fn, select_after=idx)

    def on_delete(self, idx: int):
        shown = self._shown(idx)
        if shown is None:
            return
        shown = {"label": shown.get("label"), "shape_type": stype(shown), "points": shown.get("points")}
        self.app.canvas.clearSelection()

        def fn(ann):
            self._guard(ann, idx, shown)
            delete(ann, idx)
            return f"deleted #{idx} {shown['label']}"
        n = len(self.app.current_ann.shapes) if self.app.current_ann else 0
        self.commit(fn, select_after=min(idx, n - 2) if n > 1 else None)

    def _choose_label(self, title: str, current: str | None) -> str | None:
        names = self.classes()
        s = self.session
        fixed = s.is_yolo                                  # YOLO: only classes of the class map
        start = names.index(current) if current in names else (names.index(self._last_label)
                                                               if self._last_label in names else 0)
        text, ok = QInputDialog.getItem(self.app, title, "Label (type to search" + ("" if fixed else
                                        " or enter a new one") + "):", names or [""], start, not fixed)
        text = (text or "").strip()
        return text if ok and text else None

    def on_relabel(self, idx: int, n: int):
        shown = self._shown(idx)
        if shown is None:
            return
        names = self.classes()
        if n:
            if n > len(names):
                self._say(f"✋ There is no class {n} (the list has {len(names)})", warn=True)
                return
            label = names[n - 1]
        else:
            label = self._choose_label("Relabel", shown.get("label"))
            if label is None:
                return
        if label == shown.get("label"):
            return
        snap = {"label": shown.get("label"), "shape_type": stype(shown), "points": shown.get("points")}
        can = self.session.adapter.can_use_label

        def fn(ann):
            self._guard(ann, idx, snap)
            relabel(ann, idx, label, can)
            return f"#{idx} {snap['label']} → {label}"
        self._last_label = label
        self.commit(fn, select_after=idx)

    def on_box(self, rect):
        label = self._choose_label("New box", self._last_label)
        if label is None:
            return
        W, H = self._wh()
        r = (rect.left(), rect.top(), rect.right(), rect.bottom())
        can = self.session.adapter.can_use_label
        if not can(label):
            self._say(f"✋ '{label}' is not a class of this dataset", warn=True)
            return
        self._last_label = label
        n = len(self.app.current_ann.shapes) if self.app.current_ann else 0

        def fn(ann):
            new_box(ann, r, label, W, H)
            return f"new {label} box"
        self.commit(fn, select_after=n)

    # ------------------------------------------------------------------ undo / redo
    def undo(self):
        self._step(self.history.undo, "Undone")

    def redo(self):
        self._step(self.history.redo, "Redone")

    def _step(self, op, verb):
        try:
            st = op()
        except EditRefused as exc:
            self._say(f"✋ {exc}", warn=True)
            self._update_actions()
            return
        s = self.session
        if s is not None:
            s.log([st.key], "undo" if verb == "Undone" else "edit", f"{verb.lower()}: {st.what}", tool="On-image edit")
            s.reindex_keys([st.key])
            if st.key != self.app.current_key and st.key in s.by_key:
                self.app.open_sample(st.key)                   # show where the change happened
        self.refresh()
        self._update_actions()
        self._say(f"↶ {verb}: {st.what} ({Path(st.path).name})")

    def _update_actions(self):
        a = self.app.a
        if hasattr(a, "undo_edit"):
            a.undo_edit.setEnabled(self.history.can_undo())
            a.redo_edit.setEnabled(self.history.can_redo())
            a.undo_edit.setText(f"Undo {self.history.undo_stack[-1].what}" if self.history.can_undo() else "Undo")
            a.redo_edit.setText(f"Redo {self.history.redo_stack[-1].what}" if self.history.can_redo() else "Redo")

    # ------------------------------------------------------------------ feedback
    def _hint(self, _idx):
        if not self.app.canvas.editable:
            return
        names = self.classes()[:9]
        keys = " · ".join(f"{i} {n}" for i, n in enumerate(names, 1))
        self._say(f"Edit: drag handles / body · arrows nudge · Del delete · L relabel · {keys}", timeout=6000)

    def _say(self, text: str, warn: bool = False, timeout: int = 5000):
        self.app.statusBar().showMessage(text, 8000 if warn else timeout)
