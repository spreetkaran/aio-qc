# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
InlineEditor — fixes made directly on the image (canvas gestures -> safe file writes).

Every change: the file is re-read fresh, checked unchanged since it was shown and the target shape re-identified,
the original is stored once in the backup store, the edit is written atomically, the index / views refresh, and a
byte-exact undo step is recorded (Edit ▸ Undo / Redo, Ctrl+Z / Ctrl+Shift+Z). Works for LabelMe and YOLO (detect,
segment, OBB) and every shape type; classification datasets have no shapes (use Change Class).
A flagged file that is edited becomes Fixed (“corrected — needs a look”), like Fix in Editor.
"""
from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from AIO_QC.utils import later
from PyQt6.QtCore import QObject

from AIO_QC.core.inline_edit import (
    EditHistory, EditRefused, Step, copy_shape, delete, file_sig, fit_points, new_box, new_polygon, paste_shape,
    relabel, same_shape, set_points, stype,
)
from AIO_QC.core.diag import ignored as _ignored

MIME = "application/x-aioqc-shapes"          # copied shapes on the system clipboard (also between two AIO QC windows)
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
        self._clip = None                     # the last copy (used when the system clipboard cannot be read)
        self._pasted: dict = {}               # (copy id, image) -> how many times pasted there (the offset grows)
        self._refused = ""
        c = app.canvas
        c.shapesDeleteRequested.connect(self.on_delete_many)
        c.shapesMoved.connect(self.on_moved_many)
        c.selectionChanged.connect(lambda _idx: self._update_actions())
        c.shapeEdited.connect(self.on_shape_edited)
        c.shapeDeleteRequested.connect(self.on_delete)
        c.relabelRequested.connect(self.on_relabel)
        c.boxDrawn.connect(self.on_box)
        c.polygonDrawn.connect(self.on_polygon)
        c.editRefused.connect(lambda msg: self._say(f"✋ {msg}", warn=True))
        c.shapeSelected.connect(self._hint)
        c.activeHandleChanged.connect(self._handle_hint)

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
        self.sig = s.ann_sig(sample) if s is not None and sample is not None else None
        if s is not None and s.shared_ann:                   # COCO: Undo works on this image's part of the file
            self.history.unit_read, self.history.unit_write = s.unit_read, s.unit_write
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
        sample = self.session.by_key[key]
        if self.session.shared_ann and sample.ann_path is not None:
            key = str(sample.ann_path)                      # COCO: the shared file, backed up once
        if key in self._backed:
            return
        from AIO_QC.core.backups import BackupStore
        if self._store is None:
            self._store = BackupStore(self.session.root)
        if self._op is None:
            self._op = self._store.begin("inline", "Inline edits " + time.strftime("%Y-%m-%d %H:%M"))
        if sample.ann_path is not None and Path(sample.ann_path).is_file():
            self._store.save(self._op, sample.ann_path)
        self._backed.add(key)

    def commit(self, fn, select_after=None, quiet: bool = False) -> bool:
        """quiet: one of several edits in a row (paste) — nothing is said or redrawn here, the reason of a refusal
        is left in self._refused and the caller reports once."""
        s, key = self.session, self.app.current_key
        if s is None or key is None:
            return False
        self._refused = ""
        try:
            self._backup(key)
            before, what = s.edit_file(key, fn, expect_sig=self.sig)
        except EditRefused as exc:
            self._refused = str(exc)
            if quiet:
                return False
            self._say(f"✋ Not changed: {exc}", warn=True)
            if "changed on disk" in str(exc):
                self.app.reload_current_annotation()
            else:
                self.app.load_canvas_shapes()                # put the preview back where the file is
            return False
        except Exception as exc:                             # unreadable file, YOLO class map, …
            logger.warning(f"[inline edit] {key}: {exc}")
            self._refused = str(exc)
            if quiet:
                return False
            self._say(f"✋ Not changed: {exc}", warn=True)
            self.app.load_canvas_shapes()
            return False
        smp = s.by_key[key]
        if s.shared_ann:
            path = f"unit::{key}"
            after = s.unit_read(key)
            after_sig = s.ann_sig(smp)
        else:
            path = str(smp.ann_path)
            after = Path(path).read_bytes() if Path(path).is_file() else None
            after_sig = file_sig(path)
        self.history.push(Step(key, path, before, after, after_sig, what))
        self.sig = after_sig
        logger.info(f"[inline edit] {Path(path).name}: {what}")
        if quiet:
            return True
        if s.review.status(key) == REVIEW_FLAG:
            s.set_review([key], REVIEW_FIXED, f"Fixed inline · {time.strftime('%Y-%m-%d %H:%M')} · {what}")
        self._update_actions()
        self._say(f"✓ {what} — Ctrl+Z to undo")
        if select_after is not None:
            later(0, self.app.canvas, lambda: self.app.canvas.select_shape(select_after))
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

    def _confirm_delete(self, label: str, idx: int) -> bool:
        """Ask before a shape goes (Ctrl+Z would bring it back, but an accidental Del should not need it)."""
        st = self.app.settings
        if not st.value("edit/confirm_delete", True, type=bool):
            return True
        from PyQt6.QtWidgets import QCheckBox, QMessageBox
        box = QMessageBox(self.app)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Delete shape")
        box.setText(f"Delete the “{label}” shape (#{idx})?")
        box.setInformativeText("Ctrl+Z brings it back.")
        delete_btn = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(delete_btn)
        again = QCheckBox("Don't ask again")
        again.setToolTip("To be asked again: Settings ▸ Viewer ▸ Ask before deleting a shape")
        box.setCheckBox(again)
        res = box.exec()
        clicked = box.clickedButton()
        if clicked is not delete_btn and not (clicked is None and res == int(QMessageBox.StandardButton.Yes)):
            return False                            # (None + Yes: an exec() stubbed in tests that says yes)
        if again.isChecked():
            st.setValue("edit/confirm_delete", False)
        return True

    def on_delete(self, idx: int):
        shown = self._shown(idx)
        if shown is None:
            return
        if not self._confirm_delete(str(shown.get("label")), idx):
            return
        shown = {"label": shown.get("label"), "shape_type": stype(shown), "points": shown.get("points")}
        self.app.canvas.clearSelection()

        def fn(ann):
            self._guard(ann, idx, shown)
            delete(ann, idx)
            return f"deleted #{idx} {shown['label']}"
        n = len(self.app.current_ann.shapes) if self.app.current_ann else 0
        self.commit(fn, select_after=min(idx, n - 2) if n > 1 else None)

    def _choose_label(self, title: str, current: str | None, kind: str = "box", near=None) -> str | None:
        """The label picker popup next to the shape (type to search, 1 … 9, Enter, Esc)."""
        from AIO_QC.widgets.label_picker import pick_label
        names = self.classes()
        s = self.session
        c = self.app.canvas
        at = None
        if near is not None:                               # just below the shape's bottom-right corner
            vp = c.mapFromScene(near)
            at = c.viewport().mapToGlobal(vp)
        counts = dict(s.label_totals()) if s is not None else {}
        return pick_label(self.app, title, names, current if current in names else self._last_label,
                          colors=dict(self.app.label_colors), counts=counts, allow_new=not s.is_yolo, at=at,
                          kind=kind)

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
            from PyQt6.QtCore import QPointF
            pts = shown.get("points") or [[0, 0]]
            W, H = self._wh()
            px = None
            try:
                from AIO_QC.core.inline_edit import to_px
                px = to_px(self.app.current_ann, pts, W, H)
            except Exception as _exc:
                _ignored(_exc)
                px = None
            near = QPointF(max(p[0] for p in px), max(p[1] for p in px)) if px else None
            label = self._choose_label("Change label", shown.get("label"), kind="relabel", near=near)
            if label is None:
                return
        if label == shown.get("label"):
            return
        snap = {"label": shown.get("label"), "shape_type": stype(shown), "points": shown.get("points")}
        can = self.session.adapter.can_use_label
        many = self.app.canvas.selected_indices()
        if len(many) > 1 and idx in many:                    # the whole selection gets the label, in one step
            snaps = self._snaps(many)

            def fn_many(ann):
                n = 0
                for i, sn in snaps.items():
                    self._guard(ann, i, sn)
                    if str(ann.shapes[i].get("label")) != label:
                        relabel(ann, i, label, can)
                        n += 1
                if not n:
                    raise EditRefused(f"they are all '{label}' already")
                return f"{n} shapes → {label}"
            self._last_label = label
            if self.commit(fn_many):
                self._reselect(many, idx)
            return

        def fn(ann):
            self._guard(ann, idx, snap)
            relabel(ann, idx, label, can)
            return f"#{idx} {snap['label']} → {label}"
        self._last_label = label
        self.commit(fn, select_after=idx)

    def on_box(self, rect):
        label = self._choose_label("New bbox", self._last_label, kind="box", near=rect.bottomRight())
        if label is None:
            return
        W, H = self._wh()
        r = (rect.left(), rect.top(), rect.right(), rect.bottom())
        can = self.session.adapter.can_use_label
        if not can(label):
            self._say(f"✋ '{label}' is not a class of this dataset", warn=True)
            return
        self._last_label = label
        assist = getattr(self.app, "assist", None)
        if assist is not None and assist.box_drawn(rect, label):
            return                                   # Outline Boxes with SAM: the outline is added when it is back
        n = len(self.app.current_ann.shapes) if self.app.current_ann else 0

        def fn(ann):
            new_box(ann, r, label, W, H)
            return f"new {label} box"
        self.commit(fn, select_after=n)

    def on_polygon(self, pts):
        from PyQt6.QtCore import QPointF
        near = QPointF(max(p[0] for p in pts), max(p[1] for p in pts))
        label = self._choose_label("New polygon", self._last_label, kind="polygon", near=near)
        if label is None:
            return
        if not self.session.adapter.can_use_label(label):
            self._say(f"✋ '{label}' is not a class of this dataset", warn=True)
            return
        W, H = self._wh()
        self._last_label = label
        n = len(self.app.current_ann.shapes) if self.app.current_ann else 0

        def fn(ann):
            new_polygon(ann, pts, label, W, H)
            return f"new {label} polygon ({len(pts)} corners)"
        self.commit(fn, select_after=n)

    # ------------------------------------------------------------------ several shapes
    def _snaps(self, idxs) -> dict:
        out = {}
        for i in idxs:
            sh = self._shown(i)
            if sh is not None:
                out[i] = {"label": sh.get("label"), "shape_type": stype(sh), "points": sh.get("points")}
        return out

    def _reselect(self, idxs, primary=None):
        c = self.app.canvas
        later(0, c, lambda: c.select_many(idxs, primary))

    def on_delete_many(self, idxs: list, ask: bool = True, verb: str = "deleted") -> bool:
        snaps = self._snaps(idxs)
        if not snaps:
            return False
        if len(snaps) == 1 and ask:
            self.on_delete(next(iter(snaps)))
            return True
        st = self.app.settings
        if ask and st.value("edit/confirm_delete", True, type=bool):
            from PyQt6.QtWidgets import QCheckBox, QMessageBox
            from collections import Counter
            labs = Counter(str(v["label"]) for v in snaps.values())
            box = QMessageBox(self.app)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Delete shapes")
            box.setText(f"Delete the {len(snaps)} selected shapes?")
            box.setInformativeText(", ".join(f"{n} {k}" for k, n in labs.most_common(6))
                                   + (" …" if len(labs) > 6 else "") + "\n\nCtrl+Z brings them back (one step).")
            delete_btn = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(delete_btn)
            again = QCheckBox("Don't ask again")
            box.setCheckBox(again)
            res = box.exec()
            clicked = box.clickedButton()
            if clicked is not delete_btn and not (clicked is None and res == int(QMessageBox.StandardButton.Yes)):
                return False
            if again.isChecked():
                st.setValue("edit/confirm_delete", False)
        self.app.canvas.clearSelection()

        def fn(ann):
            for i, sn in snaps.items():
                self._guard(ann, i, sn)
            for i in sorted(snaps, reverse=True):
                delete(ann, i)
            return f"{verb} {len(snaps)} shape{'s' if len(snaps) != 1 else ''}"
        return self.commit(fn)

    def on_moved_many(self, moved: list):
        """[(index, new points)] — several shapes dragged / nudged together: one write, one undo step."""
        snaps = self._snaps([i for i, _p in moved])
        moved = [(i, p) for i, p in moved if i in snaps]
        if not moved:
            return
        W, H = self._wh()
        idxs = self.app.canvas.selected_indices()
        primary = self.app.canvas.selected_group.index if self.app.canvas.selected_group else None

        def fn(ann):
            for i, _p in moved:
                self._guard(ann, i, snaps[i])
            for i, pts in moved:
                set_points(ann, i, pts, W, H)
            return f"moved {len(moved)} shapes"
        if self.commit(fn):
            self._reselect(idxs, primary)

    def select_all(self):
        c = self.app.canvas
        if c.selected_group is None:
            self._say("Select a shape first — then Ctrl+A selects them all", timeout=4000)
            return
        n = c.select_all()
        self._say(f"{n} shapes selected — drag or arrows move them · Del delete · Ctrl+C copy · 1 … 9 / L relabel",
                  timeout=6000)

    # ------------------------------------------------------------------ copy / cut / paste
    def copy(self, cut: bool = False) -> int:
        c, ann = self.app.canvas, self.app.current_ann
        idxs = c.selected_indices()
        if not idxs or ann is None:
            self._say("Select the shapes to copy first (Ctrl / Shift + click adds more)", timeout=4000)
            return 0
        if cut and not c.editable:
            self._say(f"✋ {c.edit_note or 'this file cannot be edited'}", warn=True)
            return 0
        c._flush_nudge()
        W, H = self._wh()
        shapes = []
        for i in idxs:
            if 0 <= i < len(ann.shapes) and not ann.shapes[i].get("_error") and ann.shapes[i].get("points"):
                shapes.append(copy_shape(ann, i, W, H))
        if not shapes:
            return 0
        s = self.session
        data = {"id": f"{time.time_ns():x}", "root": str(s.root) if s is not None else "",
                "key": self.app.current_key, "size": [W, H], "shapes": shapes}
        self._clip = data
        try:                                               # the system clipboard too (another AIO QC window)
            import json
            from PyQt6.QtCore import QMimeData
            from PyQt6.QtWidgets import QApplication
            md = QMimeData()
            md.setData(MIME, json.dumps(data).encode("utf-8"))
            md.setText(", ".join(f"{x['label']} ({x['shape_type']})" for x in shapes))
            QApplication.clipboard().setMimeData(md)
        except Exception as exc:
            logger.debug(f"[copy] system clipboard: {exc}")
        if cut:
            if self.on_delete_many(idxs, ask=False, verb="cut"):
                self._say(f"✂ Cut {len(shapes)} shape(s) — Ctrl+V pastes them · Ctrl+Z puts them back")
        else:
            self._say(f"Copied {len(shapes)} shape(s) — Ctrl+V pastes them here or in another image")
        self._update_actions()
        return len(shapes)

    def cut(self) -> int:
        return self.copy(cut=True)

    def _clipboard(self) -> dict | None:
        try:
            import json
            from PyQt6.QtWidgets import QApplication
            md = QApplication.clipboard().mimeData()
            if md is not None and md.hasFormat(MIME):
                data = json.loads(bytes(md.data(MIME)).decode("utf-8"))
                if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                    return data
        except Exception as exc:
            logger.debug(f"[paste] system clipboard: {exc}")
        return self._clip

    def can_paste(self) -> bool:
        return self._clipboard() is not None

    def paste(self) -> int:
        """The copied shapes into the shown image at exactly the same place — on top of their source when that is
        this image — brought inside the image, as this dataset's format allows; selected afterwards so they can be
        dragged at once. Every pasted shape is its own undo step — Ctrl+Z takes them back one by one."""
        from collections import Counter
        data = self._clipboard()
        s, key, c = self.session, self.app.current_key, self.app.canvas
        if not data or not data.get("shapes"):
            self._say("Nothing to paste — select shapes and copy them first (Ctrl+C)", timeout=4000)
            return 0
        if s is None or key is None:
            return 0
        if not c.editable:
            self._say(f"✋ {c.edit_note or 'this file cannot be edited'}", warn=True)
            return 0
        c._flush_nudge()
        W, H = self._wh()
        k = (data.get("id"), key)
        n_here = self._pasted.get(k, 0)
        off = 0                                            # the same place, every time
        start = len(self.app.current_ann.shapes) if self.app.current_ann else 0
        can = s.adapter.can_use_label
        done, skipped, whats = 0, Counter(), []
        stale = False
        s.blockSignals(True)                               # one refresh for the whole paste, not one per shape
        try:
            for src in data["shapes"]:
                try:
                    pts = fit_points(src.get("points") or [], W, H, off, off)
                except (TypeError, ValueError):
                    pts = None
                if pts is None:
                    skipped["does not fit in this image"] += 1
                    continue

                def fn(ann, src=src, pts=pts):
                    paste_shape(ann, src, pts, W, H, can)
                    return f"pasted {src.get('label')} {src.get('shape_type') or 'polygon'}"
                if self.commit(fn, quiet=True):
                    done += 1
                    whats.append(str(src.get("label")))
                else:
                    skipped[self._refused or "not possible"] += 1
                    if "changed on disk" in self._refused:
                        stale = True
                        break
        finally:
            s.blockSignals(False)
        if stale:
            self.app.reload_current_annotation()
        if done:
            self._pasted[k] = n_here + 1
            if s.review.status(key) == REVIEW_FLAG:
                s.set_review([key], REVIEW_FIXED, f"Fixed inline · {time.strftime('%Y-%m-%d %H:%M')} · pasted "
                                                  f"{done} shape(s)")
            s.annotationsChanged.emit([key])
            s.indexChanged.emit()
            self._reselect(list(range(start, start + done)))
        self._update_actions()
        why = "; ".join(f"{n} skipped: {r}" for r, n in skipped.most_common(3))
        if done:
            self._say(f"✓ Pasted {done} shape(s)" + (f" ({why})" if why else "")
                      + (" — Ctrl+Z takes them back one by one" if done > 1 else " — Ctrl+Z to undo"), timeout=7000)
        else:
            self._say(f"✋ Nothing pasted — {why or 'no shapes'}", warn=True)
        return done

    # ------------------------------------------------------------------ undo / redo
    def undo(self):
        if self._class_step(True):
            return
        self._step(self.history.undo, "Undone")

    def redo(self):
        if self._class_step(False):
            return
        self._step(self.history.redo, "Redone")

    def _class_step(self, back: bool) -> bool:
        """Classification datasets: Ctrl+Z / Ctrl+Y undo / redo Change class."""
        s = self.session
        if s is None or not s.is_classification:
            return False
        stack = s.class_undo if back else s.class_redo
        if not stack:
            return True
        what = stack[-1]["what"]
        done, errors = s.undo_class() if back else s.redo_class()
        app = self.app
        if errors:
            app.log_errors([f"[change class] {e}" for e in errors])
        if app.current_key not in s.by_key and done:
            app.file_dock.select_key(done[0])
        elif done:
            app.reload_current_annotation()
        self._say(f"{'Undone' if back else 'Redone'}: {what} ({len(done)} image(s))")
        self._update_actions()
        return True

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
        s = self.session
        if hasattr(a, "undo_edit") and s is not None and s.is_classification:
            a.undo_edit.setEnabled(bool(s.class_undo))
            a.redo_edit.setEnabled(bool(s.class_redo))
            a.undo_edit.setText(f"Undo {s.class_undo[-1]['what']}" if s.class_undo else "Undo")
            a.redo_edit.setText(f"Redo {s.class_redo[-1]['what']}" if s.class_redo else "Redo")
        elif hasattr(a, "undo_edit"):
            a.undo_edit.setEnabled(self.history.can_undo())
            a.redo_edit.setEnabled(self.history.can_redo())
            a.undo_edit.setText(f"Undo {self.history.undo_stack[-1].what}" if self.history.can_undo() else "Undo")
            a.redo_edit.setText(f"Redo {self.history.redo_stack[-1].what}" if self.history.can_redo() else "Redo")
        if hasattr(a, "copy_shapes"):
            has = bool(self.app.canvas.selected_indices())
            a.copy_shapes.setEnabled(has)
            a.cut_shapes.setEnabled(has and self.app.canvas.editable)
            a.select_all_shapes.setEnabled(has)

    # ------------------------------------------------------------------ feedback
    def _hint(self, _idx):
        if not self.app.canvas.editable:
            return
        if getattr(self.app.canvas, "_active_handle", None) is not None:   # re-selected after its edit was saved
            return
        names = self.classes()[:9]
        keys = " · ".join(f"{i} {n}" for i, n in enumerate(names, 1))
        self._say(f"Edit: arrows move it (Shift ×10) · click an edge or corner, then arrows move just that side · "
                  f"Del delete · L relabel · {keys}", timeout=8000)

    def _handle_hint(self, on):
        if on:
            self._say("Arrows move this edge / corner (Shift ×10) · Esc or a click on the shape: move the whole "
                      "shape again", timeout=8000)
        else:
            self._say("Arrows move the whole shape (Shift ×10)", timeout=3000)

    def _say(self, text: str, warn: bool = False, timeout: int = 5000):
        self.app.statusBar().showMessage(text, 8000 if warn else timeout)
