# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC main window — Dataset lifecycle: open / close, live updates, class verification, settings, label index.

A mixin of AIO_QC.app.MainWindow: the methods use the window's own attributes (self.canvas, self.session …)
set up in MainWindow.__init__.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QMessageBox,
)

from AIO_QC import __appname__, utils
from AIO_QC.core import DatasetType, detect_dataset_types, open_dataset
from AIO_QC.core.class_map import load_classes_file
from AIO_QC.widgets import (
    IndexWorker, Session,
)
from AIO_QC.widgets import anim
from AIO_QC.widgets.keymap import KeyMapOverlay
from AIO_QC.widgets.dataset_watcher import DatasetWatcher
from AIO_QC.widgets.inference_service import FuncWorker
from AIO_QC.widgets.model_setup_dialog import ModelSetupDialog
from AIO_QC.core.env_manager import remove_env

from AIO_QC.app_common import (GROUPS)
from AIO_QC.core.diag import ignored as _ignored


class DatasetMixin:

    # =====================================================================
    # Dataset lifecycle
    # =====================================================================
    def open_dir(self):
        start = self.settings.value("last_dir", "")
        path = QFileDialog.getExistingDirectory(self, "Select dataset root (LabelMe dir, YOLO root, or YOLO split root)", start)
        if path:
            self.settings.setValue("last_dir", path)
            self.open_dataset(Path(path))

    def open_dataset(self, root: Path, dtype: DatasetType | None = None):
        if dtype is None:
            types = detect_dataset_types(root)
            if not types:
                self.open_dir_btn.flash("error", f"No dataset found in {root}")
                QMessageBox.warning(self, "Open", "No LabelMe JSONs, root images/+labels/, or "
                                                  "<split>/images+labels/ found in:\n" + str(root))
                return
            dtype = types[0]
            if len(types) > 1:
                names = [t.title for t in types]
                choice, ok = QInputDialog.getItem(self, "Dataset type",
                                                  "Several layouts were detected. Open as:", names, 0, False)
                if not ok:
                    return
                dtype = types[names.index(choice)]
        if not self._end_editor_session("open another dataset"):
            return
        if self.session is not None and self._same_root(self.session.root, root):
            # the dataset that is open, opened again (Recent, Open, the start page): close it first, so the new session
            # finds its cache free instead of waiting for the old session's lock (and then running uncached)
            self._close_session()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            adapter = open_dataset(root, dtype)
            session = Session(root, adapter, self.user_classes)
        except Exception as exc:
            logger.exception("open failed")
            self.open_dir_btn.flash("error", f"Last open failed: {exc}")
            QApplication.restoreOverrideCursor()
            from AIO_QC.widgets.errors import show_error
            show_error(self, "Could not open the dataset", exc, app=self,
                       retry=lambda r=root, t=dtype: self.open_dataset(Path(r), t))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._close_session()
        self.session = session
        self.canvas.forget_view()                    # a new dataset opens at home; adjustments are per file
        self._display_by_key = {}
        session.annotationsChanged.connect(self._on_annotations_changed)
        session.annotationsChanged.connect(self._note_own_writes)        # our writes are not "outside" changes
        session.samplesRemoved.connect(self._on_samples_removed)
        session.samplesAdded.connect(self._on_samples_added)
        session.ownPaths.connect(lambda paths: self.tracker.note_own(paths) if self.tracker is not None else None)
        session.issuesChanged.connect(self._issues_save_timer.start)
        session.indexChanged.connect(self._labels_refresh.start)
        session.reviewChanged.connect(self._on_review_changed)
        session.classUndoChanged.connect(self.inline._update_actions)
        session.classUndoChanged.connect(self.class_panel.refresh)        # undo / redo: the Class panel too
        session.keysRenamed.connect(self._follow_renamed_keys)
        session.bulkChanged.connect(self._offer_undo)
        session.trashed.connect(self._offer_untrash)
        self.file_dock.model.review = session.review
        self.filmstrip.set_session(session)
        self._add_recent(root, dtype)
        self._sync_start_page()
        from AIO_QC.widgets import coach
        if not coach.done("image"):                 # the first dataset ever: a short tour of the window
            utils.later(700, self, lambda: self.session is not None and self.start_tour(force=False))

        # YOLO class names: reuse a loaded classes file if the dataset has none, then CHECK them against the labels
        self._class_scan = None
        if session.is_yolo and not adapter.class_names and self.user_classes:
            adapter.set_class_names(self.user_classes, self.user_classes_path)
            self.statusBar().showMessage("No dataset.yaml/classes.txt in dataset → using the loaded classes file", 6000)
        if session.is_yolo:
            QTimer.singleShot(0, self._check_yolo_classes_on_open)
        QTimer.singleShot(50, self._offer_unfinished_editor_sessions)

        # Label colours: the dataset's saved ones first, then the class list in order (most distinct first)
        from AIO_QC.utils import colors as _colors
        _colors.load(root, self.label_colors)
        for name in adapter.class_names:
            utils.update_label_colors(name, self.label_colors)

        self.refresh_status_buttons()
        self.dataset_label.reset()                          # a newly opened dataset starts with an empty path
        self.dataset_label.setText(f"{root.name} · {dtype.title} · {len(session.samples):,} files")
        self._size_path_area(session)
        self._update_title()
        self._external_log = []
        self._start_watcher()
        self._start_sweep()
        self.a.change_class.setEnabled(session.is_classification)
        folders = session.type == DatasetType.CLS_FOLDERS
        for act in (self.a.send_issues, self.a.send_issues_note, self.a.open_issues, self.a.review_issues,
                    self.a.return_issues):
            act.setEnabled(folders)
        split_note = getattr(adapter, "split_note", "")
        self.dataset_label.setToolTip(str(root) + (f"\n{split_note}" if split_note else ""))
        if split_note:
            self.statusBar().showMessage(split_note, 10000)
            logger.info(f"[splits] {split_note}")
        if dtype in (DatasetType.YOLO, DatasetType.YOLO_SPLIT) and adapter.class_map.problems:
            self.log_errors([f"[class map] {p}" for p in adapter.class_map.problems])
        self.file_dock.set_samples(session.samples, session.splits, session.folders)
        self.set_dataset_actions_enabled(True)
        if getattr(self, "testing", None) is not None and self.testing.isVisible():
            utils.later(0, self.testing, self.testing.use_qc_dataset)     # Testing follows the open dataset
        self._update_mapping_indicator()
        self._update_review_progress()
        if session.label_map:
            self.statusBar().showMessage(f"🔀 Saved label mapping loaded ({len(session.label_map)} label(s)) — "
                                         "Tools ▸ Label Mapping to change", 8000)
        self._start_index()
        if session.samples:
            self.file_dock.select_row(0)
        else:
            self.statusBar().showMessage("Dataset contains no samples", 5000)

    def _follow_renamed_keys(self, ren: dict):
        """Images moved by AIO QC (class folders): the vectors of the last Visual Similarity run follow them, so
        look-based sorting keeps working after a class change."""
        emb = getattr(self, "embed_index", None)
        if emb and ren:
            m = emb.get("map") or {}
            for old, new in ren.items():            # whole images (classification): (key, -1)
                if (old, -1) in m:
                    m[(new, -1)] = m.pop((old, -1))

    def reload_dataset(self):
        if self.session:
            key = self.current_key
            self.open_dataset(self.session.root, self.session.type)
            if key:
                self.file_dock.select_key(key)

    # ------------------------------------------------------------------ live updates (outside changes)
    def _start_watcher(self):
        self._stop_watcher()
        s = self.session
        if s is None or not self.a.watch.isChecked():
            return
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        exts = set(IMAGE_EXTENSIONS) | ({".txt"} if s.is_yolo else {".json"})
        self.tracker = None
        w = DatasetWatcher(s.root, exts, self)
        w.snapshotReady.connect(lambda snap, took, w=w: self._on_snapshot(snap, took, w))
        self.watcher = w
        w.start()

    def _stop_watcher(self):
        w = getattr(self, "watcher", None)
        if w is not None:
            try:
                w.snapshotReady.disconnect()
            except TypeError:
                pass
            w.stop()
        self.watcher, self.tracker = None, None

    def _toggle_watch(self, on: bool):
        self.settings.setValue("watch/enabled", bool(on))
        if on:
            self._start_watcher()
            self.statusBar().showMessage("Watching the dataset for changes made outside AIO QC", 4000)
        else:
            self._stop_watcher()

    def _note_own_writes(self, keys):
        if getattr(self, "tracker", None) is not None and self.session is not None:
            self.tracker.note_own(self.session.paths_of(keys))

    def _on_app_state(self, state):
        if state == Qt.ApplicationState.ApplicationActive and getattr(self, "watcher", None) is not None:
            self.watcher.poke()                       # back from another program: check right away

    def _on_snapshot(self, snap, _took, watcher=None):
        from AIO_QC.core.watch import ChangeTracker
        if self.session is None or watcher is not self.watcher:      # a stopped watcher's late snapshot
            return
        if self.tracker is None:
            self.tracker = ChangeTracker(snap)        # first scan = baseline
            return
        ch = self.tracker.step(snap)
        if ch:
            try:
                self._apply_external_changes(ch)
            except Exception as exc:                   # a live update must never take the app down
                logger.exception(exc)
                self.log_errors([f"[live update] {exc}"], quiet=True)

    def _apply_external_changes(self, ch):
        from AIO_QC.core.watch import is_meta
        s = self.session
        kp = s.key_by_path                                      # persistent path -> key map (no O(n) rebuild)
        ann_to_key = {p: kp[p] for p in ch.added + ch.modified + ch.removed
                      if p in kp and str(s.by_key[kp[p]].ann_path) == p}
        img_to_key = {p: kp[p] for p in ch.added + ch.modified + ch.removed
                      if p in kp and str(s.by_key[kp[p]].image_path) == p}
        modified, removed, new, image_changed = set(), set(), [], set()
        structural = [p for p in ch.added + ch.modified + ch.removed if is_meta(p)]
        for p in ch.modified:
            if p in ann_to_key:
                modified.add(ann_to_key[p])
            elif p in img_to_key:
                modified.add(img_to_key[p])
                image_changed.add(img_to_key[p])
            elif self.current_ann is not None and str(self.current_ann.image_path or "") == p and self.current_key:
                image_changed.add(self.current_key)       # LabelMe: the current file's image was replaced
        for p in ch.removed:
            if p in ann_to_key:
                (modified if s.is_yolo else removed).add(ann_to_key[p])     # YOLO: label gone -> background image
            elif p in img_to_key:
                removed.add(img_to_key[p])
        for p in ch.added:
            if p in ann_to_key:                           # YOLO: a label file created for an existing image
                modified.add(ann_to_key[p])
                continue
            if is_meta(p):
                continue
            smp, struct = s.base_adapter.sample_for_path(p)
            if struct:
                structural.append(p)
            elif smp is not None:
                new.append(smp)
        locked = {k for k in modified | removed if k in s.locked}
        modified -= removed
        if modified:                                       # audit trail: changes nobody made in AIO QC
            s.log(sorted(modified), "external", "changed on disk by another program", tool="Outside AIO QC")
        if removed:
            s.log(sorted(removed), "external", "removed on disk by another program", tool="Outside AIO QC")
        if removed:
            s.forget_samples(sorted(removed))
        added_keys = s.add_samples(new) if new else []
        if modified:
            s.reindex_keys(sorted(modified))
        if self.current_key in image_changed:
            self.pixmap_cache.clear()
            self.show_sample(self.current_key)
        elif self.current_key in modified:
            self.reload_current_annotation()
        n = len(modified) + len(removed) + len(added_keys)
        if n:
            for d in list(self.open_dialogs.values()):
                if hasattr(d, "mark_stale"):
                    d.mark_stale(n)
        now = time.strftime("%H:%M:%S")
        for kind, keys in (("changed", modified), ("added", added_keys), ("removed", removed)):
            for k in sorted(keys):
                self._external_log.append((now, kind, k))
        for p in structural:
            self._external_log.append((now, "layout", p))
        self._external_log = self._external_log[-2000:]
        if n:
            logger.info(f"[live update] outside changes: {len(modified)} changed · {len(added_keys)} added · "
                        f"{len(removed)} removed")
        parts = [f"{v:,} {t}" for v, t in ((len(modified), "changed"), (len(added_keys), "added"),
                                            (len(removed), "removed")) if v]
        if structural:
            names = ", ".join(sorted({Path(p).name for p in structural})[:3])
            self.toast.show_message(
                "<b>The dataset layout changed outside AIO QC</b><br>"
                f"{names}{' …' if len(structural) > 3 else ''} — class names or split folders may differ now. "
                "Reload to apply." + (f"<br>Also: {' · '.join(parts)} (already updated)." if parts else ""),
                kind="warn", sticky=True,
                actions=[("Reload dataset", self.reload_dataset)], details=self._external_log_text)
        elif parts:
            # one notification while it is on screen: counts accumulate, its timer restarts
            if self.toast.find("external") is None:
                self._ext_counts = Counter()
                self._ext_locked = 0
            self._ext_counts.update({"changed": len(modified), "added": len(added_keys), "removed": len(removed)})
            self._ext_locked += len(locked)
            total = [f"{v:,} {k}" for k, v in self._ext_counts.items() if v]
            extra = (f"<br>{self._ext_locked} of them are open in the editor session (it will show a conflict)."
                     if self._ext_locked else "")
            self.toast.show_message(f"<b>Updated from disk</b><br>{' · '.join(total)} outside AIO QC.{extra}",
                                    kind="warn" if self._ext_locked else "info", key="external",
                                    details=self._external_log_text)
            self.statusBar().showMessage(f"Dataset updated from disk: {' · '.join(parts)}", 6000)

    def _on_samples_added(self, keys):
        s = self.session
        self.file_dock.insert_samples(s.last_inserted)        # discovery-order rows, like a reload
        self.dataset_label.set_info(f"{s.type.title} · {len(s.samples):,} files")   # keeps the file shown
        self._size_path_area(s)
        self._update_position()

    def _external_log_text(self) -> str:
        """Changes made outside AIO QC this session, newest first (the notification's Details)."""
        by_key = self.session.by_key if self.session else {}
        icon = {"changed": "✎", "added": "＋", "removed": "−", "layout": "⚠"}
        lines = [f"{t}  {icon.get(k, '·')} {k:8s} {by_key[x].display if x in by_key else x}"
                 for t, k, x in reversed(self._external_log)]
        return "\n".join(lines) or "No outside changes in this session."

    def _size_path_area(self, session):
        """The toolbar path is sized for the dataset's median file name (and split name)."""
        import statistics
        from pathlib import PurePath
        samples = session.samples
        step = max(1, len(samples) // 4000)                 # a sample is plenty for a median
        names = [len(PurePath(x.name).stem) for x in samples[::step]]
        splits = [len(x.split or "") for x in samples[::step]]
        if names:
            self.dataset_label.set_typical(int(statistics.median(names)),
                                           int(statistics.median(splits)) if splits else 0)

    def _update_title(self):
        """One pattern in every mode: '<mode> — <what is open> — AIO QC App'."""
        s = getattr(self, "session", None)
        self.setWindowTitle(f"Image QC — {s.root.name} — {__appname__}" if s is not None
                            else f"Image QC — {__appname__}")

    @staticmethod
    def _same_root(a, b) -> bool:
        try:
            import os
            return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))
        except (OSError, ValueError):
            return False

    def _close_session(self):
        for mode in (getattr(self, "sketch", None), getattr(self, "roi", None)):
            if mode is not None and mode.active:
                mode.leave()                         # saves the drawing / ROIs of the dataset that is closing
        self._stop_watcher()
        self._stop_sweep()
        if self.session is not None:
            try:
                self.session.issues.save()
            except OSError as exc:
                logger.warning(f"could not save the issues inbox: {exc}")
        self._save_review(compact=True)
        self._stop_index()
        scan = getattr(self, "_class_scan_worker", None)      # the YOLO class-id scan of this dataset
        if scan is not None:
            try:
                scan.stop()
            except RuntimeError:                              # already finished and freed
                pass
            self._class_scan_worker = None
        hk = getattr(self, "_housekeeper", None)
        if hk is not None:
            try:
                if hk.isRunning():
                    hk.wait(30000)                  # it uses the cache that is closed next
            except RuntimeError:                    # already deleted
                pass
            self._housekeeper = None
        if self.session is not None:
            self.session.close()
        for d in list(self.open_dialogs.values()):
            d.close()
        self.open_dialogs.clear()
        self.generation += 1
        self._inflight.clear()
        self.pixmap_cache.clear()
        self.session = None
        self._sync_start_page()
        self.check_guide.reset()
        self._update_title()
        self._update_flagged_count()
        self.current_key = self.current_ann = None
        self.inline.reset()
        self.embed_index = None
        self.label_colors.clear()
        self.visible_labels.clear()
        self.hidden_labels.clear()
        self.unique_labels.clear()
        self.annotations_list.clear()
        self.canvas.clearAll()
        self.statusBar().set_image(None, None)
        self.statusBar().set_file(None)
        self.statusBar().set_pointer(None)
        self.zoom_spin.setEnabled(False)
        self.zoom_spin.set_zoom(None)
        self.zoom_slider.set_zoom(None)
        self.file_dock.model.review = None
        self.filmstrip.set_session(None)

    def load_classes_file(self) -> bool:
        """
        LabelMe: reference list for ambiguity / unknown-label checks.
        YOLO:    becomes the class map (id -> name) when the dataset has no dataset.yaml.
        """
        if self.session and self.session.is_yolo and self.session.base_adapter.has_yaml:
            src = self.session.base_adapter.class_map.source.name
            if QMessageBox.question(
                    self, "Load classes",
                    f"This YOLO dataset defines its classes in {src}, which takes precedence.\n"
                    "A loaded classes.txt will NOT change its class names.\n\nLoad one anyway (kept for LabelMe datasets)?"
            ) != QMessageBox.StandardButton.Yes:
                return False
        start = str(self.session.root) if self.session else self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select classes list", start,
                                              "Class lists (classes.txt *.txt *.names *.yaml *.yml);;All files (*)")
        if not path:
            return False
        try:
            classes = load_classes_file(path)
        except Exception as exc:
            _ignored(exc)
            self.load_classes_btn.flash("error", f"Could not read {Path(path).name}: {exc}")
            QMessageBox.warning(self, "Load classes", str(exc))
            return False
        if not classes:
            self.load_classes_btn.flash("error", f"{Path(path).name} contains no class names")
            QMessageBox.warning(self, "Load classes", "No class names found in that file.")
            return False
        if len(set(classes)) != len(classes):
            dups = sorted({c for c in classes if classes.count(c) > 1})
            QMessageBox.warning(self, "Load classes", f"Duplicate class names: {', '.join(dups)}\n"
                                                      "Loaded anyway — fix the file to avoid wrong id mapping.")
        s = self.session
        if s and s.is_yolo and not s.base_adapter.has_yaml:
            res = self._verify_yolo_classes(classes, Path(path))
            if res == "retry":
                return self.load_classes_file()
            if res == "cancel":
                return False
        self.user_classes, self.user_classes_path = classes, Path(path)
        if s:
            s.user_classes, s.user_classes_path = classes, Path(path)
            if s.is_yolo and not s.base_adapter.has_yaml:
                self._apply_yolo_classes(classes, Path(path))
        self.refresh_status_buttons()
        self.statusBar().showMessage(f"Loaded {len(classes)} classes from {Path(path).name}", 5000)
        return True

    # ------------------------------------------------------------------ YOLO class verification
    def _scan_class_ids(self):
        """Class ids used by the label files — scanned once per session (threaded), refreshed after edits."""
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QProgressDialog
        from AIO_QC.core.class_check import scan_class_ids
        from AIO_QC.widgets.inference_service import FuncWorker
        s = self.session
        cached = getattr(self, "_class_scan", None)
        if cached is not None and cached[0] is s:
            return cached[1]
        files = [x.ann_path for x in s.samples if x.ann_path is not None]
        dlg = QProgressDialog(f"Checking the class ids used in {len(files):,} label files …", None, 0, 0, self)
        dlg.setWindowTitle("Class check")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(400)
        loop, out = QEventLoop(), {}
        w = FuncWorker(lambda: scan_class_ids(files), self)
        w.done.connect(lambda r: (out.update(scan=r), loop.quit()))
        w.failed.connect(lambda m: (out.update(error=m), loop.quit()))
        w.start()
        loop.exec()
        w.wait()
        dlg.close()
        if "error" in out:
            logger.warning(f"class id scan failed: {out['error']}")
            return None
        self._class_scan = (s, out["scan"])
        return out["scan"]

    def _scan_class_ids_async(self, then):
        """The same scan without blocking (no progress dialog, no nested event loop): `then(scan)` when done, if the
        dataset is still the one open."""
        from AIO_QC.core.class_check import scan_class_ids
        from AIO_QC.widgets.inference_service import FuncWorker
        s = self.session
        cached = getattr(self, "_class_scan", None)
        if cached is not None and cached[0] is s:
            then(cached[1])
            return
        files = [x.ann_path for x in s.samples if x.ann_path is not None]
        old = getattr(self, "_class_scan_worker", None)
        if old is not None:                                  # a scan for a dataset no longer open: stop it
            try:
                old.stop()
            except RuntimeError:
                pass
        box = {}
        w = FuncWorker(lambda: scan_class_ids(files, workers=2, stop=box["w"].cancel_event.is_set), self)
        box["w"] = w

        def done(scan):
            if getattr(scan, "cancelled", False):
                return
            if self.session is s:
                self._class_scan = (s, scan)
                then(scan)
        w.done.connect(done)
        w.failed.connect(lambda m: logger.warning(f"class id scan failed: {m}"))
        w.finished.connect(w.deleteLater)                    # one per opened dataset: freed once it has ended
        self._class_scan_worker = w
        w.start()

    def _warn_yaml_class_mismatch(self, scan):
        from AIO_QC.core.class_check import verify_classes
        ad = self.session.base_adapter
        src_name = ad.class_map.source.name if ad.class_map.source else "dataset.yaml"
        v = verify_classes(ad.class_names, scan, src_name) if scan else None
        if v is not None and v.kind == "mismatch":
            self.load_classes_btn.flash("warning", v.details)
            self.log_errors([f"[class check] {v.details}"], quiet=True)
            self.statusBar().showMessage(f"⚠ {v.title}: {v.details.splitlines()[0]}", 12000)

    def _verify_yolo_classes(self, names: list[str], source: Path | None) -> str:
        """
        Are these class names right for THIS dataset?  -> "ok" | "accept" (mismatch, used anyway) | "retry" | "cancel"
          mismatch (a label uses an id without a name): say so immediately and ask for the correct classes.txt;
          match: ask the user to confirm it is the final list (a list in the wrong ORDER also "matches"),
                 remembered per dataset until the file changes.
        """
        from AIO_QC.core.class_check import verify_classes
        from AIO_QC.core.state import classes_confirmed, confirm_classes
        scan = self._scan_class_ids()
        if scan is None:
            return "ok"
        src_name = source.name if source else "the class list"
        v = verify_classes(names, scan, src_name)
        if v.kind == "no_labels":
            return "ok"
        box = QMessageBox(self)
        box.setWindowTitle("Class check")
        if v.kind == "mismatch":
            self.load_classes_btn.flash("error", f"{src_name} does not match the labels")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(f"<b>{v.title}</b>")
            box.setInformativeText(v.details + "\n\nLoad the correct classes.txt for this dataset?")
            retry = box.addButton("Load the correct classes.txt…", QMessageBox.ButtonRole.AcceptRole)
            anyway = box.addButton("Use it anyway", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(retry)
            box.exec()
            if box.clickedButton() is retry:
                return "retry"
            if box.clickedButton() is anyway:
                self.log_errors([f"[class check] {v.details}"], quiet=True)
                return "accept"
            return "cancel"
        if source is not None and classes_confirmed(self.session.root, source, names):
            return "ok"
        preview = "\n".join(f"{i:>3}  {n}" for i, n in enumerate(names[:20])) + \
                  (f"\n  …  ({len(names) - 20} more)" if len(names) > 20 else "")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"<b>Is {src_name} the final class list for this dataset?</b>")
        box.setInformativeText(f"{v.details}\n\nThe ids match, but a list in the wrong order would match too — "
                               "please confirm the names below are correct (line N = class id N).")
        box.setDetailedText(preview)
        yes = box.addButton("Yes, it's the final list", QMessageBox.ButtonRole.AcceptRole)
        other = box.addButton("Load a different file…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(yes)
        box.exec()
        if box.clickedButton() is yes:
            if source is not None:
                confirm_classes(self.session.root, source, names)
            return "ok"
        return "retry" if box.clickedButton() is other else "cancel"

    def _check_yolo_classes_on_open(self):
        s = self.session
        if s is None or not s.is_yolo:
            return
        ad = s.base_adapter
        if ad.has_yaml:                        # the YAML is authoritative: only warn about a mismatch — so the
            self._scan_class_ids_async(self._warn_yaml_class_mismatch)    # scan runs in the background, no dialog
            return
        if not ad.class_names:
            self._ask_yolo_classes()
            return
        res = self._verify_yolo_classes(list(ad.class_names), ad.class_map.source)
        if res == "retry":
            self.load_classes_file()

    def _apply_yolo_classes(self, classes, path):
        """Switch a YOLO dataset's id -> name map; everything label-derived is rebuilt."""
        s = self.session
        old = list(s.base_adapter.class_names)
        s.base_adapter.set_class_names(classes, path)
        s.set_label_map(s.label_map)           # rebuild the mapped view on top of the new names
        if old == list(classes):
            return
        for name in s.adapter.class_names:
            utils.update_label_colors(name, self.label_colors)
        self.visible_labels.clear()            # labels like "class_3" are gone
        self.hidden_labels.clear()
        s.index.clear()
        s.raw_index.clear()
        s.index_errors.clear()
        s.index_complete = False
        for d in list(self.open_dialogs.values()):
            d.close()                          # their results used the old names
        self.reload_current_annotation()
        self._start_index()
        self.log_errors([f"[class map] {p}" for p in s.base_adapter.class_map.problems], quiet=True)

    def _ask_yolo_classes(self):
        s = self.session
        if not s or not s.is_yolo or s.base_adapter.class_names:
            return
        self.load_classes_btn.flash("warning", "No dataset.yaml / classes.txt — class names unknown")
        reply = QMessageBox.question(
            self, "Class names missing",
            "This YOLO dataset has no dataset.yaml / data.yaml and no classes.txt.\n\n"
            "Load a classes.txt now? (line N = class id N)\n"
            "Without it labels show as class_0, class_1, … and the ambiguity check is unavailable.")
        if reply == QMessageBox.StandardButton.Yes:
            self.load_classes_file()

    def refresh_status_buttons(self):
        """Single place that decides the colour of Open Dir / Load Classes."""
        s = self.session
        if s is None:
            self.open_dir_btn.set_status("default")
        elif not s.samples:
            self.open_dir_btn.set_status("warning", f"{s.root}\nNo samples found")
        else:
            self.open_dir_btn.set_status("success", f"{s.root}\n{s.type.title} · {len(s.samples):,} files")

        if s is None:
            if self.user_classes:
                self.load_classes_btn.set_status("success", f"{len(self.user_classes)} classes from "
                                                            f"{self.user_classes_path.name}")
            else:
                self.load_classes_btn.set_status("default")
        elif s.is_yolo:
            cm = s.base_adapter.class_map
            if not cm.names:
                self.load_classes_btn.set_status("warning", "No dataset.yaml / classes.txt — load classes.txt")
            elif cm.problems:
                self.load_classes_btn.set_status("warning", f"{len(cm.names)} classes from {s.classes_source}\n"
                                                            + "\n".join(cm.problems))
            else:
                self.load_classes_btn.set_status("success", f"{len(cm.names)} classes from {s.classes_source}")
        elif s.user_classes:
            self.load_classes_btn.set_status("success", f"{len(s.user_classes)} classes from "
                                                        f"{s.user_classes_path.name if s.user_classes_path else 'file'}")
        else:
            self.load_classes_btn.set_status("default", "Optional for LabelMe: enables unknown-label checks")

    def _watch_theme(self):
        """Dark ↔ light while running (Qt ≥ 6.5 colour scheme, or a palette change): swap icon sets."""
        hints = QApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda _s: self._on_theme_changed())
        self._dark = utils.is_dark()

    def set_zoom_slider_visible(self, on: bool):
        """Settings ▸ Viewer ▸ Zoom slider in the toolbar (off in 3.0: − 37 % ▾ + is the zoom control)."""
        act = getattr(self, "_slider_action", None)
        if act is not None:
            act.setVisible(bool(on))

    def apply_canvas_background(self, which: str | None = None):
        """Settings ▸ Viewer ▸ Canvas background: follow the theme (default — light grey in the light theme, near
        black in the dark one), dark grey, light grey, or the window colour."""
        from PyQt6.QtGui import QBrush, QColor
        which = which or str(self.settings.value("viewer/canvas_bg", "auto") or "auto")
        light, dark = self.CANVAS_BG.get(which, self.CANVAS_BG["auto"])
        col = dark if utils.is_dark() else light
        self.canvas.setBackgroundBrush(QBrush(QColor(col)) if col else QBrush())

    def _on_theme_changed(self):
        self._dark = utils.is_dark()
        self.apply_canvas_background()
        utils.refresh_icons()
        for i, (key, _t, _q) in enumerate(GROUPS):
            self.tabs.setTabIcon(i, utils.newIcon(f"group-{key}"))
        self.tabs.setTabIcon(len(GROUPS), utils.newIcon("brightness-contrast"))
        self.statusBar().showMessage(f"{'Dark' if self._dark else 'Light'} icon set applied", 3000)

    def open_model_setup(self):
        if not self.validate_inputs():
            return
        dlg = self.open_dialogs.get("model_setup")
        if dlg is None:
            dlg = ModelSetupDialog(self)
            dlg.tool_key = "model_setup"
            dlg.setWindowIcon(utils.newIcon("tool-model-setup"))
            self.open_dialogs["model_setup"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _cleanup_temp_envs(self):
        """Temporary envs that could not be deleted last time (locked files, crash) are removed now."""
        pending = list(self.settings.value("inference/pending_delete", []) or [])
        tmp = self.inference.temp_env
        if tmp and self.inference.delete_temp_on_exit and Path(tmp).exists():
            pending.append(tmp)          # the app did not exit cleanly last time
        left = []
        for p in dict.fromkeys(pending):
            if Path(p).exists() and not remove_env(Path(p)):
                left.append(p)
        self.settings.setValue("inference/pending_delete", left)
        if tmp and not Path(tmp).exists():
            if self.inference.python.startswith(tmp):
                self.inference.set_python("")
            self.inference.set_temp_env("", self.inference.delete_temp_on_exit)

    def clear_cache(self):
        s = self.session
        if s is None or s.cache is None:
            return
        st = s.cache.stats()
        if QMessageBox.question(self, "Clear cache",
                                f"Delete cached data for this dataset?\n\n{st['labels']:,} label entries · "
                                f"{st['images']:,} image entries · {st['preds']:,} model predictions\n\n"
                                "Everything is rebuilt automatically when needed.") != QMessageBox.StandardButton.Yes:
            return
        s.cache.clear()
        self.statusBar().showMessage("Dataset cache cleared", 4000)

    # ------------------------------------------------------------------ space for the image
    def _set_toolbar_compact(self, on: bool):
        """Compact: one tidy row — the zoom box goes single-line and every icon button becomes a square of exactly
        its height (Open / Load Classes match it too). Full: icons with text under them, as before."""
        from PyQt6.QtWidgets import QToolButton
        on = bool(on)
        if self.isVisible():
            anim.crossfade(self.toolbar, 160)
        self.settings.setValue("ui/compact_toolbar", on)
        self.zoom_spin.set_compact(on)
        h = max(26, self.zoom_spin.sizeHint().height())
        icon = QSize(18, 18) if on else QSize(24, 24)
        style = Qt.ToolButtonStyle.ToolButtonIconOnly if on else Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        self.toolbar.setToolButtonStyle(style)
        self.toolbar.setIconSize(icon)
        for b in self.toolbar.findChildren(QToolButton):
            act = b.defaultAction()
            if act is None or b.parent() is not self.toolbar or b.property("chrome"):
                continue
            b.setToolButtonStyle(style)
            b.setIconSize(icon)
            if on:
                b.setFixedSize(h, h)
            else:
                b.setMinimumSize(0, 0)
                b.setMaximumSize(16777215, 16777215)
            name = act.text().replace("&&", "\0").replace("&", "").replace("\0", "&").replace("…", "")
            tip = act.toolTip()
            b.setToolTip(tip if (not on or name.lower() in tip.lower()) else f"<b>{name}</b> — {tip}")
        utils.style(self.toolbar, "QToolBar { spacing: 4px; padding: 6px 12px; }"
                                  "QToolBar::separator { background: @line; width: 1px; margin: 9px 8px; }")

    def toggle_keymap(self):
        if self._keymap is None:
            self._keymap = KeyMapOverlay(self)
        self._keymap.toggle()

    def open_guidelines(self, scope=None):
        self.open_tool("guidelines")
        dlg = self.open_dialogs.get("guidelines")
        if dlg is not None and scope:
            dlg.select(scope)

    def _panel_action(self, dock, title):
        """View ▸ Panels entry that slides the panel open / closed (instead of the dock's instant toggle)."""
        act = QAction(title, self)
        act.setCheckable(True)
        act.setChecked(not dock.isHidden())
        act.triggered.connect(lambda on, d=dock: anim.toggle_dock(self, d, on))
        dock.visibilityChanged.connect(lambda _v, a=act, d=dock: a.setChecked(not d.isHidden()))
        return act

    def _set_animations(self, on: bool):
        from AIO_QC.widgets import motion
        motion.set_mode("full" if on else "off")           # also sets anim.ENABLED

    # ------------------------------------------------------------------ settings
    def settings_sections(self):
        from AIO_QC.widgets.settings_pages import image_sections
        return image_sections(self)

    def open_settings(self, key=None):
        from AIO_QC.widgets.settings_dialog import open_settings
        return open_settings(self, self.settings_sections(), "Settings — Image QC", key)

    def toggle_focus_mode(self, on=None):
        """Only the image: every dock and the toolbar hidden; the exact layout comes back on the second press.
        (The canvas re-fits by itself when it was fitted to the window.)"""
        if not isinstance(on, bool):
            on = self._focus_state is None
        anim.crossfade(self)
        if on and self._focus_state is None:
            self._focus_state = self.saveState()
            for d in self.findChildren(QDockWidget):
                d.hide()
            self.toolbar.hide()
            self.statusBar().showMessage("Focus mode — F11 (or Ctrl+Alt+F) brings the panels back", 5000)
        elif not on and self._focus_state is not None:
            state, self._focus_state = self._focus_state, None
            self.restoreState(state)
            self.toolbar.show()
        self.a.focus.blockSignals(True)
        self.a.focus.setChecked(self._focus_state is not None)
        self.a.focus.blockSignals(False)

    def _show_history(self):
        anim.toggle_dock(self, self.history_dock, True)
        self.history_dock.raise_()
        self.history_panel.set_key(self.current_key)

    def _start_new_box(self):
        c = self.canvas
        if not c.editable:
            self.statusBar().showMessage(f"✋ {c.edit_note or 'nothing to edit'}", 6000)
            return
        c.start_drawing("box")                     # the hint over the image says what to do

    def _start_new_polygon(self):
        c = self.canvas
        if not c.editable:
            self.statusBar().showMessage(f"✋ {c.edit_note or 'nothing to edit'}", 6000)
            return
        c.start_drawing("polygon")

    def open_palette(self):
        from AIO_QC.widgets.command_palette import CommandPalette
        if self._palette is None:
            self._palette = CommandPalette(self, self.settings)
        self._palette.popup()

    def open_storage(self):
        if not self.session:
            return
        from AIO_QC.widgets.storage_dialog import StorageDialog
        dlg = self.open_dialogs.get("storage")
        if dlg is None or dlg.session is not self.session:
            dlg = StorageDialog(self)
            self.open_dialogs["storage"] = dlg
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def backup_ops_in_use(self) -> set:
        """Backup operations an open undo still needs (never pruned)."""
        keep = set()
        b = getattr(self.open_dialogs.get("dim_area"), "_last_backup", None)
        if getattr(b, "op", None) is not None:
            keep.add(b.op)
        es = getattr(self.editor_dialog, "es", None) if getattr(self, "editor_dialog", None) else None
        if es is not None and es.backup_op is not None:
            keep.add(es.backup_op)
        return keep

    def _housekeeping(self):
        """Keep .aio_qc small (background, after indexing): retention, unused results, removed files, temp files."""
        from AIO_QC.core import storage as S
        from AIO_QC.widgets.storage_dialog import load_cfg
        s = self.session
        cfg = load_cfg(self.settings)
        if s is None or not cfg["auto"]:
            return
        root, cache, keep = s.root, s.cache, self.backup_ops_in_use()
        alive = [p for x in s.samples for p in (x.ann_path, x.image_path)]
        w = FuncWorker(lambda: S.housekeeping(root, cache, cfg, alive_paths=alive, keep_ops=keep), self)

        def done(r):
            for n in r["notes"]:
                logger.info(f"storage: {n}")
            if r["freed"] >= 5 * 2 ** 20 and self.session is s:
                self.toast.show_message(f"<b>Storage tidied</b><br>Freed {S.human(r['freed'])} in .aio_qc — "
                                        + "; ".join(r["notes"][:2]) + ".", kind="success", key="storage",
                                        actions=[("Storage…", self.open_storage)],
                                        details="\n".join(r["notes"]) or None)

        w.done.connect(done)
        w.failed.connect(lambda m: logger.warning(f"storage housekeeping failed: {m}"))
        w.finished.connect(w.deleteLater)
        w.finished.connect(w.deleteLater)                    # _close_session copes with an already freed one
        self._housekeeper = w
        w.start()

    def open_trash(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        if not self.session:
            return
        trash = self.session.root / ".aio_qc_trash"
        if not trash.exists():
            QMessageBox.information(self, "Trash", "Nothing has been moved to the trash yet.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(trash)))

    # ------------------------------------------------------------ label index
    def _start_index(self):
        self._stop_index()
        s = self.session
        self.index_worker = IndexWorker(s.adapter, s.samples, s.cache, s.label_namespace())
        self._index_started = time.monotonic()
        gen = self.generation
        self.index_worker.batch.connect(lambda b, g=gen: self._on_index_batch(b, g))
        from AIO_QC.widgets import tasks          # the status bar's loading ring + Background Tasks
        self._index_task = tasks.start(f"Indexing · {s.root.name}", owner=self)
        self.index_worker.progress.connect(
            lambda i, n, t=self._index_task: t.update(i / max(n, 1), f"{i:,} / {n:,} files") if i < n else None)
        w = self.index_worker
        w.completed.connect(lambda cancelled, g=gen, w=w: self._on_index_done(cancelled, g, w.hits))
        self.index_worker.finished.connect(self.index_worker.deleteLater)
        self.index_worker.start()

    def _stop_index(self):
        t = getattr(self, "_index_task", None)
        if t is not None:
            self._index_task = None
            t.finish()
        w = self.index_worker
        self.index_worker = None
        if w is not None:
            try:
                if w.isRunning():
                    w.stop()
                    w.wait(3000)
            except RuntimeError:  # already deleted
                pass

    def _on_index_batch(self, batch, gen):
        if gen != self.generation or not self.session:
            return
        flags = {}
        for key, counts, error in batch:
            self.session.update_index(key, counts, error)
            flags[key] = "error" if error else ("empty" if counts is not None and not counts else None)
        self.file_dock.model.set_flags(flags)
        self._labels_refresh.start()

    def _ask_cls_mode(self):
        """LabelMe-flag classification: images with several classes — on purpose (multi-label) or mistakes? Asked
        once per dataset (Classification Check changes it later)."""
        s = self.session
        if s is None or s.type.value != "cls_labelme" or s.cls_settings().get("mode"):
            return
        n = s.multi_label_count()
        if not n:
            s.save_cls_settings(mode="single")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Classification")
        box.setText(f"{n:,} image(s) have more than one class set.\n\nCan an image have several classes in this "
                    "dataset (multi-label), or is it one class per image?")
        multi = box.addButton("Several classes (multi-label)", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("One class per image", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        mode = "multi" if box.clickedButton() is multi else "single"
        s.save_cls_settings(mode=mode)
        self.statusBar().showMessage(
            "Multi-label: Change class (C) adds / removes classes; 1 … 9 toggle one" if mode == "multi" else
            f"One class per image — Classification Check lists the {n:,} image(s) with several", 8000)

    def _on_index_done(self, cancelled, gen, hits: int = 0):
        if gen != self.generation or not self.session:
            return
        self.session.index_complete = not cancelled
        if self.session.cache is not None:
            try:
                self.session.cache.flush()                  # release the cache's write lock now, not 500 rows later
            except Exception as exc:
                logger.warning(f"cache flush: {exc}")
        self.index_label.setText("")
        t = getattr(self, "_index_task", None)
        if t is not None:
            self._index_task = None
            t.finish()
        if not cancelled and not getattr(self.session, "_housekept", False):
            self.session._housekept = True
            QTimer.singleShot(1500, self._housekeeping)
        n_err = len(self.session.index_errors)
        if n_err:
            self.log_errors([f"[index] {Path(k).name}: {e}" for k, e in self.session.index_errors.items()])
        self.refresh_label_list()
        if not cancelled:
            self.check_guide.dataset_ready()
            QTimer.singleShot(0, self._ask_cls_mode)
        self.statusBar().showMessage(
            f"Indexed {len(self.session.samples):,} files in {time.monotonic() - self._index_started:.1f}s"
            + (f" ({hits:,} from cache)" if hits else "")
            + (f" · {n_err} unreadable (red in file list)" if n_err else ""),
            6000)
