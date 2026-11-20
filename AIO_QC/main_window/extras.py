# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC main window — Logs, first steps, predictions, zoom, label mapping, review markers, filmstrip, recent datasets.

A mixin of AIO_QC.app.MainWindow: the methods use the window's own attributes (self.canvas, self.session …)
set up in MainWindow.__init__.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QInputDialog, QMessageBox,
    QTextEdit, QVBoxLayout,
)

from AIO_QC import __version__, utils
from AIO_QC.core import DatasetType
from AIO_QC.core.state import REVIEW_FLAG, REVIEW_FIXED, REVIEW_OK
from AIO_QC.widgets import (
    Canvas, LabelMapDialog,
)
from AIO_QC.widgets.filmstrip import WINDOW as FILMSTRIP_WINDOW
from AIO_QC.widgets import anim
from AIO_QC.widgets.flagged_dialog import FlaggedFilesDialog

from AIO_QC.app_common import (MAX_RECENT, QC_TOOLS, _find_ffmpeg)
from AIO_QC.core.diag import ignored as _ignored


class ExtrasMixin:

    # =====================================================================
    # Logs / misc
    # =====================================================================
    def log_errors(self, errors, quiet: bool = False):
        new = [e for e in errors if e not in self.error_log[-500:]]
        if not new:
            return
        self.error_log.extend(new)
        for e in new:
            logger.warning(e)
        self.a.logs.setEnabled(True)
        self.a.logs.setText(f"Logs ({len(self.error_log)})")
        if not quiet:
            from html import escape
            first = str(new[-1]).strip().splitlines()[0] if str(new[-1]).strip() else "an error"
            if len(first) > 140:
                first = first[:137] + "…"
            more = f" (and {len(new) - 1} more)" if len(new) > 1 else ""
            self.toast.show_message(
                f"<b>Something did not work</b><br>{escape(first)}{more}. The full details are in the log.",
                "warn", actions=[("Show details", self.show_log)], key="errors")

    def show_log(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Logs")
        lay = QVBoxLayout(dialog)
        text = QTextEdit(readOnly=True)
        text.setPlainText("\n".join(self.error_log))
        lay.addWidget(text)
        row = QHBoxLayout()

        def clear():
            self.error_log.clear()
            text.clear()
            self.a.logs.setText("Logs")
            self.a.logs.setEnabled(False)

        clear_btn = utils.newButton("Clear", slot=clear)
        close = utils.newButton("Close", slot=dialog.accept)
        row.addStretch(1)
        row.addWidget(clear_btn)
        row.addWidget(close)
        lay.addLayout(row)
        dialog.resize(760, 420)
        dialog.exec()

    def show_source_video(self, key=None):
        """QC -> Frame Extractor: the video an image was extracted from, at its frame. Only on the user's click."""
        s = self.session
        key = key or self.current_key
        smp = s.by_key.get(key) if s is not None and key else None
        if smp is None:
            return
        img = smp.image_path
        if img is None and hasattr(s.adapter, "resolve_image"):
            try:
                img = s.adapter.resolve_image(smp, None)
            except Exception as _exc:
                _ignored(_exc)
                img = None
        if img is None:
            QMessageBox.information(self, "Source video", "This file has no image.")
            return
        from AIO_QC.tools.frame_extractor.core.frame_sources import lookup
        from AIO_QC.utils.fx_settings import fx_settings
        recent = fx_settings().value("recent/files", []) or []
        r = lookup(str(img), recent if isinstance(recent, list) else [recent])
        if r is None:
            QMessageBox.information(self, "Source video",
                                    f"No source video is known for\n{img}\n\nFrames saved by the Frame Extractor "
                                    "record their video; older frames named <video>_frame_<n> are matched against "
                                    "the Frame Extractor's recent videos.")
            return
        self._shell_call("show_in_frame_extractor", r["video"], r["frame"], r.get("time"))

    def open_training(self, yaml=None, run=None):
        """Train Model (Models tab) — one window; its dataset: `yaml`, else the open YOLO dataset's file, else the
        last Convert & Split YOLO output of the open dataset."""
        from AIO_QC.widgets.train_dialog import TrainDialog
        dlg = getattr(self, "train_dialog", None)
        if dlg is None:
            dlg = TrainDialog(self)
            dlg.keep_after_close = True
            dlg.setWindowFlag(Qt.WindowType.Window, True)
            self.train_dialog = dlg
        if yaml is None and run is None and self.session is not None and not dlg.yaml.text().strip():
            yaml = self._dataset_yaml_for_training()
        if yaml is not None:
            dlg.set_dataset(yaml)
        if run is not None:
            dlg.refresh_runs(select=run, project=Path(run).parent)
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()

    # ---- Model Assist
    def _set_assist_auto(self, on):
        self.settings.setValue("assist/auto", bool(on))
        if on:
            self.assist.image_shown()

    def _set_assist_sam(self, on):
        self.settings.setValue("assist/sam", bool(on))
        self.statusBar().showMessage("Drawn boxes become outlines (SAM)" if on else "Drawn boxes stay boxes", 4000)

    def _assist_settings(self):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Assist")
        form = QFormLayout(dlg)
        conf = QDoubleSpinBox()
        conf.setRange(0.01, 0.99)
        conf.setSingleStep(0.05)
        conf.setDecimals(2)
        conf.setValue(self.assist.conf)
        form.addRow("Suggest objects the model is at least this sure of:", conf)
        sam = QLineEdit(self.assist.sam_model)
        sam.setPlaceholderText("sam2.1_b.pt")
        sam.setToolTip("A weights file, or a name ultralytics downloads: sam2.1_b.pt, sam2.1_t.pt, mobile_sam.pt …")
        form.addRow("SAM weights for outlines:", sam)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.setValue("assist/conf", float(conf.value()))
            if sam.text().strip():
                self.settings.setValue("assist/sam_model", sam.text().strip())

    def open_improve(self, run=None, check: bool = False):
        """Models ▸ Improve Model — the train → check → label → retrain loop of the open dataset. run: select that
        training run's best.pt; check: start the check at once."""
        if not self.validate_inputs():
            return
        from AIO_QC.widgets.improve_dialog import ImproveDialog
        dlg = self.open_dialogs.get("improve")
        if dlg is None:
            dlg = ImproveDialog(self)
            dlg.tool_key = "improve"
            self.open_dialogs["improve"] = dlg
        if run is not None:
            best = Path(run) / "weights" / "best.pt"
            dlg.refresh_models(select=str(best) if best.is_file() else None)
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()
        if check and dlg.model_path():
            dlg.check()
        return dlg

    def open_augment(self):
        """Image Augmentation (Models tab) — one window; it follows the open dataset and the shown image."""
        from AIO_QC.widgets.augment_dialog import AugmentDialog
        dlg = getattr(self, "augment_dialog", None)
        if dlg is not None:
            try:
                dlg.isVisible()
            except RuntimeError:                           # deleted with its close
                dlg = None
        if dlg is None:
            dlg = AugmentDialog(self)
            dlg.setWindowFlag(Qt.WindowType.Window, True)
            self.augment_dialog = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def _dataset_yaml_for_training(self):
        s = self.session
        if s.type.value == "cls_folders":                   # a classifier trains from the folder itself
            from AIO_QC.core.dataset import class_folder_splits
            return s.root if dict(class_folder_splits(s.root)).get("train") else None
        if s.is_yolo:
            for base in (s.root, s.root.parent):
                for n in ("dataset.yaml", "data.yaml", "data.yml"):
                    if (base / n).is_file():
                        return base / n
            return None
        try:
            saved = json.loads(utils.app_settings().value("convert_yolo/" + str(s.root), "") or "{}")
        except ValueError:
            saved = {}
        out = saved.get("out")
        if out and (Path(out) / "dataset.yaml").is_file():
            return Path(out) / "dataset.yaml"
        return None

    def open_testing(self, video=None, start=None, end=None):
        """Testing (Models tab) - one window per app, on the dataset open here when there is one. Never opens or
        drives the Frame Extractor's player."""
        from AIO_QC.widgets.testing.model_lab import ModelLab
        from AIO_QC.widgets.testing.qc_bridge import QCBridge
        lab = getattr(self, "testing", None)
        if lab is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                ff = self.settings.value("fx/playback/ffmpeg", "") or ""
                lab = ModelLab(self, {}, lambda: _find_ffmpeg(ff), __version__, qc=QCBridge(self))
                lab.setWindowFlag(Qt.WindowType.Window, True)
                lab.setWindowIcon(utils.newIcon("tool-testing"))
                lab.help_requested.connect(lambda page: self.show_help_page(page or "09-model-lab.html"))
            finally:
                QApplication.restoreOverrideCursor()
            self.testing = lab
        else:
            lab.use_qc_dataset()
        if video:
            lab.add_video(video, start, end)
        fresh = not lab.isVisible()
        lab.show()
        if fresh:
            anim.window_fade_in(lab)
        lab.raise_()
        lab.activateWindow()
        return lab

    def _shell_call(self, name, *args):
        """The app shell (both modes) — absent when the window runs on its own (tests, embedding)."""
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, "AIO QC", "The other modes are available when AIO QC is started normally.")
            return
        getattr(shell, name)(*args)

    def open_autoannotate(self):
        from AIO_QC.tools.frame_extractor.widgets.autoannotate_window import open_window
        open_window(self)

    def open_data_prep(self):
        from AIO_QC.widgets.data_prep import DataPrepDialog
        dlg = self.open_dialogs.get("data_prep")
        if dlg is None:
            dlg = DataPrepDialog(self)
            dlg.tool_key = "data_prep"
            self.open_dialogs["data_prep"] = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------ first steps (3.0)
    def _offer_undo(self, backup, what: str, files: int):
        """Every change to many files at once ends with a notification that can take it back."""
        if files < 1 or self.session is None:
            return
        session = self.session
        name = (what or "change").replace("_", " ")

        def undo():
            if self.session is not session:
                return
            keys, skipped = session.undo_backup(backup)
            msg = f"<b>Undone</b><br>{len(keys):,} file{'s' if len(keys) != 1 else ''} put back as they were"
            if skipped:
                msg += f" · {len(skipped):,} left alone (changed since)"
            self.toast.show_message(msg, "success", key="undo-bulk")
        self.toast.show_message(
            f"<b>Changed {files:,} file{'s' if files != 1 else ''}</b> · {name}<br>The originals are kept — "
            "Undo puts them back.", "success", actions=[("Undo", undo)], timeout_ms=14000, key="undo-bulk")

    def _offer_untrash(self, record):
        session = self.session
        n = len(record.get("moves", []))

        def undo():
            if self.session is not session:
                return
            back, left = session.untrash(record)
            msg = f"<b>Put back</b><br>{back:,} file{'s' if back != 1 else ''} restored from the trash"
            if left:
                msg += f" · {left:,} left there (something new is in their place)"
            self.toast.show_message(msg, "success", key="undo-trash")
        self.toast.show_message(
            f"<b>Moved {n:,} file{'s' if n != 1 else ''} to the trash</b><br>Nothing is deleted — they are in "
            ".aio_qc_trash inside the dataset.", "info", actions=[("Undo", undo), ("Show", self.open_trash)],
            timeout_ms=14000, key="undo-trash")

    def open_health_check(self):
        from AIO_QC.widgets.health_check import open_health_check
        return open_health_check(self)

    def open_sample_dataset(self):
        from AIO_QC.core.sample_data import PLANTED, sample_dataset
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            root = sample_dataset()
        finally:
            QApplication.restoreOverrideCursor()
        from AIO_QC.widgets import coach
        first = not coach.done("image")             # the tour comes first; its last steps point at the check
        self.open_dataset(root)
        if self.session is not None and not first:
            self.toast.show_message(
                f"<b>Sample dataset open</b><br>It has {len(PLANTED)} problems planted on purpose. Run a health "
                "check to find them — nothing here is your data.", "info",
                actions=[("Run health check", self.open_health_check)], timeout_ms=16000, key="sample")

    def tour_steps(self):
        from AIO_QC.widgets.coach import Step
        return [
            Step(lambda: self.open_dir_btn, "Your dataset",
                 "Open a <b>LabelMe folder</b> or a <b>YOLO dataset</b> here (Ctrl+O). Its name and file count show "
                 "next to it."),
            Step(lambda: self.file_list_dock, "Files",
                 "Every image of the dataset. Click one — or use <b>← →</b> to move through them. The search and "
                 "the filter show only the files you need."),
            Step(lambda: self.canvas, "The image and its annotations",
                 "Click a shape to select it, drag its corners to fix it, <b>Del</b> removes it. <b>Ctrl+Z</b> "
                 "undoes any change."),
            Step(lambda: self.unique_labels_dock, "Labels",
                 "Every class in the dataset with how often it is used. The eye hides a class on the image."),
            Step(lambda: self.health_btn, "Not sure where to start?",
                 "<b>Health check</b> runs the common checks in one go — unreadable files, missing partners, label "
                 "names spelt two ways, duplicates — and says what it found in plain words."),
            Step(lambda: self.tabs, "All the checks",
                 "Grouped by the question they answer: Dataset, Labels, Shapes, Models, Review. Click a tab to see "
                 "its tools (<b>Alt+1 … Alt+5</b>)."),
            Step(lambda: self.search_box if hasattr(self, "search_box") else None, "Find anything",
                 "<b>Ctrl+K</b> finds any command by name, <b>?</b> shows every shortcut, and every panel's "
                 "<b>?</b> opens its page of the guide."),
        ]

    def start_tour(self, force: bool = False):
        from AIO_QC.widgets import coach
        if self.session is None:                     # nothing to point at yet: show it on the sample dataset
            coach.mark_done("image")                 # (opening it must not start a second tour)
            self.open_sample_dataset()
            if self.session is None:
                return None
        return coach.run(self, self.tour_steps(), "image", force=force)

    def _disabled_reason(self, act) -> str:
        if self.session is None:
            return "open a dataset first (Ctrl+O), or try the sample dataset (Help ▸ Open a Sample Dataset)."
        for t in QC_TOOLS:
            from AIO_QC.app_common import tool_supported
            if self.qc_actions.get(t.key) is act and t.dialog is not None and not tool_supported(t, self.session):
                return f"this check does not apply to {self.session.type.title} datasets."
        if act in (self.a.undo_edit,):
            return "nothing to undo yet."
        if act in (self.a.redo_edit,):
            return "nothing to redo."
        return ""

    def apply_ui_level(self, simple: bool):
        """Simple menus: the QC tabs leave out the same specialist tools as the menus."""
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            if hasattr(page, "refilter"):
                page.refilter()

    def _build_start_page(self):
        from AIO_QC.widgets import start_page as SP
        from AIO_QC.widgets import ui_level
        shell = lambda: getattr(QApplication.instance(), "shell", None)

        def other(mode):
            return lambda: shell() and shell().switch(mode)

        def level_text():
            return "Show all menus" if ui_level.simple() else "Simple menus"
        lvl_link = SP._link(SP.Action(level_text(), lambda: ui_level.set_simple(not ui_level.simple())))
        lvl_link.setToolTip("Simple menus show the everyday commands; Ctrl+K always finds every command")
        self._start_level_link = lvl_link
        ui_level.bus.changed.connect(self._level_link_text)          # a bound slot: let go of with the window
        footer = [SP.Action("Quick tour", lambda: self.start_tour(force=True)),
                  SP.Action("Keyboard shortcuts  (?)", lambda: self.toggle_keymap()), lvl_link]
        return SP.StartPage(
            "Welcome to AIO QC", "What would you like to do?",
            cards=[SP.Card("image", "Check annotations",
                           "Find and fix problems in a LabelMe or YOLO dataset: wrong or misspelt labels, missing "
                           "files, duplicates, boxes that don't fit.",
                           [SP.Action("Open a dataset folder…", self.open_dir, primary=True, icon="folder-line",
                                      hint="Ctrl+O"),
                            SP.Action("Try a sample dataset", self.open_sample_dataset)], current=True),
                   SP.Card("video", "Get frames from videos",
                           "Play videos frame by frame and save the frames you need — or turn a whole folder of "
                           "videos into unique, auto-annotated images.",
                           [SP.Action("Open the Frame Extractor", other("video"))]),
                   SP.Card("data", "Clean a table",
                           "Open a CSV, Excel or Parquet file, fix it step by step, chart it and export it.",
                           [SP.Action("Open Data Studio", other("data"))])],
            recents=lambda: SP.all_recents(shell(), first="image"),
            drop_hint="You can also drop a dataset folder anywhere on this window.",
            footer=footer, parent=self)

    def _level_link_text(self, simple: bool):
        link = getattr(self, "_start_level_link", None)
        from PyQt6 import sip
        if link is not None and not sip.isdeleted(link):
            link.setText("Show all menus" if simple else "Simple menus")

    def _sync_status_bar(self, *_):
        """The welcome page has no status bar — unless a background job is running (it shows there)."""
        ind = getattr(self, "_task_indicator", None)
        if ind is not None and not getattr(self, "_status_hooked", False):
            self._status_hooked = True
            ind.active_changed.connect(self._sync_status_bar)
        busy = ind is not None and ind.active()
        self.statusBar().setVisible(self.session is not None or busy)

    def _sync_start_page(self):
        """Nothing open: the welcome page covers the window below the toolbar; a dataset open: it goes."""
        if not hasattr(self, "toolbar"):
            return
        page = getattr(self, "start_page", None)
        self._sync_status_bar()
        if self.session is None:
            if page is None:
                page = self.start_page = self._build_start_page()
                self.installEventFilter(self._start_filter())
            page.refresh()
            self._place_start_page()
            page.show()
            page.raise_()
        elif page is not None:
            page.hide()

    def _start_filter(self):
        win = self

        class _F(QObject):
            def eventFilter(self, obj, ev):
                if ev.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest, QEvent.Type.Show):
                    win._place_start_page()
                return False
        self._sp_filter = _F(self)
        return self._sp_filter

    def _place_start_page(self):
        page = getattr(self, "start_page", None)
        if page is None or not page.isVisible() and self.session is not None:
            return
        top = 0
        for w in (self.menuBar(), self.toolbar):
            if w is not None and w.isVisible():
                top = max(top, w.geometry().bottom() + 1)
        sb = self.statusBar()
        bottom = sb.geometry().top() if sb is not None and sb.isVisible() else self.height()
        page.setGeometry(0, top, self.width(), max(0, bottom - top))

    def show_guide(self):
        """F1: the app's one Help Center, at the Image QC guide."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, "image")

    def show_help_page(self, page=None):
        """Help for one feature. Frame Extractor and Testing chapters live in the shared Help Center."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, page)

    def reset_all(self):
        """Soft reset: all labels visible, filters cleared, fit to window."""
        self.hidden_labels.clear()
        self.refresh_label_list()
        self.file_dock.clear_all_filters()
        self.canvas.clearSelection()
        self.load_canvas_shapes()
        self.update_annotations_list()
        self.canvas.apply_default_zoom()

    def _focus_search(self):
        self.file_list_dock.show()
        self.file_dock.search.setFocus()
        self.file_dock.search.selectAll()

    def _toggle_tags(self, on):
        self.canvas.show_tags = on
        self.load_canvas_shapes()

    def _toggle_fill(self, on):
        self.canvas.fill_shapes = on
        self.load_canvas_shapes()

    def _set_pixel_view(self, on):
        self.settings.setValue("view/pixel_view", bool(on))
        self.canvas.pixmap_item.pixel_view = bool(on)
        self.canvas._set_smooth(self.canvas.current_zoom, False)
        self.canvas.pixmap_item.invalidate()

    # ---- model predictions layer (Prediction Explorer data)
    def pred_layer_on(self) -> bool:
        return bool(self.a.pred_layer.isChecked()) if hasattr(self, "a") else False

    def set_pred_layer(self, on):
        on = bool(on)
        self.settings.setValue("view/pred_layer", on)
        if self.a.pred_layer.isChecked() != on:
            self.a.pred_layer.blockSignals(True)
            self.a.pred_layer.setChecked(on)
            self.a.pred_layer.blockSignals(False)
        dlg = self.open_dialogs.get("model_check")
        panel = getattr(dlg, "explorer", None)
        if panel is not None and panel.layer_action.isChecked() != on:
            panel.layer_action.blockSignals(True)
            panel.layer_action.setChecked(on)
            panel.layer_action.blockSignals(False)
        self.refresh_pred_layer()

    def refresh_pred_layer(self):
        items = self.pred_layer_items(self.current_key, self.canvas.pixmap_item.pixmap().width())
        if items:
            self.canvas.set_pred_layer(items)
        else:
            self.canvas.clear_pred_layer()

    def pred_layer_items(self, key, pix_width: int) -> list:
        """The predictions layer of one file (View ▸ Model Predictions) for an image `pix_width` pixels wide —
        [] when the layer is off or the file has none. Also used by the batch screenshots."""
        if not self.pred_layer_on() or not self.session or not key:
            return []
        try:
            from AIO_QC.widgets.explorer import layer_objects
            from AIO_QC.widgets.explorer.paint import object_color, object_text
            objs, d = layer_objects(self, key)
        except Exception as exc:                          # the viewer must never break for the layer
            logger.warning(f"predictions layer: {exc}")
            objs, d = [], None
        if not objs:
            return []
        i = d.index_of(key)
        W = float(d.img_w[i] or 0) if i is not None else 0.0
        k = (pix_width / W) if W and pix_width else 1.0                  # predictions are in image pixels
        dlg = self.open_dialogs.get("model_check")
        panel = getattr(dlg, "explorer", None)
        mode = panel.view.color if panel is not None and panel.data is d else "match"
        items = []
        for o in objs:
            col = object_color(o, mode, self.label_colors, d.model_colors)
            items.append({"box": tuple(v * k for v in o.box),
                          "poly": None if o.poly is None else [(x * k, y * k) for x, y in o.poly],
                          "kpts": None if o.kps is None else [(x * k, y * k, c) for x, y, c in o.kps],
                          "text": object_text(o, d.model_names, with_tag=True), "color": col.name()})
        return items

    def _on_mode_changed(self, mode):
        (self.a.select if mode == Canvas.SELECT else self.a.drag).setChecked(True)
        sel = mode == Canvas.SELECT
        self.mode_label.setIcon("cursor-line" if sel else "hand-line")
        self.mode_label.setText("Select" if sel else "Drag")

    # =====================================================================
    # Zoom
    # =====================================================================
    def updateZoomSpinbox(self):
        self._sync_zoom_buttons()
        if self.canvas.pixmap_item.pixmap().isNull():
            self.zoom_slider.set_zoom(None)
            return
        self.zoom_spin.set_zoom(self.canvas.current_zoom)
        self.zoom_slider.set_zoom(self.canvas.current_zoom)

    def _zoom_live(self, zoom: float):
        """Zoom box and slider follow the canvas while it glides (like Windows Photos: N clicks, N updates)."""
        if not self.canvas.pixmap_item.pixmap().isNull():
            self.zoom_spin.set_zoom(zoom)
            if self.zoom_slider.isVisible():
                self.zoom_slider.set_zoom(zoom)
        if not self.canvas.is_animating():           # mid-glide: once, when it lands (zoomFinished)
            self._sync_zoom_buttons()

    def _zoom_from_slider(self, zoom: float):
        """Live while dragging: no glide (the slider is the animation), around the view centre."""
        c = self.canvas
        if zoom > 0 and not c.pixmap_item.pixmap().isNull():
            c.animate_to(zoom, anchor_scene=c._view_center(), anchor_view=c.viewport().rect().center(), duration=0)
            self.zoom_spin.set_zoom(c.current_zoom)

    def _sync_zoom_buttons(self):
        """Fit Window / Actual Size look pressed whenever the view IS fitted / at 100 % — however it got there
        (button, shortcut, double-click, wheel, zoom box, a new image, a window resize)."""
        if not hasattr(self, "a"):
            return
        c = self.canvas
        on_fit = on_actual = False
        if not c.pixmap_item.pixmap().isNull():
            z = c._target_zoom if c.is_animating() else c.current_zoom
            fit = c.compute_fit_scale()
            on_fit = fit > 0 and abs(z / fit - 1.0) < 0.004
            on_actual = abs(z - 1.0) < 0.004
        for act, on in ((self.a.fit, on_fit), (self.a.actual, on_actual)):
            if act.isChecked() != on:
                act.setChecked(on)              # (not signal-blocked: the toolbar buttons follow `changed`)

    def onZoomSpinboxChanged(self, zoom: float):
        """ZoomWidget: typed value, preset, wheel or ↑ / ↓ (1.0 = 100 %)."""
        if zoom > 0 and not self.canvas.pixmap_item.pixmap().isNull():
            self.canvas.zoom_to(zoom)

    # =====================================================================
    # 🔀 Label mapping (view-time)
    # =====================================================================
    def open_label_map(self, prefill: dict | None = None, focus: str | None = None, scope: int = 2):
        """The single place to rename / merge / map labels (view-only, or written to files)."""
        if self.validate_inputs():
            LabelMapDialog(self, prefill=prefill, focus=focus, scope=scope).exec()

    def set_label_map(self, mapping: dict[str, str]):
        s = self.session
        if s is None or dict(mapping) == dict(s.label_map):
            return
        for d in list(self.open_dialogs.values()):
            d.close()                               # their results used the previous names
        s.set_label_map(mapping)
        self.visible_labels.clear()
        self.hidden_labels.clear()
        for name in s.adapter.class_names:
            utils.update_label_colors(name, self.label_colors)
        self.reload_current_annotation()
        self.refresh_label_list()
        self._update_mapping_indicator()
        self.refresh_status_buttons()
        self.statusBar().showMessage(f"🔀 Label mapping: {len(s.label_map)} label(s) mapped" if s.label_map
                                     else "Label mapping disabled", 6000)

    def _update_mapping_indicator(self):
        n = len(self.session.label_map) if self.session else 0
        self.mapping_btn.setText(f"Mapping ({n})")
        self.mapping_btn.setVisible(n > 0)
        self.a.label_map.setText(f"Label Mapping… ({n} active)" if n else "Label Mapping…")

    # =====================================================================
    # ✓ Review markers
    # =====================================================================
    def mark_review(self, status, note=None):
        """Mark the open file, then (auto-advance) open the NEXT one. At the end of the list it stays on the last file
        — it never goes back. With a review filter the marked file may leave the list: it stays open (parked) and the
        file after it is next (core of FileDock's parking; works with live inserts / removals too)."""
        key = self.current_key
        if not self.session or not key or key not in self.session.by_key:
            return
        self.session.set_review([key], status, note)          # the dock parks the file if it left the list
        parked = self.file_dock.current_key() != key
        nxt = self.file_dock.key_at_offset(1)                 # strictly forward (from the gap when parked)
        advance = status is not None and self.a.auto_advance.isChecked()
        if (advance or parked) and nxt is not None:
            self.file_dock.select_key(nxt)
        elif advance or parked:
            mark = {REVIEW_OK: "✓ OK", REVIEW_FLAG: "⚑ flagged", REVIEW_FIXED: "fixed"}.get(status, "cleared")
            self.statusBar().showMessage(f"Marked {mark} · that was the last file in the list", 5000)
            self._update_position()

    def flag_with_note(self):
        key = self.current_key
        if not self.session or not key:
            return
        entry = self.session.review.get(key) or {}
        note, ok = QInputDialog.getText(self, "Flag with note", "What needs fixing?", text=entry.get("note", ""))
        if ok:
            self.mark_review(REVIEW_FLAG, note.strip())

    def goto_next_unreviewed(self):
        if not self.session:
            return
        status = self.session.review.status
        k = self.file_dock.next_matching(lambda key: status(key) is None)     # stops at the first hit
        if k is not None:
            self.file_dock.select_key(k)
            return
        self.statusBar().showMessage("Every file in the list has been reviewed", 6000)

    def _on_review_changed(self, keys):
        self.history_panel.touched(keys)
        self.file_dock.review_changed(keys)
        self.filmstrip.refresh_labels()
        self._update_review_badge()
        self._update_review_progress()
        self._review_save.start()

    def _save_review(self, compact: bool = False):
        """Periodic: append the new marks to the journal. compact=True (closing): one full rewrite."""
        if self.session is not None:
            try:
                self.session.review.save(compact=compact)
                if compact:
                    self.session.issues.save()
            except OSError as exc:
                self.log_errors([f"[review] could not save: {exc}"])

    def _update_review_badge(self):
        e = self.session.review.get(self.current_key) if self.session and self.current_key else None
        hud = getattr(self, "hud", None)
        if hud is not None:
            hud.set_flag(e["status"] if e else None, (e or {}).get("note", "") or "")
        if not e:
            self.review_badge.set_status(None, tip="Not reviewed yet · S = OK · X = flag · Shift+X = flag with note")
            return
        note, st = e.get("note", "") or "", e["status"]
        if st == REVIEW_FIXED:
            tip = (note or "Corrected in an editor") + "\nPress S after checking it to mark it OK"
        elif st == REVIEW_OK:
            tip = f"Reviewed {e.get('ts', '')}"
        else:
            tip = (f"Flag note: {note}\n" if note else "") + f"Flagged {e.get('ts', '')}"
        self.review_badge.set_status(st, note, tip.strip())

    def _update_flagged_count(self, c=None):
        """Status bar: ⚑ N flagged (dataset-wide, running count)."""
        if not self.session:
            self.flagged_btn.setVisible(False)
            return
        n = (c if c is not None else self.session.review_totals()).get(REVIEW_FLAG, 0)
        on = self.file_dock.review_box.currentData() == REVIEW_FLAG
        self.flagged_btn.setText(f"{n:,} flagged")
        self.flagged_btn.setVisible(True)
        # one fixed slot, as wide as the largest count this dataset can have (bold): its neighbours never move
        from PyQt6.QtGui import QFont, QFontMetrics
        bf = QFont(self.flagged_btn.font())
        bf.setBold(True)
        most = len(self.session.samples) if self.session is not None else 9999
        need = QFontMetrics(bf).horizontalAdvance(f"{max(most, 99):,} flagged".translate(
            str.maketrans("0123456789", "8888888888"))) + self.flagged_btn.iconSize().width() + 22
        if self.flagged_btn.width() != need:
            self.flagged_btn.setFixedWidth(need)
        self.flagged_btn.setEnabled(n > 0 or on)
        self.flagged_btn.setToolTip(("Showing only the flagged files — click to show all files" if on else
                                     "Click to show only the flagged files") + " (file list ▸ review filter)")
        utils.style(self.flagged_btn, "QToolButton{color:@warn; font-weight:bold; padding:0 6px;}" if n else
                    "QToolButton{color:@muted; padding:0 6px;}")

    def toggle_flagged_filter(self):
        box = self.file_dock.review_box
        want = None if box.currentData() == REVIEW_FLAG else REVIEW_FLAG
        box.setCurrentIndex(max(0, box.findData(want)))
        self._update_flagged_count()

    def _update_review_progress(self):
        if not self.session:
            self.review_progress.setText("")
            self._update_flagged_count()
            return
        c = self.session.review_totals()                       # running counts: O(1) per keypress
        self._update_flagged_count(c)
        n = len(self.session.samples)
        done = c.get(REVIEW_OK, 0) + c.get(REVIEW_FLAG, 0) + c.get(REVIEW_FIXED, 0)
        pct = done * 100 / max(n, 1)
        c = {k: c.get(k, 0) for k in (REVIEW_OK, REVIEW_FLAG, REVIEW_FIXED)}
        self.review_progress.setText(f"Reviewed {done:,} / {n:,} · {pct:.1f} %\n"
                                     f"{c[REVIEW_OK]:,} OK · {c[REVIEW_FLAG]:,} flagged · {c[REVIEW_FIXED]:,} fixed")

    def save_flagged(self):
        if not self.session:
            return
        self._review_save.stop()
        self._save_review()                                # marks are on disk before anything reads them
        dlg = FlaggedFilesDialog(self)
        dlg.exec()

    def export_review(self):
        import csv
        if not self.session:
            return
        from AIO_QC.utils.save_dirs import ask_save
        path, _ = ask_save(self, "Export review report", f"{self.session.root.name}_review_report.csv", "CSV (*.csv)",
                           kind="reports", fallback=self.session.root.parent)
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "split", "status", "note", "reviewed_at"])
            for s in self.session.samples:
                e = self.session.review.get(s.key) or {}
                w.writerow([s.ann_path or s.image_path, s.split, e.get("status", "unreviewed"),
                            e.get("note", ""), e.get("ts", "")])
        self.statusBar().showMessage(f"Review report written: {path}", 6000)

    # =====================================================================
    # Filmstrip, recent datasets, drag & drop
    # =====================================================================
    def _orient_filmstrip(self, area=None):
        """Left/right dock -> vertical column; top/bottom -> horizontal strip; floating -> by window shape."""
        dock = self.filmstrip_dock
        if dock.isFloating():
            vertical = dock.height() > dock.width()
        else:
            area = area if area is not None else self.dockWidgetArea(dock)
            vertical = area in (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea)
        self.filmstrip.set_vertical(vertical)

    def _update_filmstrip(self):
        if self.session and self.current_key and self.filmstrip_dock.isVisible():
            keys = self.file_dock.visible_keys(around=self.current_key, before=FILMSTRIP_WINDOW,
                                               after=FILMSTRIP_WINDOW)
            self.filmstrip.show_around(keys, self.current_key)

    def _recent(self) -> list[tuple[str, str]]:
        raw = self.settings.value("recent_datasets", []) or []
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for item in raw:
            path, _, dtype = str(item).partition("|")
            if path:
                out.append((path, dtype))
        return out

    def _add_recent(self, root: Path, dtype: DatasetType):
        entry = f"{root}|{dtype.value}"
        items = [f"{p}|{t}" for p, t in self._recent() if p != str(root)]
        self.settings.setValue("recent_datasets", [entry] + items[:MAX_RECENT - 1])

    def _build_recent_menu(self):
        self.recent_menu.clear()
        recent = self._recent()
        if not recent:
            act = self.recent_menu.addAction("(none yet)")
            act.setEnabled(False)
            return
        for i, (path, dtype) in enumerate(recent):
            exists = Path(path).is_dir()
            try:
                dt = DatasetType(dtype)
                label = dt.title
            except ValueError:
                dt, label = None, "?"
            act = self.recent_menu.addAction(f"&{(i + 1) % 10}  {Path(path).name}  —  {label}")
            act.setToolTip(path)
            act.setEnabled(exists)
            act.triggered.connect(lambda _c=False, p=path, t=dt: self.open_dataset(Path(p), t))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear list", lambda: self.settings.setValue("recent_datasets", []))
