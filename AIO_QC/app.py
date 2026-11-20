#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC App — annotation quality control for LabelMe and YOLO datasets.

Dependencies: pip install PyQt6 numpy opencv-python loguru
Optional:     pip install natsort pillow pyyaml orjson
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QMainWindow,
)

from AIO_QC import utils
from AIO_QC.widgets import (
    IndexWorker, PreloadSignals, Session,
)
from AIO_QC.widgets import anim
from AIO_QC.widgets.guidelines import CanvasGuide
from AIO_QC.widgets.keymap import PanelCycler
from AIO_QC.widgets.lifecycle import set_exit_reason, stop_everything
from AIO_QC.widgets.toast import ToastManager
from AIO_QC.core.ocr_service import OcrService
from AIO_QC.widgets.inference_service import InferenceService


from AIO_QC.app_common import (  # noqa: F401  (also re-exported: AIO_QC.app.QC_TOOLS …)
    GROUPS, IMAGE_ADVANCED, MAX_RECENT, PIXMAP_CACHE_BYTES, QCTool, QC_TOOLS, TOOL_NOT_HERE, _find_ffmpeg, natsorted)
from AIO_QC.main_window.ui import UIMixin
from AIO_QC.main_window.dataset import DatasetMixin
from AIO_QC.main_window.display import DisplayMixin
from AIO_QC.main_window.tools import ToolsMixin
from AIO_QC.main_window.extras import ExtrasMixin


class MainWindow(UIMixin, DatasetMixin, DisplayMixin, ToolsMixin, ExtrasMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self._update_title()
        self.setWindowIcon(utils.appIcon())
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.75)
        self.resize(w, int(w * 0.55))
        self.settings = utils.app_settings()
        from AIO_QC.widgets.status_fields import PaintStatusBar
        self.setStatusBar(PaintStatusBar(self))           # pointer · selection · image size, then messages
        from AIO_QC.widgets import motion
        anim.ENABLED = motion.enabled()                    # one switch for every animation (ui/animations)
        from AIO_QC.core import backups as _backups
        _backups.EXACT_IMAGE_DATA = self.settings.value("storage/exact_image_data", False, type=bool)
        self.inference = InferenceService(self.settings, self)       # shared model server (Model Setup / Check)
        from AIO_QC.widgets.train_dialog import TrainWatch
        self.train_watch = TrainWatch(self)                         # tells when a training run ends
        self._cleanup_temp_envs()

        self.session: Session | None = None
        self.user_classes: list[str] | None = None
        self.user_classes_path: Path | None = None
        self.current_key: str | None = None
        self.current_ann = None
        self._pending_shape: int | None = None
        self._pending_extras: tuple = (None, None)
        self._index_started = 0.0
        self.label_colors: dict[str, QColor] = {}
        self.visible_labels: set[str] = set()
        self.hidden_labels: set[str] = set()      # survives label-list rebuilds
        self._display_by_key: dict[str, dict] = {}  # image adjustments made per file (this session)
        self._display_owner = None                # the file a pending slider change belongs to
        self.error_log: list[str] = []
        self.open_dialogs: dict[str, QDialog] = {}
        self.editor_dialog = None                  # active external-editor session window (Fix in Editor)
        self.watcher = self.tracker = None         # live updates: background scanner + change tracker
        self.sweep = None                          # background image-integrity check
        self.broken: dict = {}                     # key -> [(code, severity, message)] (integrity errors)
        self._issues_save_timer = QTimer(self, singleShot=True, interval=2000,
                                         timeout=lambda: self.session.issues.save() if self.session else None)
        self.ocr = OcrService(log=lambda line: logger.debug(f"[ocr] {line}"))   # OCR engine process (warm)
        from AIO_QC.core.embed_service import EmbedService
        self.embedder = EmbedService(log=lambda line: logger.debug(f"[embed] {line}"))   # Visual Similarity model
        self.embed_index = None                    # vectors of the last Visual Similarity run (Crop Gallery sorts)
        self._external_log: list = []
        from AIO_QC.utils import memory as _mem
        # the preload window (1 back, 3 ahead) + the files just visited; at most 512 MB and never more than ~10 % of
        # this computer's memory (8 GB: 512 MB, 4 GB: ~410 MB)
        self.pixmap_cache = utils.LRUCache(capacity=10, max_bytes=_mem.cache_budget(PIXMAP_CACHE_BYTES, share=0.10))
        self._low_mem_told = 0.0
        self.generation = 0                       # invalidates stale background results
        self._inflight: set[str] = set()
        self.index_worker: IndexWorker | None = None
        self.preload_signals = PreloadSignals()
        self.preload_signals.loaded.connect(self._on_preloaded)
        self._labels_refresh = QTimer(self, singleShot=True, interval=300, timeout=self.refresh_label_list)
        self._focus_state = None                  # window layout saved while Focus Mode hides the panels
        self._palette = None

        self._build_canvas()
        self._build_docks()
        self._build_actions()
        self._build_toolbar()
        self._build_qc_panel()
        self._build_menus()
        self._build_statusbar()
        for act in (self.a.fit, self.a.actual):     # pressed look = "the view is like this now"
            act.setCheckable(True)
            act.triggered.connect(lambda _c=False: QTimer.singleShot(0, self._sync_zoom_buttons))
        from AIO_QC.widgets.inline_editor import InlineEditor
        self.inline = InlineEditor(self)            # edits made directly on the image
        from AIO_QC.widgets.model_assist import ModelAssist
        self.assist = ModelAssist(self)             # the models suggest labels; click to accept
        self.canvas_guide = CanvasGuide(self)       # the selected class's guideline, over the image
        from AIO_QC.widgets.sketch import RoiMode, SketchMode
        self.sketch = SketchMode(self)              # P: draw on the image to explain something
        self.roi = RoiMode(self)                    # O: snipping-tool style ROI boxes with real coordinates
        self.panel_cycler = PanelCycler(self)       # F6
        self._keymap = None
        self._restore_settings()
        self._orient_filmstrip()
        self._watch_theme()
        self.toast = ToastManager(self)             # colour-coded pop-up notifications
        QApplication.instance().applicationStateChanged.connect(self._on_app_state)
        self.settings.setValue("get_started/shown", True)    # first start: the welcome page (and the tour) instead
        self.settings.setValue("guide_shown", True)
        self.set_dataset_actions_enabled(False)
        self.refresh_status_buttons()
        from AIO_QC.widgets import ui_level
        lvl = ui_level.install(self, advanced=IMAGE_ADVANCED, reason=self._disabled_reason)
        self.menus.view.addSeparator()
        self.menus.view.addAction(lvl.action)
        lvl.refresh_all()
        lvl._apply(ui_level.simple())
        self._sync_start_page()

    # ------------------------------------------------------------------ Display tab
    DISPLAY_DEFAULTS = {"brightness": 0, "contrast": 0, "gamma": 100, "auto": False, "fill": 55, "outline": 100, "tag": 100}
    IMAGE_KEYS = ("brightness", "contrast", "gamma", "auto")      # per image (the rest is annotation styling)

    def closeEvent(self, event):
        if getattr(self, "_quit_reason", None) is None and not self._end_editor_session("quit"):
            event.ignore()
            logger.debug("close cancelled: an editor session is still active")
            return
        if not self._ask_about_training():
            event.ignore()
            return
        set_exit_reason(getattr(self, "_quit_reason", None) or "the user closed the window")
        logger.info(f"closing AIO QC ({getattr(self, '_quit_reason', None) or 'window closed'}) — saving state")
        self._stop_watcher()
        self._stop_sweep()
        self.ocr.shutdown()
        self.embedder.shutdown()
        self._save_review(compact=True)
        self._stop_index()
        if self.session is not None:
            self.session.close()
        self.inference.shutdown()          # stops the model server; deletes a temporary env if asked to
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self._focus_state if self._focus_state is not None else self.saveState())
        self._stop_index()
        for d in list(self.open_dialogs.values()):
            d.close()
        if getattr(self, "testing", None) is not None:
            self.testing._dirty = False                  # its session is autosaved (with the dataset)
            self.testing.close()
        if getattr(self, "train_dialog", None) is not None:
            self.train_dialog.close()
        try:
            if getattr(self, "augment_dialog", None) is not None:
                self.augment_dialog.close()                # (stops a running Generate; what is written stays)
        except RuntimeError:
            pass
        stop_everything(self, "app closing")             # nothing left running when Qt tears the window down
        logger.info("state saved, background work stopped")
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast"):
            self.toast.reposition()
        if hasattr(self, "dataset_label"):           # the path's fixed area follows the window width only
            self.dataset_label.set_area_width(min(420, max(180, int(self.width() * 0.22))))

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "toast"):              # notifications posted while another mode was in front
            self.toast.reposition()

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.PaletteChange and getattr(self, "_dark", None) is not None \
                and utils.is_dark() != self._dark:
            self._on_theme_changed()
        super().changeEvent(event)

    # (light theme, dark theme). "auto" (default): a soft warm light grey in the light theme (like labelme's light
    # canvas), near-black in the dark theme
    CANVAS_BG = {"auto": ("#E9E7E1", "#0E0F11"), "dark": ("#24262A", "#0E0F11"), "light": ("#E9E7E1", "#2A2D33"),
                 "window": (None, None)}

    TOOL_NOT_HERE = TOOL_NOT_HERE           # (module level: the guide's "Which tool for what?" uses it too)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        if p.is_file():
            p = p.parent
        if p.is_dir():
            self.settings.setValue("last_dir", str(p))
            self.open_dataset(p)
