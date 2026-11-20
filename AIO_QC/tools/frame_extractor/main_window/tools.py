# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor window — markers, timeline, models, auto-annotate, batch, save, commands, help, first steps, gallery, tracking, keyframe index.

A mixin of AIO_QC.tools.frame_extractor.app.MainWindow: the methods use the window's own attributes, set up in its __init__.
"""
import os
from loguru import logger
from PyQt6.QtWidgets import (
    QMessageBox, QApplication, QFileDialog, QMenu,
)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor, QKeySequence, QPixmap
from AIO_QC.tools.frame_extractor import __appname__
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.core.metrics import sharpness
from AIO_QC.tools.frame_extractor.core.sources import format_time
from AIO_QC.tools.frame_extractor.widgets import (
    BatchQueueDialog,
)
from AIO_QC.tools.frame_extractor.core.common import model_hex
from AIO_QC.tools.frame_extractor.widgets.detect_worker import DetectWorker
from AIO_QC.tools.frame_extractor.widgets.models_dialog import (
    ModelsDialog,
    save_model_cfg,
)
from AIO_QC.tools.frame_extractor.widgets.markers_dock import MarkerWorker
from AIO_QC.tools.frame_extractor.widgets.track_worker import TrackWorker
from AIO_QC.tools.frame_extractor.widgets.command_palette import CommandPalette
from AIO_QC.tools.frame_extractor.widgets.shortcuts_dialog import ShortcutsDialog
from AIO_QC.tools.frame_extractor.widgets.gallery import ReviewGallery
from AIO_QC.utils.qt_utils import later



class ToolsMixin:


    # ============================================================ markers
    def marker_out_dir(self):
        name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "video"
        return os.path.join(self.output_dir, name, "markers")

    def refresh_markers(self, save=True):
        states = {f: ("saved" if m.get("saved") else m.get("src") == "suggested" and "suggested" or "pending")
                  for f, m in self.markers.items()}
        self.strip.set_markers(states)
        if hasattr(self, "tc"):
            self.update_tc_context()
        busy = self.marker_worker.pending() if self.marker_worker else 0
        self.mdock.set_markers(self.markers, lambda f: self.markers.get(f, {}).get("sharp"),
                               self.marker_out_dir() if self.video_path else "", busy)
        if save:
            self.save_session()

    def _marker_frame(self):
        if self.current is None or self.current.preview:
            return None
        return self.current

    def toggle_marker(self):
        fr = self._marker_frame()
        if fr is None:
            return
        if fr.idx in self.markers:
            self.remove_markers([fr.idx])
            self.osd(f"Marker #{fr.idx} removed")
            return
        self.add_marker(fr.idx, fr.bgr)
        pend = sum(1 for m in self.markers.values() if not m.get("saved"))
        self.osd(f"Marker #{fr.idx}" + (" · saving" if self.mdock.auto.isChecked() else f" · {pend} pending"))

    def add_marker(self, idx, bgr=None, src="manual", refresh=True):
        roi = list(self.roi) if (self.roi and self.mdock.crop.isChecked()) else None
        sharp = None
        if bgr is None and self.current is not None and self.current.idx == idx:
            img0, sc = self.frame_image()
            if img0 is not None and sc != 1.0:
                r = [int(v * sc) for v in roi] if roi else None
                img = img0[r[1]:r[3], r[0]:r[2]] if r else img0
                sharp = round(sharpness(img), 1) if img.size else None
        if bgr is not None:
            img = bgr[roi[1]:roi[3], roi[0]:roi[2]] if roi else bgr
            sharp = round(sharpness(img), 1) if img.size else None
        else:
            s = self.strip.nearest_sample(idx, max_dist=1)
            sharp = round(s.sharp, 1) if s else None
        self.markers[idx] = dict(roi=roi, saved=None, src=src, sharp=sharp)
        if self.mdock.auto.isChecked() and src == "manual":
            self.ensure_marker_worker().save(idx, roi)
        if refresh:
            self.refresh_markers()

    def remove_markers(self, frames):
        for f in frames:
            self.markers.pop(f, None)
            self.mdock.thumbs.pop(f, None)
        self.refresh_markers()

    def move_marker(self, old, new):
        if old not in self.markers or new in self.markers or not self.engine:
            return
        m = self.markers.pop(old)
        m.update(saved=None, sharp=None, src="manual")
        self.markers[self.engine.clamp(new)] = m
        self.mdock.thumbs.pop(old, None)
        self.refresh_markers()
        self.goto_frame(new)

    def jump_marker(self, direction):
        marks = sorted(self.markers)
        base = self.current_frame if self.is_playing else self.requested_frame
        cands = [m for m in marks if (m > base if direction > 0 else m < base)]
        if cands:
            self.goto_frame(cands[0] if direction > 0 else cands[-1])

    def ensure_marker_worker(self):
        if self.marker_worker is None and self.video_path:
            o = self.mdock.options()
            from AIO_QC.tools.frame_extractor.core.hashdb import default_db_path
            self.marker_worker = MarkerWorker(self.video_path, self.marker_out_dir(),
                                              self.engine.backend if self.engine else "auto", self.out_cfg,
                                              dict(self.model_cfg) if o["preannot"] != "none" else None,
                                              o["preannot"], default_db_path(self.output_dir))
            self.marker_worker.saved.connect(self.on_marker_saved)
            self.marker_worker.snapped.connect(self.on_marker_snapped)
            self.marker_worker.failed.connect(self.on_marker_failed)
            self.marker_worker.idle.connect(lambda: (self.refresh_markers(save=False), self._marker_task_tick()))
            self.marker_worker.saved.connect(lambda *_: self._marker_task_tick())
            self.marker_worker.start(QThread.Priority.LowPriority)
        return self.marker_worker

    def stop_marker_worker(self):
        if self.marker_worker is not None:
            self.marker_worker.stop()
            self.marker_worker = None
        if getattr(self, "_mtask", None) is not None:
            self._mtask.finish()
            self._mtask, self._mtask_total = None, 0

    def on_marker_options(self):
        o = self.mdock.options()
        if o["preannot"] != "none" and not self.model_cfg["models"]:
            self.flash_status("Pre-annotations need models: Settings → Models…")
        self.stop_marker_worker()        # restarted with the new options on next use
        self.save_session()

    def extract_markers(self, frames=None):
        if not self.video_path:
            return
        todo = [f for f in (frames or sorted(self.markers)) if f in self.markers and not self.markers[f].get("saved")]
        if not todo:
            self.osd("No pending markers")
            return
        w = self.ensure_marker_worker()
        for f in todo:
            w.save(f, self.markers[f].get("roi"))
        self._marker_task(len(todo))
        self.osd(f"Extracting {len(todo)} marker(s)…")
        self.refresh_markers(save=False)

    def _marker_task(self, n):
        """Marker saving in the shared background-task indicator (finished when the worker goes idle)."""
        from AIO_QC.widgets import tasks
        if getattr(self, "_mtask", None) is None:
            self._mtask = tasks.start(f"Saving markers · {os.path.basename(self.video_path or '')}",
                                      cancel=lambda: self.stop_marker_worker(), owner=self)
        self._mtask_total = getattr(self, "_mtask_total", 0) + n

    def _marker_task_tick(self):
        t = getattr(self, "_mtask", None)
        if t is None:
            return
        busy = self.marker_worker.pending() if self.marker_worker else 0
        total = max(1, getattr(self, "_mtask_total", 1))
        if not busy:
            t.finish()
            self._mtask, self._mtask_total = None, 0
        else:
            t.update(1 - busy / total, f"{busy} left")

    def on_marker_saved(self, idx, path, thumb):
        if self.sender() is not self.marker_worker:
            return
        if idx in self.markers:
            self.markers[idx]["saved"] = path
            self.mdock.thumbs[idx] = QPixmap.fromImage(thumb)
        self.refresh_markers()

    def on_marker_failed(self, idx, msg):
        self.flash_status(f"Marker {idx}: {msg}" if idx >= 0 else msg, 6000)
        logger.warning(f"marker {idx}: {msg}")

    def snap_markers(self, frames, radius, then_extract=False):
        frames = [f for f in (frames or [f for f, m in self.markers.items() if not m.get("saved")]) if f in self.markers]
        if not frames:
            self.osd("No markers to snap")
            return
        w = self.ensure_marker_worker()
        if then_extract:
            self._extract_after_snap |= set(frames)
        for f in frames:
            w.snap(f, radius, self.markers[f].get("roi"))
        self.osd(f"Snapping {len(frames)} marker(s) to the sharpest frame (±{radius})…")

    def on_marker_snapped(self, old, new, sharp):
        if self.sender() is not self.marker_worker or old not in self.markers:
            return
        m = self.markers.pop(old)
        target = new if new not in self.markers else old
        m.update(sharp=round(sharp, 1))
        if target != old:
            m.update(saved=None)
            self.mdock.thumbs.pop(old, None)
        self.markers[target] = m
        if old in self._extract_after_snap:
            self._extract_after_snap.discard(old)
            if not m.get("saved"):
                self.marker_worker.save(target, m.get("roi"))
        self.refresh_markers()

    def suggest_markers(self, mode, every_s, quiet=False):
        from AIO_QC.tools.frame_extractor.core.markers import suggest
        if not self.engine or not self.strip.samples:
            self.osd("Timeline analysis hasn't produced samples yet")
            return []
        samples = {s.idx: (s.sharp, max(0.0, s.motion), s.flagged) for s in self.strip.samples.values()}
        if mode == "flagged" and not any(v[2] for v in samples.values()):
            QMessageBox.information(self, "Suggest", "No model-flagged samples. Turn on Settings → "
                                    "'Run models in timeline analysis' (needs models).")
            return []
        lo = self.in_frame or 0
        hi = self.out_frame if self.out_frame is not None else self.frame_count - 1
        frames = suggest(samples, mode, lo, hi, self.frame_rate, every_s,
                         float(self.settings.value("batch/min_sharpness", 0) or 0), self.strip.cuts,
                         self.engine.timestamps)
        near = max(1, int(self.frame_rate / 4))
        added = [f for f in frames if not any(abs(f - m) <= near for m in self.markers)]
        for f in added:
            self.add_marker(f, None, src="suggested", refresh=False)
        self.refresh_markers()
        done = 100 * len(samples) // max(1, len(self.strip.grid))
        if not quiet:
            self.osd(f"{len(added)} suggestion(s)" + (f" · timeline {done}% analysed" if done < 100 else ""))
        return added

    def auto_pick(self):
        """Suggest the sharpest frame per window, snap each to the exact sharpest frame, then extract."""
        added = self.suggest_markers("sharpest", self.mdock.every.value(), quiet=True)
        if added:
            self.snap_markers(added, self.mdock.radius.value(), then_extract=True)
            self.osd(f"Auto-pick: {len(added)} frame(s) → snap → extract")

    def send_markers_to_batch(self):
        pending = sorted(f for f, m in self.markers.items() if not m.get("saved")) or sorted(self.markers)
        if not pending:
            self.osd("No markers")
            return
        self.open_batch_queue(add_current=False)
        self.batch_dialog.add_marker_job(self.video_path, [[f, self.markers[f].get("roi")] for f in pending])

    def review_markers(self):
        d = self.marker_out_dir()
        if os.path.isdir(d):
            self.open_gallery(d)
        else:
            self.osd("Nothing extracted yet")


    # ============================================================ timeline
    def on_samples(self, samples):
        if self.sender() is not self.timeline:
            return   # late batch from a stopped worker
        self.strip.add_samples(samples)
        if getattr(self, "_ttask", None) is not None and self.strip.grid:
            self._ttask.update(len(self.strip.samples) / max(1, len(self.strip.grid)))
        if self._hover and self.preview.isVisible():
            self.on_timeline_hover(*self._hover)

    def on_analysis_done(self):
        if self.sender() is self.timeline:
            self.strip.analyzing = False
            self.strip.update()
            if getattr(self, "_ttask", None) is not None:
                self._ttask.finish()
                self._ttask = None

    def on_timeline_hover(self, frame, gpt):
        if not self.engine:
            return
        self._hover = (frame, gpt)
        sample = self.strip.nearest_sample(frame, max_dist=self.strip.grid_spacing() * 0.75)
        if sample is None and self.timeline:
            self.timeline.request(frame)   # analyse this spot next
        sub = f"frame {frame}"
        desc = self.strip.describe(frame)
        if desc:
            sub += " · " + desc
        top = self.seekbar.mapToGlobal(self.seekbar.rect().topLeft()).y()
        self.preview.show_at(gpt.x(), top, sample.thumb if sample else None,
                             format_time(self.engine.time_of(frame)), sub)

    def hide_preview(self):
        self._hover = None
        if hasattr(self, "preview"):
            self.preview.hide()

    def strip_menu(self, pos):
        m = QMenu(self)
        f = self.strip.marker_at(pos.x(), tol=8)
        if f is not None:
            mk = self.markers.get(f, {})
            m.addSection(f"Marker #{f}" + (" (saved)" if mk.get("saved") else ""))
            m.addAction("Go to marker").triggered.connect(lambda: self.goto_frame(f))
            m.addAction(utils.newIcon("snap"), "Snap to sharpest").triggered.connect(
                lambda: self.snap_markers([f], self.mdock.radius.value()))
            ex = m.addAction(utils.newIcon("extract"), "Extract now")
            ex.setEnabled(not mk.get("saved"))
            ex.triggered.connect(lambda: self.extract_markers([f]))
            m.addAction(utils.newIcon("delete"), "Remove marker").triggered.connect(lambda: self.remove_markers([f]))
            m.exec(self.strip.mapToGlobal(pos))
            return
        thr = m.addAction("Set blur threshold from current frame")
        thr.setEnabled(self.current_sharpness is not None)
        thr.triggered.connect(lambda: self._set_blur_threshold(round(self.current_sharpness, 1)))
        clr = m.addAction("Clear blur threshold")
        clr.triggered.connect(lambda: self._set_blur_threshold(0))
        m.addSeparator()
        m.addAction(self.actions_.timeline_models)
        re = m.addAction("Re-analyze timeline")
        re.triggered.connect(self.start_timeline)
        m.exec(self.strip.mapToGlobal(pos))

    def _set_blur_threshold(self, v):
        self.settings.setValue("batch/min_sharpness", v)
        self.strip.set_blur_threshold(v)
        if self.batch_dialog:
            self.batch_dialog.sharp_sb.setValue(v)
        self.osd(f"Blur threshold {v:g}" if v else "Blur threshold off")

    def jump_flag(self, direction):
        flagged = self.strip.flagged_frames() or sorted(self.strip.cuts)
        if not flagged:
            self.osd("No flagged samples yet")
            return
        base = self.current_frame if self.is_playing else self.requested_frame
        cands = [f for f in flagged if (f > base if direction > 0 else f < base)]
        if cands:
            self.goto_frame(cands[0] if direction > 0 else cands[-1])

    def set_timeline_models(self, on):
        self.timeline_models = on
        self.settings.setValue("timeline/models", on)
        if on and not self.model_cfg["models"]:
            self.flash_status("No models configured: Settings → Models…")
        if self.engine:
            self.start_timeline()

    # ============================================================== models
    def open_models(self, tab=0):
        d = ModelsDialog(self, self.model_cfg,
                         frame_getter=lambda: self.current.bgr if self.current is not None and not self.current.preview else None,
                         show_detections=lambda per_model, cfg: self.show_detections(per_model, cfg))
        d.tabs.setCurrentIndex(tab)
        if d.exec():
            self.model_cfg = d.cfg_out()
            self.stop_marker_worker()
            save_model_cfg(self.model_cfg)
            self.stop_detector()
            if self.batch_dialog:
                self.batch_dialog.refresh_models_label()
            if self.engine and self.timeline_models:
                self.start_timeline()
            if self.actions_.detect.isChecked() and self.current is not None:
                self.request_detections(self.current)

    def stop_detector(self):
        if self.detector:
            self.detector.stop()
            self.detector = None
        self.detections_cache = {}

    def toggle_detections(self, on):
        if on and not self.model_cfg["models"]:
            self.actions_.detect.setChecked(False)
            self.open_models()
            if not self.model_cfg["models"]:
                return
            self.actions_.detect.setChecked(True)   # setChecked doesn't re-trigger; fall through
        if on and self.current is not None:
            self.request_detections(self.current)
        elif not on:
            self.canvas.set_detections([])
            self.pixel_label.clear()

    def request_detections(self, frame):
        if not self.model_cfg["models"] or (self.review is not None and self.review.active):
            return
        cached = self.detections_cache.get(frame.idx)
        if cached is not None:
            self.show_detections(cached)
            return
        if self.detector is None:
            self.detector = DetectWorker(dict(self.model_cfg))
            self.detector.result.connect(self.on_detections)
            self.detector.error.connect(self.on_detect_error)
            self.detector.start()
        self.detector.request(frame.idx, frame.bgr)

    def on_detections(self, idx, per_model):
        if self.sender() is not self.detector:
            return
        if len(self.detections_cache) > 500:
            self.detections_cache.clear()
        self.detections_cache[idx] = per_model
        if idx == self.current_frame and self.actions_.detect.isChecked() and not self.is_playing:
            self.show_detections(per_model)

    def show_detections(self, per_model, cfg=None):
        c = cfg or self.model_cfg
        boxes = []
        multi = len(per_model) > 1
        from AIO_QC.tools.frame_extractor.widgets.det_paint import ClassColors, label_text
        cc = ClassColors.shared()                    # a class has the same colour here, in the review and Image QC
        for m, dets in enumerate(per_model):
            mcol = QColor(model_hex(m))
            if dets and all(d.x1 == 0 and d.y1 == 0 and d.x2 == dets[0].x2 and d.y2 == dets[0].y2 for d in dets) \
                    and dets[0].x2 > 1:               # a classifier: the frame's best class, as a pill (no box)
                d = max(dets, key=lambda d: d.conf)
                badge = (str(m + 1), mcol, d.model) if multi else None
                boxes.append((d.x1, d.y1, d.x2, d.y2, label_text(d.name, d.conf), cc.get(d.name), False, badge,
                              float(d.conf), "class"))
                continue
            for d in dets:
                uncertain = c["unc_low"] <= d.conf < c["unc_high"]
                if d.conf < c["conf"] and not uncertain:
                    continue
                badge = (str(m + 1), mcol, d.model) if multi else None      # which model: its numbered badge
                boxes.append((d.x1, d.y1, d.x2, d.y2, label_text(d.name, d.conf), cc.get(d.name), uncertain, badge,
                              float(d.conf)))
        self.canvas.set_detections(boxes)
        n_unc = sum(1 for b in boxes if b[6])
        self.pixel_label.setText(f"{len(boxes)} detection(s)" + (f", {n_unc} uncertain (dashed)" if n_unc else ""))

    def on_detect_error(self, msg):
        self.actions_.detect.setChecked(False)
        self.stop_detector()
        QMessageBox.warning(self, "Models", f"Could not run the models:\n{msg}")

    # =============================================================== auto-annotate
    def open_autoannotate(self, source=""):
        if self.is_playing:
            self.pause()
        from AIO_QC.tools.frame_extractor.widgets.autoannotate_window import open_window
        return open_window(self, source)              # the last folders used (nothing is changed behind your back)

    # =============================================================== batch
    def open_batch_queue(self, add_current=True):
        if self.is_playing:
            self.pause()
        if self.batch_dialog is None:
            self.batch_dialog = BatchQueueDialog(
                self, lambda: self.output_dir,
                sharpness_getter=lambda: self.current_sharpness if self.current is not None else None,
                model_cfg_getter=lambda: self.model_cfg, open_models=self.open_models,
                open_gallery=self.open_gallery, output_cfg_getter=lambda: self.out_cfg,
                output_cfg_setter=self.set_output_cfg, open_output=self.open_output_settings)
            self.batch_dialog.sharp_sb.valueChanged.connect(self.strip.set_blur_threshold)
        self.batch_dialog.refresh_output_summary()
        self.batch_dialog.refresh_output_label()
        if add_current and self.engine and not self.batch_dialog.is_running():
            tf = lambda f: format_time(self.engine.time_of(f)) if f is not None else ""
            self.batch_dialog.add_job(self.video_path, tf(self.in_frame), tf(self.out_frame), self.roi)
        self.batch_dialog.show()
        self.batch_dialog.raise_()
        self.batch_dialog.activateWindow()

    # ================================================================ save
    def save_frame(self, full=False):
        if self.current is None or self.engine is None or not self.video_path:   # a frame left over from a closed video
            return
        if self.current.preview:
            self.flash_status("Showing a keyframe preview - release the seekbar to load the exact frame first")
            return
        if self.current.bgr is None:          # real-time playback frame: pause, save the exact frame
            self._save_after_exact = full
            self.pause()
            return
        from AIO_QC.tools.frame_extractor.core.naming import Namer
        from AIO_QC.tools.frame_extractor.core.imageio import imwrite
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        save_dir = os.path.join(self.output_dir, video_name)
        img, kind, roi = self.current.bgr, "frame", None
        if self.roi and not full:
            x1, y1, x2, y2 = self.roi
            img, kind, roi = img[y1:y2, x1:x2], "roi", self.roi
        c = self.out_cfg
        tpl = c["template"]
        if kind == "roi" and "{roi" not in tpl and "{kind" not in tpl:
            tpl += "_roi"                       # never overwrite the full frame with its crop
        namer = Namer(tpl, c["on_exists"], self.video_path, self.frame_rate, self.engine.time_of)
        save_path, rel = namer.path(save_dir, self.current.idx, c["ext"], img, kind, "", roi)
        if save_path is None:
            self.flash_status(f"{rel}{c['ext']} exists — skipped (Output ▸ If the file exists)")
            self.osd("Exists — skipped")
            return
        try:
            from AIO_QC.tools.frame_extractor.core.extract import frame_provenance
            imwrite(save_path, img, c["ext"], c["quality"],
                    provenance=frame_provenance(self.video_path, self.current.idx,
                                                self.engine.time_of(self.current.idx)))
            from AIO_QC.tools.frame_extractor.core import frame_sources
            frame_sources.record(save_path, self.video_path, self.current.idx, self.engine.time_of(self.current.idx))
        except Exception as e:
            logger.error(f"Save failed: {e}")
            self.flash_status(f"Save failed: {e}")
            return
        logger.info(f"Frame saved to {save_path}")
        self.flash_status(f"Frame saved to {save_path}")
        self.osd(f"Saved {os.path.basename(save_path)}")

    # ================================================================ misc
    def toggle_toolbar(self, checked):
        self.toolbar.setVisible(checked)
        self.update_status()

    def sync_toggle_toolbar(self, checked):
        self.toggleToolbar.setChecked(checked)

    # ====================================================== commands & keys
    def _registry(self):
        """(id, QAction, category) for every named command, from the menus."""
        out, seen = [], set()
        for title, menu in (("File", self.menus.file), ("Playback", self.menus.edit), ("Marks", self.menus.marks),
                            ("View", self.menus.view), ("Tools", self.menus.tools),
                            ("Settings", self.menus.settings), ("Help", self.menus.help)):
            stack = [(title, menu)]
            while stack:
                cat, m = stack.pop()
                for act in m.actions():
                    if act.menu():
                        stack.append((f"{cat} › {act.menu().title()}", act.menu()))
                    elif act.objectName() and act.objectName() not in seen and not act.isSeparator():
                        seen.add(act.objectName())
                        out.append((act.objectName(), act, cat))
        return out

    def _init_shortcuts(self):
        for k, act in vars(self.actions_).items():
            act.setObjectName(k)
        self.play_pause.setObjectName("play_pause")
        self.toggleToolbar.setObjectName("toggle_toolbar")
        self.dock.toggleViewAction().setObjectName("toggle_marks")
        self._default_shortcuts = {aid: act.shortcut().toString() for aid, act, _ in self._registry()}
        self.settings.beginGroup("shortcuts")
        custom = {k: self.settings.value(k, "") for k in self.settings.childKeys()}
        self.settings.endGroup()
        self._apply_shortcuts(custom)

    def _apply_shortcuts(self, mapping):
        for aid, act, _ in self._registry():
            if aid in mapping:
                act.setShortcut(QKeySequence(mapping[aid] or ""))

    def show_shortcuts(self):
        d = ShortcutsDialog(self, self._registry(), self._default_shortcuts)
        d.reference_requested.connect(lambda: self.open_help("shortcuts"))
        if d.exec():
            new = d.result_map()
            self._apply_shortcuts(new)
            self._refresh_help()
            self.settings.beginGroup("shortcuts")
            self.settings.remove("")
            for aid, seq in new.items():
                if seq != self._default_shortcuts.get(aid, ""):
                    self.settings.setValue(aid, seq)
            self.settings.endGroup()
            self.flash_status("Shortcuts saved")

    def open_palette(self):
        CommandPalette(self, self._registry()).exec()

    def refresh_theme(self):
        """AIO QC's palette changed (light <-> dark): repaint what draws its own colours."""
        self.strip.update()
        self._refresh_help()

    def _refresh_help(self):
        """The Help Center is shared by every mode and may have been closed and deleted: never keep a wrapper."""
        from AIO_QC.widgets.help_center_bridge import live_help
        w = live_help()
        if w is not None:
            w.refresh()

    # ============================================================ help
    def open_help(self, page=None):
        """F1: AIO QC's one Help Center (this mode's chapters, Testing, the other modes, all shortcuts)."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, page or "video")

    def _shortcut_entries(self):
        """[(category, command, [key sequences])] as configured right now (for the shortcut reference)."""
        out = []
        for _aid, act, cat in self._registry():
            name = act.text().replace("\n", " ").replace("&&", "\0").replace("&", "").replace("\0", "&").rstrip("…")
            out.append((cat, name, [k.toString() for k in act.shortcuts()]))
        return out

    # ------------------------------------------------------------------ first steps (3.0)
    def open_sample_video(self):
        from AIO_QC.core.sample_data import sample_video
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = sample_video()
        finally:
            QApplication.restoreOverrideCursor()
        if path is None:
            QMessageBox.information(self, __appname__, "The sample video needs PyAV (pip install av).")
            return
        self.load_video(str(path))

    def toggle_keymap(self):
        from AIO_QC.widgets.keymap import KeyMapOverlay
        if getattr(self, "_keymap", None) is None:
            self._keymap = KeyMapOverlay(self)
        self._keymap.toggle()

    def _disabled_reason(self, act) -> str:
        if self.video_path is None:
            return "load a video first (Ctrl+O) — or try the sample video (Help ▸ Open a Sample Video)."
        return ""

    def tour_steps(self):
        from AIO_QC.widgets.coach import Step
        a = self.actions_
        btn = lambda act: self.toolbar.widgetForAction(act)
        return [
            Step(lambda: btn(a.load), "Open a video", "Any common format (Ctrl+O) — or drop it on the window."),
            Step(lambda: self.canvas, "Play and step",
                 "<b>Space</b> plays and pauses, <b>← →</b> step one frame, the wheel scrubs. Drag on the picture "
                 "to draw a region (ROI) if you only want part of it."),
            Step(lambda: self.seekbar, "The timeline",
                 "Click or drag to jump. After loading, the strip under it shows scene changes, blur and motion, so "
                 "you can see where the useful moments are."),
            Step(lambda: btn(a.save), "Save a frame", "<b>Ctrl+S</b> saves the frame you see into the output "
                 "folder (shown at the bottom left)."),
            Step(lambda: self.mdock, "Mark many, save once",
                 "<b>M</b> drops a marker at the playhead — even while playing. <b>Extract pending</b> saves them "
                 "all; <b>Suggest</b> picks the sharpest frames for you."),
            Step(lambda: btn(a.autoannotate), "A whole folder of videos",
                 "<b>Auto-Annotate</b> turns many videos into unique frames with annotations from your model or an "
                 "AI — every step optional. <b>Batch Extract</b> just extracts."),
            Step(None, "Find anything", "<b>Ctrl+K</b> finds any command by name and <b>F1</b> opens the guide."),
        ]

    def start_tour(self, force: bool = False):
        from AIO_QC.widgets import coach
        if self.video_path is None:                  # nothing to point at yet: show it on the sample video
            coach.mark_done("video")                 # (loading it must not start a second tour)
            self.open_sample_video()
            tries = [0]

            def when_loaded():                       # the video opens in the background
                if self.video_path is not None:
                    coach.run(self, self.tour_steps(), "video", force=True)
                elif tries[0] < 50:
                    tries[0] += 1
                    later(100, self, when_loaded)
            when_loaded()
            return None
        return coach.run(self, self.tour_steps(), "video", force=force)

    def _build_start_page(self):
        from AIO_QC.widgets import start_page as SP
        shell = getattr(QApplication.instance(), "shell", None)
        return SP.StartPage(
            "Get frames from your videos",
            "Play a video frame by frame, mark the moments you need and save them as images — or turn a whole "
            "folder of videos into unique, annotated frames.",
            actions=[SP.Action("Open a video…", lambda: self.load_video(), primary=True, icon="open", hint="Ctrl+O"),
                     SP.Action("A folder of videos → frames…", lambda: self.open_autoannotate(),
                               icon="tool-autoannotate", tip="Auto-Annotate: unique frames, optionally annotated")],
            links=[SP.Action("Try a sample video", self.open_sample_video),
                   SP.Action("Quick tour", lambda: self.start_tour(force=True))],
            recents=lambda: SP.all_recents(shell, first="video"),
            drop_hint="…or drop a video anywhere on this window.", parent=self)

    def _sync_start_page(self):
        page = getattr(self, "start_page", None)
        if self.video_path is None:
            if page is None:
                page = self.start_page = self._build_start_page()
                self.central_stack.addWidget(page)
            page.refresh()
            self.central_stack.setCurrentWidget(page)
        elif page is not None:
            self.central_stack.setCurrentIndex(0)

    def _ffmpeg(self):
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
        return find_ffmpeg(self.settings.value("playback/ffmpeg", ""))

    def open_output_in_qc(self, folder=None):
        """Frame Extractor -> Image QC: this video's output folder (markers, batch extraction, saved frames)."""
        if folder is None and self.video_path:
            cand = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video_path))[0])
            folder = cand if os.path.isdir(cand) else None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Folder with extracted frames", self.output_dir)
            if not folder:
                return
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, __appname__, "Image QC is available when AIO QC is started normally.")
            return
        shell.open_in_qc(folder)

    def _testing(self):
        """Image QC's Testing window if it exists (its models and colours are reused for the annotated export)."""
        shell = getattr(QApplication.instance(), "shell", None)
        img = getattr(shell, "image", None) if shell is not None else None
        return getattr(img, "testing", None) if img is not None else None

    def open_at(self, path, frame=None, seconds=None):
        """Show a video at a frame (or time) - loads it first when it is not the current one."""
        if not path:
            return
        if self.video_path and self.engine and os.path.normcase(os.path.abspath(path)) == \
                os.path.normcase(os.path.abspath(self.video_path)):
            f = frame if frame is not None else self.engine.index_of(seconds or 0.0)
            self.goto_frame(max(0, min(self.frame_count - 1, int(f))))
        elif os.path.isfile(path):
            self._goto_after_open = (path, frame, seconds)
            self.load_video(path)
        else:
            self.flash_status(f"Not found: {path}", 4000)
            return
        self.raise_()
        self.activateWindow()

    def _search_context(self):
        fr = self.current if self.current is not None and not self.current.preview else None
        return dict(video=self.video_path, frame=fr.bgr if fr is not None else None,
                    time=self.engine.time_of(fr.idx) if (fr is not None and self.engine) else None,
                    roi=list(self.roi) if self.roi else None, env_python=self.model_cfg.get("env_python", ""),
                    ffmpeg=self._ffmpeg())

    def _marker_at(self, video, seconds):
        if not (self.video_path and self.engine and os.path.abspath(video) == os.path.abspath(self.video_path)):
            self.open_at(video, seconds=seconds)
            self.flash_status("Opened the video - press M to drop the marker", 5000)
            return
        f = max(0, min(self.frame_count - 1, int(self.engine.index_of(seconds))))
        if f not in self.markers:
            self.add_marker(f)
        self.goto_frame(f)
        self.osd(f"Marker #{f}")

    def export_clip(self):
        if not self.engine:
            return
        ff = self._ffmpeg()
        if not ff:
            QMessageBox.warning(self, __appname__, "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import ClipExportDialog
        dur = self.engine.duration()
        a = self.engine.time_of(self.in_frame) if self.in_frame is not None else self.engine.time_of(self.current_frame)
        b = self.engine.time_of(self.out_frame) if self.out_frame is not None else dur
        if self.in_frame is None and self.out_frame is None:
            self.flash_status("Tip: set in / out points (I / O) to choose the range", 4000)
        if self.is_playing:
            self.pause()
        ClipExportDialog(self, self.video_path, a, b, dur, ff).exec()

    def export_annotated(self):
        if not self.engine:
            return
        ff = self._ffmpeg()
        if not ff:
            QMessageBox.warning(self, __appname__, "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.core.testing import pipeline as P
        specs, slots = [], None
        lab = self._testing()
        if lab is not None and lab.specs:
            specs, slots = list(lab.specs), list(lab.slots)
        else:
            specs = P.unique_labels([P.spec_from_cfg(m, self.model_cfg) for m in self.model_cfg.get("models", [])
                                     if os.path.isfile(m)])
        if not specs:
            QMessageBox.information(self, __appname__, "Add a model first (Settings → Models, or Testing).")
            return
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import AnnotatedExportDialog
        dur = self.engine.duration()
        a = self.engine.time_of(self.in_frame) if self.in_frame is not None else 0.0
        b = self.engine.time_of(self.out_frame) if self.out_frame is not None else dur
        if self.is_playing:
            self.pause()
        AnnotatedExportDialog(self, self.video_path, specs, ff, a, b, dur, conf=float(self.model_cfg.get("conf", 0.25)),
                              slots=slots).exec()

    def reset_layout(self):
        self.restoreState(self._default_state)
        self.toolbar.show()
        self.dock.show()

    # ============================================================ gallery
    def open_gallery(self, folder=None):
        if folder is None:
            cand = []
            if self.video_path:
                name = os.path.splitext(os.path.basename(self.video_path))[0]
                cand = [os.path.join(self.output_dir, name, "batch_extract"), os.path.join(self.output_dir, name)]
            folder = next((c for c in cand if os.path.isdir(c)), None)
            if folder is None:
                folder = QFileDialog.getExistingDirectory(self, "Folder with extracted frames", self.output_dir)
                if not folder:
                    return
        if self.gallery is None:
            self.gallery = ReviewGallery(self, folder)      # belongs to the app: hides and returns with it
            self.gallery.setWindowIcon(utils.newIcon("gallery"))
        else:
            self.gallery.set_folder(folder)
        self.gallery.show()
        self.gallery.raise_()
        self.gallery.activateWindow()

    # =========================================================== tracking
    def toggle_tracking(self, on):
        if not on:
            self.stop_tracking()
            self.osd("ROI tracking off")
            return
        if not self.roi or self.current is None or self.current.preview:
            self.actions_.track.setChecked(False)
            self.osd("Draw an ROI on the object first (R)")
            return
        self.tracker = TrackWorker()
        self.tracker.result.connect(self.on_track_result)
        self.tracker.error.connect(self.on_track_error)
        self.tracker.start()
        img, sc = self.frame_image()
        self.tracker.init(self.current.idx, img, self.roi, sc)
        self._track_last = self.current.idx
        self.osd("ROI tracking on: play or step forward")

    def track_frame(self, frame):
        last = self._track_last
        img, sc = self.frame_image(frame)
        if img is None:
            return
        if last is not None and 0 < frame.idx - last <= 60:
            self.tracker.update(frame.idx, img, sc)
        elif frame.idx != last and self.roi:
            self.tracker.init(frame.idx, img, self.roi, sc)   # jumped: restart from here
        self._track_last = frame.idx

    def on_track_result(self, idx, roi):
        if self.sender() is not self.tracker:
            return
        if roi is None:
            self.actions_.track.setChecked(False)
            self.stop_tracking()
            self.osd("Tracking lost - redraw the ROI and press T")
            return
        self.roi = roi
        self.canvas.set_roi(roi)
        self.update_roi_label()

    def on_track_error(self, msg):
        self.actions_.track.setChecked(False)
        self.stop_tracking(save=False)
        QMessageBox.warning(self, "ROI tracking", f"Tracker unavailable:\n{msg}\n\n"
                            "Install opencv-contrib-python for the CSRT/KCF trackers.")

    def stop_tracking(self, save=True):
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
            if save:
                self.save_session()
        self._track_last = None

    # ======================================================= keyframe index
    def on_keyframe_index(self, index):
        if self.sender() is self.timeline and self.engine:
            self.engine.set_keyframe_index(index)
            if self.engine.fast_scrub:
                logger.info(f"Fast keyframe scrubbing on ({len(index.keyframes)} keyframes)")

    def on_grid_changed(self, grid, unit):
        if self.sender() is self.timeline:
            self.strip.set_grid(grid, unit)


    def show_about(self):
        from AIO_QC.widgets.about import show_about
        from AIO_QC.utils import newIcon as aio_icon
        show_about(self, __appname__, "Video frame extraction for image annotation", aio_icon("mode-video"))

    def close_window(self):
        self.close()
