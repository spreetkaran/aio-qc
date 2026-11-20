# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC main window — QC tools: tool windows, external editors, image integrity, classification, findings.

A mixin of AIO_QC.app.MainWindow: the methods use the window's own attributes (self.canvas, self.session …)
set up in MainWindow.__init__.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QInputDialog, QLabel, QMenu, QMessageBox,
)

from AIO_QC import utils
from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
from AIO_QC.widgets import anim
from AIO_QC.widgets.image_quality_dialog import IntegritySweep, image_of, integrity_findings
from AIO_QC.core.image_quality import TITLES as IQ_TITLES
from AIO_QC.widgets.lifecycle import stop_threads
from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog

from AIO_QC.app_common import (QC_TOOLS)
from AIO_QC.core.diag import ignored as _ignored


class ToolsMixin:

    # =====================================================================
    # QC tools
    # =====================================================================
    def open_tool(self, key: str):
        if key == "testing":                     # works without a dataset too (test sets, videos only)
            self.open_testing()
            return
        if key == "data_prep":                   # Preparation: folders, no dataset needed
            self.open_data_prep()
            return
        if key in ("convert_yolo", "split"):     # Convert & Split on YOLO (its Output step) / on its Split step
            self.open_tool("export")
            dlg = self.open_dialogs.get("export")
            if dlg is not None:
                dlg.show_section(key)
            return
        if not self.validate_inputs():
            return
        if key == "crop_gallery" and self.session.is_classification:        # whole images, not crops
            key = "class_gallery"
        tool = next(t for t in QC_TOOLS if t.key == key)
        if key in ("fix_editor", "sampling"):            # no scan to finish: opening it counts
            self.check_guide.record(key)
        if key == "fix_editor":
            self.open_editor_session()
            return
        if tool.dialog is None:
            getattr(self.a, tool.action).trigger()
            return
        from AIO_QC.app_common import tool_for
        tool = tool_for(tool, self.session)          # classification: the classifier version of a box check
        if tool.dialog.SUPPORTED is not None and self.session.type not in tool.dialog.SUPPORTED:
            QMessageBox.information(self, tool.title, f"{tool.title} is not available for {self.session.type.title}.")
            return
        if key == "remove_class" and not self.session.index_complete:
            QMessageBox.information(self, tool.title, "The label index is still being built (see status bar). "
                                    "Class removal needs complete counts — try again in a moment.")
            return
        if key == "ambiguity" and self.session.is_yolo and not self.session.true_classes:
            self.load_classes_btn.flash("warning", "Load classes.txt first")
            reply = QMessageBox.question(self, tool.title,
                                         "No dataset.yaml or classes.txt found for this YOLO dataset.\n"
                                         "Load classes.txt first to check label ambiguity?")
            if reply != QMessageBox.StandardButton.Yes or not self.load_classes_file():
                return
        dlg = self.open_dialogs.get(key)
        if dlg is None and key == "remove_class" and self.session.is_classification:
            from AIO_QC.widgets.cls_classes_dialog import ClsClassesDialog
            dlg = ClsClassesDialog(self)               # classes of a classification dataset: rename / merge / remove
            dlg.tool_key = key
            dlg.setWindowIcon(utils.newIcon(tool.icon))
            self.open_dialogs[key] = dlg
        if dlg is None:
            dlg = tool.dialog(self)
            dlg.tool_key = key
            dlg.setWindowIcon(utils.newIcon(tool.icon))
            self.open_dialogs[key] = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------ external editor sessions
    def _end_editor_session(self, why: str) -> bool:
        """An editor session must end (or be kept for later) before the dataset goes away."""
        dlg = self.editor_dialog
        if dlg is None or dlg.es is None or not dlg.es.active:
            return True
        if dlg.mini is not None and dlg.mini.isVisible():
            dlg.from_mini()
        else:
            dlg.show()
            dlg.raise_()
        QMessageBox.information(self, "Fix in Editor", f"Finish the editor session before you {why} "
                                                       "(or choose “Keep for later” when closing its window).")
        return False

    def open_editor_session(self, keys: list[str] | None = None):
        if not self.validate_inputs():
            return
        if self.editor_dialog is not None and self.editor_dialog.es is not None and self.editor_dialog.es.active:
            if self.editor_dialog.mini is not None and self.editor_dialog.mini.isVisible():
                self.editor_dialog.from_mini()
            else:
                self.editor_dialog.show()
                self.editor_dialog.raise_()
            return
        old = self.open_dialogs.pop("fix_editor", None)
        if old is not None:
            old.close()
        dlg = EditorSessionDialog(self, keys=keys)
        dlg.tool_key = "fix_editor"
        self.open_dialogs["fix_editor"] = dlg
        dlg.show()

    def _offer_unfinished_editor_sessions(self):
        from AIO_QC.core.edit_session import EditSession
        s = self.session
        if s is None:
            return
        for work in EditSession.unfinished(s.root):
            try:
                es = EditSession.load(work, s.root)
            except Exception as exc:
                logger.warning(f"unreadable editor session {work}: {exc}")
                continue
            c = es.counts()
            box = QMessageBox(self)
            box.setWindowTitle("Unfinished editor session")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(f"An editor session from {es.created} ({es.source_title}, {len(es.entries)} files) was not "
                        f"finished.\n{c.get('fixed', 0)} fixed · {c.get('waiting', 0)} waiting.\n\n"
                        "Resume it (edits saved in the editor since then are synced now)?")
            resume = box.addButton("Resume", QMessageBox.ButtonRole.AcceptRole)
            finish = box.addButton("Sync and finish", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() in (resume, finish):
                dlg = EditorSessionDialog(self, resume=work)
                dlg.tool_key = "fix_editor"
                self.open_dialogs["fix_editor"] = dlg
                dlg.show()
                if box.clickedButton() is finish:
                    dlg._finish()
            return                                  # one at a time

    def open_batch_shots(self, keys=None):
        """Review ▸ Flagged Screenshots, or File list ▸ Save screenshots of N files (those files preselected)."""
        self.open_tool("screenshots")
        dlg = self.open_dialogs.get("screenshots")
        if dlg is not None:
            dlg.set_keys(keys)
        return dlg

    def copy_file_names(self, keys, full: bool = False) -> str:
        """File list ▸ Copy file name(s) / path(s): one per line, in list order. The name is the file shown in the
        list (the JSON for LabelMe, the image for YOLO / classification)."""
        s = self.session
        if s is None:
            return ""
        out = []
        for k in keys:
            smp = s.by_key.get(k)
            if smp is None:
                continue
            path = Path(k)
            out.append(str(path) if full else smp.name if getattr(smp, "name", None) else path.name)
        text = "\n".join(out)
        QApplication.clipboard().setText(text)
        what = ("path" if full else "name") + ("s" if len(out) > 1 else "")
        self.statusBar().showMessage(f"Copied {len(out):,} file {what}" if len(out) > 1 else f"Copied: {text}", 4000)
        return text

    def _file_list_menu(self, pos):
        """Right-click on the file list: open / review marks / fix in editor for the selected files."""
        view = self.file_dock.view
        idx = view.indexAt(pos)
        if idx.isValid() and not view.selectionModel().isSelected(idx):
            view.setCurrentIndex(idx)
        keys = self.file_dock.selected_keys()
        if not keys or self.session is None:
            return
        n = f"{len(keys)} file(s)" if len(keys) > 1 else "file"
        m = QMenu(self)
        m.addAction(utils.newIcon("eye"), "Open", lambda: self.open_sample(keys[0]))
        names = "names" if len(keys) > 1 else "name"
        m.addAction(utils.newIcon("copy"), f"Copy file {names}", lambda: self.copy_file_names(keys))
        m.addAction(f"Copy file path{'s' if len(keys) > 1 else ''}", lambda: self.copy_file_names(keys, full=True))
        m.addSeparator()
        m.addAction(utils.newIcon("review-flag"), f"Flag {n}", lambda: self.session.set_review(keys, REVIEW_FLAG))
        m.addAction(utils.newIcon("review-ok"), f"Mark {n} OK", lambda: self.session.set_review(keys, REVIEW_OK))
        m.addAction(utils.newIcon("review-clear"), "Clear review mark", lambda: self.session.set_review(keys, None))
        m.addSeparator()
        m.addAction(utils.newIcon("tool-editor"), f"Fix {n} in editor…", lambda: self.open_editor_session(keys))
        m.addAction(utils.newIcon("screenshot"), f"Save screenshots of {n}…", lambda: self.open_batch_shots(keys))
        if len(keys) == 1:
            m.addSeparator()
            m.addAction(utils.newIcon("mode-video"), "Show source video in Frame Extractor",
                        lambda: self.show_source_video(keys[0]))
        m.exec(view.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ image integrity (background + viewer)
    def _start_sweep(self):
        self._stop_sweep()
        s = self.session
        if s is None or not s.samples:
            return
        self.broken = {}
        sw = IntegritySweep(s.samples, s.adapter, s.cache, self)
        sw.batch.connect(lambda found, checked, sw=sw: self._on_sweep_batch(found, checked, sw))
        sw.finished.connect(lambda sw=sw: self._on_sweep_done(sw))
        sw.finished.connect(sw.deleteLater)                  # one per opened dataset: freed once it has ended
        self.sweep = sw
        sw.start()

    def _stop_sweep(self):
        sw = self.sweep
        self.sweep = None
        if sw is not None:
            try:
                sw.batch.disconnect()
                sw.stop()
            except (TypeError, RuntimeError):                # nothing connected / already ended and freed
                pass

    def _on_sweep_batch(self, found, checked, sw):
        from AIO_QC.core.issues import Issue
        if sw is not self.sweep or self.session is None:          # a previous dataset's sweep
            return
        issues = {k: [Issue("integrity", sev, f"{IQ_TITLES.get(c, c)}: {m}") for c, sev, m in f] for k, f in found.items()}
        self.session.issues.publish("integrity", "Image integrity (automatic)", set(checked), issues)
        self.session.issuesChanged.emit()
        new = {k: [x for x in f if x[1] == "error"] for k, f in found.items()}
        new = {k: v for k, v in new.items() if v}
        if new:
            self.broken.update(new)
            self.file_dock.model.set_flags({k: "corrupt" for k in new})
            self.notify_broken_images(sorted(self.broken), source="the background image check")
            if self.current_key in new:
                self._check_current_image()

    def _on_sweep_done(self, sw):
        if sw is self.sweep:
            logger.info(f"[image check] {len(self.session.samples) if self.session else 0:,} images checked, "
                        f"{len(self.broken):,} broken")
            self.check_guide.refresh()

    def notify_broken_images(self, keys, source: str = "Image Quality"):
        """One amber notification (updated in place) with the broken images; 'Show' filters the file list."""
        keys = sorted(set(keys))
        for k in keys:
            self.broken.setdefault(k, [("corrupt", "error", "see Image Quality")])
        self.file_dock.model.set_flags({k: "corrupt" for k in keys})
        allk = sorted(self.broken)
        self.toast.show_message(
            f"<b>⚠ {len(allk):,} broken image(s)</b><br>Corrupt, truncated or inconsistent files found by {source}. "
            "Training tools may crash or silently skip them.",
            kind="warn", key="broken-images",
            actions=[("Show", lambda: self.file_dock.set_key_filter(sorted(self.broken), "Broken images")),
                     ("Image Quality…", lambda: self.open_tool("image_quality"))],
            details=self._broken_text)

    def _broken_text(self) -> str:
        by_key = self.session.by_key if self.session else {}
        lines = []
        for k in sorted(self.broken):
            why = "; ".join(str(f[-1]) for f in (self.broken[k] or [])[:3])
            lines.append(f"{by_key[k].display if k in by_key else k}  —  {why}")
        return "\n".join(lines) or "No broken images."

    def _check_current_image(self):
        """Red banner over the viewer when the open image is broken (cached check, ~2 ms otherwise)."""
        banner = getattr(self, "_banner", None)
        if banner is None:
            banner = self._banner = QLabel(self.canvas)
            banner.setWordWrap(True)
            banner.setStyleSheet("QLabel{background: rgba(198,40,40,215); color: white; border-radius: 8px;"
                                 "padding: 8px 12px; font-weight: 600;}")
            banner.hide()
        chip = getattr(self, "_class_chip", None)
        if chip is None:
            chip = self._class_chip = QLabel(self.canvas)
            chip.setStyleSheet("QLabel{background: rgba(21,101,192,220); color: white; border-radius: 12px;"
                               "padding: 4px 14px; font-weight: 700; font-size: 13pt;}")
            chip.hide()
        s = self.session
        if s is None or self.current_key not in (s.by_key if s else {}):
            banner.hide()
            chip.hide()
            return
        if s.is_classification and self.current_ann is not None:
            tags = self.current_ann.tags
            chip.setText("  ·  ".join(tags) if tags else "no class")
            chip.adjustSize()
            chip.show()
            self._place_class_chip()
        else:
            chip.hide()
        found = self.broken.get(self.current_key)
        if found is None:
            try:
                path, w, h = image_of(s.adapter, s.by_key[self.current_key])
                found = [f for f in integrity_findings(s.cache, path, w, h) if f[1] == "error"] if path else []
            except Exception as _exc:
                _ignored(_exc)
                found = []
            if found:
                self.broken[self.current_key] = found
                self.file_dock.model.set_flags({self.current_key: "corrupt"})
        if found:
            banner.setText("⚠ " + "\n⚠ ".join(m for _c, _s, m in found[:3]))
            banner.setFixedWidth(max(240, self.canvas.width() - 28))
            banner.adjustSize()
            top = 14 + (self.hint_banner.occupied() if hasattr(self, "hint_banner") else 0)
            banner.move(14, top + (chip.height() + 8 if chip.isVisible() else 0))
            banner.show()
            banner.raise_()
        else:
            banner.hide()

    def _place_class_chip(self):
        """The class chip of a classifier image: top-centre of the image area, below a hint card — the corners are
        the file read-out's (left) and the flag's (right)."""
        chip = getattr(self, "_class_chip", None)
        if chip is None or not chip.isVisible():
            return
        g = self.canvas.viewport().geometry()
        top = g.top() + 14 + (self.hint_banner.occupied() if hasattr(self, "hint_banner") else 0)
        chip.move(max(g.left() + 14, g.left() + (g.width() - chip.width()) // 2), top)
        chip.raise_()

    # ------------------------------------------------------------------ classification
    # ------------------------------------------------------------------ issues folder (class folders)
    def issues_folder(self):
        """This dataset's issues folder (Settings ▸ Classification: inside the dataset, next to it, or a folder of
        your choice)."""
        from AIO_QC.core import issues_folder as IF
        st = self.settings
        place = st.value("cls/issues_place", "dataset", str) or "dataset"
        custom = st.value("cls/issues_dir", "", str) or ""
        if place == "custom" and not custom.strip():
            place = "dataset"
        return IF.location(self.session.root, place, custom)

    def _issues_ok(self) -> bool:
        s = self.session
        if s is None or s.type.value != "cls_folders":
            self.statusBar().showMessage("The issues folder is for class-folder datasets", 5000)
            return False
        return True

    def send_to_issues(self, note=None, keys=None):
        """I: the selected image(s) — or the one shown — to issues/<class>/ (Shift+I: with a note). One step of
        Undo; the next image opens."""
        if not self._issues_ok():
            return []
        s = self.session
        if keys is None:
            keys = self.file_dock.selected_keys() or ([self.current_key] if self.current_key else [])
        keys = [k for k in keys if k in s.by_key]
        if not keys:
            return []
        if note is True:
            text, ok = QInputDialog.getText(self, "Send to Issues", f"Note for {len(keys):,} image(s) (kept in the "
                                            "issues log):")
            if not ok:
                return []
            note = text.strip()
        folder = self.issues_folder()
        with s.as_tool("Send to issues"):
            done, errors = s.send_to_issues(keys, folder, note or "")
        if errors:
            self.log_errors([f"[issues] {e}" for e in errors])
        if done:
            classes = sorted({it["class"] or "no class" for it in s._last_issues})
            where = f"{folder.name}/{classes[0]}" if len(classes) == 1 else f"{folder.name} ({len(classes)} classes)"
            self.statusBar().showMessage(f"{len(done):,} image(s) → {where} · Ctrl+Z puts them back"
                                         + (f" · {len(errors)} failed" if errors else ""), 6000)
            self.toast.show_message(
                f"<b>Set aside {len(done):,} image{'s' if len(done) != 1 else ''}</b> in {where}<br>Out of the "
                "dataset until you put them back — Ctrl+Z undoes.", "info",
                actions=[("Undo", self.inline.undo), ("Show", self.open_issues_folder)], timeout_ms=6000,
                key="issues")
        return done

    def open_issues_folder(self):
        if not self._issues_ok():
            return
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        f = self.issues_folder()
        if not f.is_dir():
            self.statusBar().showMessage(f"Nothing set aside yet ({f})", 5000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(f)))

    def _issues_count(self) -> int:
        from AIO_QC.core import issues_folder as IF
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        return len(IF.contents(self.issues_folder(), IMAGE_EXTENSIONS))

    def review_issues(self):
        """The issues folder as a class-folder dataset (its own window state): fix classes in the Class Gallery;
        File ▸ Open Recent brings the dataset back, then Put Issues Back."""
        if not self._issues_ok():
            return
        f = self.issues_folder()
        n = self._issues_count() if f.is_dir() else 0
        if not n:
            QMessageBox.information(self, "Review Issues", "Nothing is set aside in the issues folder.")
            return
        if QMessageBox.question(self, "Review Issues", f"Open the issues folder ({n:,} image(s)) as a dataset?\n\n"
                                f"{f}\n\nFix their classes there (Class Gallery), then reopen this dataset and use "
                                "Put Issues Back — each image returns to the class folder it is in then.") != \
                QMessageBox.StandardButton.Yes:
            return
        from AIO_QC.core.dataset import DatasetType
        self.open_dataset(f, DatasetType.CLS_FOLDERS)

    def return_issues(self):
        """Every image in the issues folder back into the dataset (into the class folder it is in now)."""
        if not self._issues_ok():
            return
        f = self.issues_folder()
        n = self._issues_count() if f.is_dir() else 0
        if not n:
            QMessageBox.information(self, "Put Issues Back", "Nothing is set aside in the issues folder.")
            return
        if QMessageBox.question(self, "Put Issues Back", f"Move {n:,} image(s) from\n{f}\nback into the dataset — "
                                "each into the class folder it is in now?") != QMessageBox.StandardButton.Yes:
            return
        s = self.session
        with s.as_tool("Put issues back"):
            keys, errors, skipped = s.return_from_issues(f)
        if errors:
            self.log_errors([f"[issues] {e}" for e in errors])
        self.statusBar().showMessage(f"{len(keys):,} image(s) back in the dataset" +
                                     (f" · {skipped} not in a class folder there (left)" if skipped else "") +
                                     (f" · {len(errors)} failed" if errors else ""), 8000)
        if keys:
            self.file_dock.select_key(keys[0])

    def change_class(self):
        """C: the class of the selected images (or the one shown) — one class, or add / remove (multi-label)."""
        from AIO_QC.widgets.class_picker import ClassPicker
        s = self.session
        if s is None or not s.is_classification:
            return
        keys = self.file_dock.selected_keys() or ([self.current_key] if self.current_key else [])
        keys = [k for k in keys if k in s.by_key]
        if not keys:
            return
        cur = Counter(t for k in keys for t in s.tags_of(k))
        multi = s.cls_mode == "multi"
        dlg = ClassPicker(self, s.cls_classes(), cur, len(keys), multi=multi, counts=s.label_totals())
        if not dlg.exec() or dlg.result is None:
            return
        if dlg.result[0] == "set":
            name = dlg.result[1][0].strip()
            if not name or not s.base_adapter.can_use_label(name):
                self.statusBar().showMessage(f"“{name}” cannot be a class name", 5000)
                return
            if s.type.value == "cls_folders" and name not in s.cls_classes() and QMessageBox.question(
                    self, "Change class", f"Create the new class folder “{name}”?") != QMessageBox.StandardButton.Yes:
                return
            self._apply_classes(keys, tags=[name], what=f"class → {name}")
        else:
            _k, add, remove = dlg.result
            if add or remove:
                self._apply_classes(keys, add=add, remove=remove,
                                    what=" ".join([f"+{a}" for a in add] + [f"−{r}" for r in remove]))

    def _apply_classes(self, keys, tags=None, add=(), remove=(), what=""):
        s = self.session
        was_current = self.current_key in keys
        done, errors = s.set_tags(keys, tags=tags, add=add, remove=remove, what=what)
        if errors:
            self.log_errors([f"[change class] {e}" for e in errors])
        if done and was_current:
            self.file_dock.select_key(done[0] if len(done) == 1 or self.current_key not in s.by_key
                                      else self.current_key)
        if done and self.current_key in s.by_key:
            self.reload_current_annotation()
        self.statusBar().showMessage(f"{len(done)} image(s): {what}" + (f" · {len(errors)} failed" if errors else "")
                                     + (" · Ctrl+Z undoes it" if done else ""), 6000)

    def class_key(self, n: int):
        """1 … 9 in a classification dataset: that class (in the Classes order) for the selected images — toggled
        in a multi-label dataset."""
        s = self.session
        names = s.cls_classes()
        if n - 1 >= len(names):
            self.statusBar().showMessage(f"There is no class {n} (the dataset has {len(names)})", 4000)
            return
        keys = self.file_dock.selected_keys() or ([self.current_key] if self.current_key else [])
        keys = [k for k in keys if k in s.by_key]
        if not keys:
            return
        name = names[n - 1]
        if s.cls_mode == "multi":
            have = all(name in s.tags_of(k) for k in keys)
            self._apply_classes(keys, add=() if have else [name], remove=[name] if have else (),
                                what=("−" if have else "+") + name)
        else:
            self._apply_classes(keys, tags=[name], what=f"class → {name}")

    def forget_dialog(self, dlg):
        """A tool window closed: stop its threads and DELETE it. Without the delete a closed window stayed alive as
        a hidden child of the main window — with its result rows, thumbnails and image caches — and opening the
        tool again made another one (memory grew with every open / close)."""
        stop_threads(dlg, f"{getattr(dlg, 'TITLE', type(dlg).__name__)} closing")      # before Qt destroys it
        key = getattr(dlg, "tool_key", None)
        if key and self.open_dialogs.get(key) is dlg:
            del self.open_dialogs[key]
        if not getattr(dlg, "keep_after_close", False):
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)   # Qt deletes it once the close completes

    def _ask_about_training(self) -> bool:
        """Closing with a training run going: it keeps running on its own (default) — or stop it. False: cancel."""
        w = getattr(self, "train_watch", None)
        runs = w.running() if w is not None else []
        if not runs or getattr(self, "_quit_reason", None) == "test":
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Training is running")
        box.setText(f"{len(runs)} training run(s) going: {', '.join(r.name for r in runs[:3])}.\n\nThey keep going on "
                    "their own after AIO QC closes and show up again in Train Model when it reopens.")
        keep = box.addButton("Keep training", QMessageBox.ButtonRole.AcceptRole)
        stop = box.addButton("Stop them (keep results)", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is stop:
            from AIO_QC.core import training as TR
            for r in runs:
                TR.control(r, "stop")
            return True
        return box.clickedButton() is keep

    def validate_inputs(self) -> bool:
        if self.session is None:
            QMessageBox.warning(self, "No dataset", "Open a dataset directory first (Ctrl+O).")
            self.flash_button(self.open_dir_btn, "warning", "Open a dataset first")
            return False
        return True

    def flash_button(self, status_btn, state: str = "warning", detail: str = ""):
        status_btn.flash(state, detail)

    def open_sample(self, key: str, shape_index: int | None = None, overlay: list | None = None,
                    focus_box=None):
        """
        Called by QC dialogs: jump to a file (and shape), even if it is currently filtered out.
        overlay: temporary dashed boxes (e.g. model predictions); focus_box: glide to a pixel box.
        """
        if not self.session or key not in self.session.by_key:
            self.statusBar().showMessage(f"Not a viewable sample: {key}", 6000)
            return
        self._note_finding_source(key, shape_index)
        if key == self.current_key:
            self.reload_current_annotation()
            self._apply_extras(shape_index, overlay, focus_box)
            return
        self._pending_extras = (overlay, focus_box)
        self._pending_shape = shape_index
        if not self.file_dock.select_key(key):
            self.file_dock.clear_all_filters()
            self.file_dock.select_key(key)

    # ------------------------------------------------------------------ F8 / Shift+F8: findings from the image
    def _finding_windows(self):
        out = []
        for dlg in list(self.open_dialogs.values()):
            try:
                if dlg.isVisible() and hasattr(dlg, "finding_count"):
                    out.append(dlg)
            except RuntimeError:
                pass
        return out

    def _note_finding_source(self, key, shape_index):
        """A finding opened from a check's window: that window's list becomes what F8 steps through, and the
        image takes the keyboard (arrows, Del, relabel act on the shape at once)."""
        if getattr(self, "_stepping", False):
            return
        from PyQt6.QtWidgets import QApplication
        w = QApplication.activeWindow()
        if w is None or not hasattr(w, "finding_count") or w is self.window():
            return
        try:
            pos = w.finding_index(key, shape_index)
        except Exception as _exc:
            _ignored(_exc)
            return
        self._finding_src, self._finding_pos = w, pos
        self._finding_cur = (key, shape_index)
        self._focus_canvas_soon()

    def _focus_canvas_soon(self):
        def go():
            host = self.window()
            host.raise_()
            host.activateWindow()
            self.canvas.setFocus()
        utils.later(0, self, go)

    def step_finding(self, step: int, source=None):
        """F8 / Shift+F8: the next / previous finding of the check used last — opened on the image with its shape
        selected, the keyboard on the image. `source`: F8 pressed in that check's own window."""
        if source is not None and source is not getattr(self, "_finding_src", None):
            self._finding_src = source
            sel = self.canvas.selected_group
            self._finding_pos = source.finding_index(self.current_key, sel.index if sel is not None else None,
                                                     exact=True) if self.current_key and sel is not None else -1
        src = getattr(self, "_finding_src", None)
        try:
            alive = src is not None and src.isVisible()
        except RuntimeError:
            alive = False
        if not alive:
            wins = self._finding_windows()
            src = wins[-1] if wins else None
            self._finding_src, self._finding_pos = src, -1
        if src is None:
            self.statusBar().showMessage("F8 steps through a check's findings — open a check (QC drawer) and run it "
                                         "first", 6000)
            return
        n = src.finding_count()
        if not n:
            self.statusBar().showMessage(f"No findings in {getattr(src, 'TITLE', 'this check')}", 5000)
            return
        pos = getattr(self, "_finding_pos", -1)
        cur = getattr(self, "_finding_cur", None)
        here = src.finding_index(*cur, exact=True) if cur is not None and pos >= 0 else -1
        if here >= 0:
            pos = here                                     # the list may have changed under us (a rescan)
        elif pos >= 0 and cur is not None and step > 0:
            pos -= 1                                       # that finding was fixed and left: the next one moved up
        new = pos + step if pos >= 0 or step > 0 else n - 1
        if not 0 <= new < n:
            self.statusBar().showMessage(f"{'Last' if step > 0 else 'First'} finding of "
                                         f"{getattr(src, 'TITLE', 'this check')} ({n:,})", 5000)
            return
        self._finding_pos = new
        self._stepping = True
        try:
            src.open_finding(new)
        finally:
            self._stepping = False
        self._finding_cur = src.finding_at(new) if hasattr(src, "finding_at") else None
        self.canvas.setFocus()
        self.statusBar().showMessage(f"Finding {new + 1:,} of {n:,} · {getattr(src, 'TITLE', '')} — "
                                     "F8 next · Shift+F8 previous · arrows move the shape", 10000)

    def _apply_extras(self, shape_index, overlay, focus_box):
        if overlay:
            self.canvas.set_overlay(overlay)
        if shape_index is not None:
            self.canvas.focus_shape(shape_index)
        elif focus_box is not None:
            from PyQt6.QtCore import QPointF, QRectF
            x1, y1, x2, y2 = focus_box
            self.canvas.focus_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())

    def open_class_gallery(self) -> "object | None":
        """The Class Gallery (opened if needed) — for tools that hand it a class or an image list."""
        self.open_tool("class_gallery")
        return self.open_dialogs.get("class_gallery")

    def open_review_grid(self, keys, title: str):
        self.open_tool("review_grid")
        dlg = self.open_dialogs.get("review_grid")
        if dlg is not None:
            dlg.load_custom(keys, title)

    def open_crop_instances(self, pairs, title: str):
        """Crop gallery on an explicit list of (file key, shape index) — used by Model Check drill-downs."""
        self.open_tool("crop_gallery")
        dlg = self.open_dialogs.get("crop_gallery")
        if dlg is not None:
            dlg.show_instances(pairs, title)

    def apply_file_filter(self, keys, title: str):
        keys = [k for k in keys if self.session and k in self.session.by_key]
        if not keys:
            self.statusBar().showMessage("No matching files", 4000)
            return
        self.file_dock.set_key_filter(keys, title)
        self.file_list_dock.show()
        self.file_list_dock.raise_()
        self._update_position()
