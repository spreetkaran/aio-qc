# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Tools ▸ Storage… — what AIO QC keeps next to the dataset (<root>/.aio_qc), and the controls to keep it small:
backup operations (export / delete), cached results per model / OCR setting (delete), editor sessions, old-style
backup folders (compact), the trash, retention settings, and "Clean up now".
"""
from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from AIO_QC.core import backups as B
from AIO_QC.core import storage as S
from AIO_QC.utils import newButton, newIcon, style as apply_style
from .inference_service import FuncWorker

ROLE = Qt.ItemDataRole.UserRole


def load_cfg(settings) -> dict:
    return {k: settings.value(f"storage/{k}", v, type=type(v)) for k, v in S.DEFAULTS.items()}


class StorageDialog(QDialog):
    TITLE = "Storage"

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(f"{self.TITLE} — {self.session.root.name}")
        self.setWindowIcon(newIcon("open"))
        self.resize(860, 620)
        self.worker = None
        st = app.settings
        lay = QVBoxLayout(self)

        self.header = QLabel("")
        self.header.setWordWrap(True)
        lay.addWidget(self.header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["What", "Size", "Details"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        h = self.tree.header()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        lay.addWidget(self.tree, 1)

        row = QHBoxLayout()
        self.export_btn = newButton("Export originals…", icon="save", slot=self._export,
                                    tip="Write the backed-up originals of the selected operation to a folder")
        self.delete_btn = newButton("Delete selected", icon="delete", slot=self._delete)
        self.compact_btn = newButton("Compact old backups", icon="tool-export", slot=self._compact_legacy,
                                     tip="Move old-style backup folders (full copies) into the compact store — "
                                         "nothing is lost, usually 90 %+ smaller")
        self.clean_btn = newButton("Clean up now", icon="tool-quality", slot=self._clean_now,
                                   tip="Apply the retention below, drop unused results, forget removed files")
        self.open_btn = newButton("Open folder", icon="open", slot=self._open_folder)
        for b in (self.export_btn, self.delete_btn, self.compact_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.clean_btn)
        row.addWidget(self.open_btn)
        lay.addLayout(row)

        box = QGroupBox("Retention (applied automatically after a dataset is indexed)")
        form = QFormLayout(box)
        cfg = load_cfg(st)
        self.backup_days = self._days(cfg["backup_days"], "Keep forever")
        self.session_days = self._days(cfg["session_days"], "Keep forever")
        self.result_days = self._days(cfg["result_days"], "Keep forever")
        form.addRow("Delete backups older than:", self.backup_days)
        form.addRow("Delete finished editor sessions after:", self.session_days)
        form.addRow("Drop model / OCR results unused for:", self.result_days)
        self.auto = QCheckBox("Clean up automatically")
        self.auto.setChecked(cfg["auto"])
        self.exact = QCheckBox("Keep embedded images (imageData) byte-exact in backups — uses about the image size "
                               "per file")
        self.exact.setChecked(st.value("storage/exact_image_data", False, type=bool))
        self.exact.setToolTip("Off (default): a backup stores the JSON without its embedded image when the image can "
                              "be rebuilt from the image file (restores are checked by SHA-256 and are byte-exact in "
                              "almost every case). JSONs without an image file always keep their embedded image.")
        form.addRow("", self.auto)
        form.addRow("", self.exact)
        for w in (self.backup_days, self.session_days, self.result_days):
            w.valueChanged.connect(self._save_cfg)
        self.auto.toggled.connect(self._save_cfg)
        self.exact.toggled.connect(self._save_cfg)
        lay.addWidget(box)

        foot = QHBoxLayout()
        self.status = QLabel("")
        apply_style(self.status, "color:@muted;")
        foot.addWidget(self.status, 1)
        foot.addWidget(newButton("Refresh", slot=self.refresh))
        foot.addWidget(newButton("Close", slot=self.close))
        lay.addLayout(foot)
        self.refresh()

    @staticmethod
    def _days(v: int, zero: str) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(0, 3650)
        sb.setSuffix(" days")
        sb.setSpecialValueText(zero)
        sb.setValue(int(v))
        return sb

    def _save_cfg(self, *_):
        st = self.app.settings
        st.setValue("storage/backup_days", self.backup_days.value())
        st.setValue("storage/session_days", self.session_days.value())
        st.setValue("storage/result_days", self.result_days.value())
        st.setValue("storage/auto", self.auto.isChecked())
        st.setValue("storage/exact_image_data", self.exact.isChecked())
        B.EXACT_IMAGE_DATA = self.exact.isChecked()

    def cfg(self) -> dict:
        return {"backup_days": self.backup_days.value(), "session_days": self.session_days.value(),
                "result_days": self.result_days.value(), "auto": self.auto.isChecked()}

    # ------------------------------------------------------------------ fill
    def refresh(self):
        self.tree.clear()
        parts = S.usage(self.session.root, self.session.cache)
        total = sum(p.bytes for p in parts if p.key != "trash")
        trash = next((p.bytes for p in parts if p.key == "trash"), 0)
        self.header.setText(f"<b>AIO QC uses {S.human(total)}</b> for this dataset in <code>.aio_qc</code>"
                            + (f" · trash {S.human(trash)}" if trash else "")
                            + "<br><span style='color:gray'>Images hard-linked into editor sessions cost nothing "
                              "and are not counted.</span>")
        for p in parts:
            top = QTreeWidgetItem([p.title, S.human(p.bytes), p.detail])
            top.setData(0, ROLE, (p.key, None))
            top.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.tree.addTopLevelItem(top)
            for it in p.items:
                child = self._child(p.key, it)
                if child is not None:
                    child.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    top.addChild(child)
            top.setExpanded(p.key in ("backups", "legacy") and len(p.items) <= 30)
        self._update_buttons()

    def _child(self, key, it) -> QTreeWidgetItem | None:
        if key == "backups":
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(it["created"]))
            ratio = (f" · originals {S.human(it['original_bytes'])}" if it["original_bytes"] else "")
            c = QTreeWidgetItem([f"{it['title'] or it['kind']}", S.human(it["stored_bytes"]),
                                 f"{when} · {it['files']:,} file(s){ratio}"])
        elif key == "cache":
            used = it.get("used")
            c = QTreeWidgetItem([_sig_title(it["sig"]), S.human(it["bytes"]),
                                 f"{it['rows']:,} result(s)" + (f" · last used {S._fmt_age(used)}" if used else "")])
        elif key == "legacy":
            c = QTreeWidgetItem([Path(it["path"]).name, S.human(it["bytes"]), "full copies"])
        elif key == "sessions":
            c = QTreeWidgetItem([Path(it["path"]).name, S.human(it["bytes"]),
                                 ("open" if it["active"] else "finished") + f" · {S._fmt_age(it['mtime'])}"])
        else:
            return None
        c.setData(0, ROLE, (key, it))
        return c

    def _selected(self) -> list[tuple[str, dict]]:
        out = []
        for item in self.tree.selectedItems():
            key, it = item.data(0, ROLE) or (None, None)
            if it is not None:
                out.append((key, it))
            elif key == "trash":
                out.append((key, {}))
        return out

    def _update_buttons(self):
        sel = self._selected()
        kinds = {k for k, _ in sel}
        self.export_btn.setEnabled(len(sel) == 1 and kinds == {"backups"})
        deletable = all(k in ("backups", "cache", "legacy", "trash") or (k == "sessions" and not it.get("active"))
                        for k, it in sel)
        self.delete_btn.setEnabled(bool(sel) and deletable and self.worker is None)
        self.compact_btn.setEnabled(bool(B.legacy_folders(self.session.root)) and self.worker is None)
        self.clean_btn.setEnabled(self.worker is None)

    # ------------------------------------------------------------------ actions
    def _run(self, fn, done_text):
        self.status.setText("Working …")
        for b in (self.delete_btn, self.compact_btn, self.clean_btn, self.export_btn):
            b.setEnabled(False)
        w = FuncWorker(fn, self)
        self.worker = w

        def finished(result):
            self.worker = None
            self.status.setText(done_text(result) if callable(done_text) else done_text)
            self.refresh()

        def failed(msg):
            self.worker = None
            logger.warning(f"storage: {msg}")
            self.status.setText(f"Failed: {msg}")
            self.refresh()

        w.done.connect(finished)
        w.failed.connect(failed)
        w.finished.connect(w.deleteLater)
        w.start()

    def _delete(self):
        sel = self._selected()
        if not sel:
            return
        names = "\n".join(f"• {self._name(k, it)}" for k, it in sel[:12]) + ("\n…" if len(sel) > 12 else "")
        warn = ""
        if any(k == "backups" for k, _ in sel):
            warn = "\n\nDeleted backups cannot be used for undo any more."
        if any(k == "trash" for k, _ in sel):
            warn += "\n\nFiles in the trash are deleted permanently."
        if QMessageBox.question(self, self.TITLE, f"Delete?\n\n{names}{warn}") != QMessageBox.StandardButton.Yes:
            return
        root, cache = self.session.root, self.session.cache

        def work():
            import shutil
            before = S.dir_bytes(root)
            store = B.BackupStore(root)
            for k, it in sel:
                if k == "backups":
                    store.delete(it["id"])
                elif k == "cache" and cache is not None:
                    cache.drop_sig(it["sig"])
                elif k in ("legacy", "sessions"):
                    shutil.rmtree(it["path"], ignore_errors=True)
                elif k == "trash":
                    shutil.rmtree(root / ".aio_qc_trash", ignore_errors=True)
            store.close()
            return max(0, before - S.dir_bytes(root))

        self._run(work, lambda freed: f"Deleted — freed {S.human(freed)} in .aio_qc.")

    @staticmethod
    def _name(k, it) -> str:
        if k == "backups":
            return it.get("title") or it.get("kind")
        if k == "cache":
            return _sig_title(it["sig"])
        if k == "trash":
            return "the trash folder"
        return Path(it.get("path", "")).name

    def _export(self):
        sel = self._selected()
        if len(sel) != 1:
            return
        op = sel[0][1]
        from AIO_QC.utils.save_dirs import ask_dir
        out = ask_dir(self, "Export the originals to …", kind="datasets", fallback=self.session.root.parent)
        if not out:
            return
        target = Path(out) / f"aio_qc_backup_{op['id']}_{time.strftime('%Y%m%d', time.localtime(op['created']))}"
        root = self.session.root

        def work():
            store = B.BackupStore(root)
            try:
                return store.export(op["id"], target)
            finally:
                store.close()

        self._run(work, lambda n: f"Exported {n:,} file(s) to {target}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(out))))

    def _compact_legacy(self):
        root = self.session.root
        self._run(lambda: B.migrate_legacy(root),
                  lambda r: f"Compacted {r[0]} folder(s) — {S.human(r[1])} of full copies moved into backups.sqlite.")

    def _clean_now(self):
        root, cache, cfg = self.session.root, self.session.cache, self.cfg()
        alive = [p for s in self.session.samples for p in (s.ann_path, s.image_path)]
        keep = self.app.backup_ops_in_use() if hasattr(self.app, "backup_ops_in_use") else set()

        def work():
            return S.housekeeping(root, cache, cfg, alive_paths=alive, keep_ops=keep)

        self._run(work, lambda r: (f"Freed {S.human(r['freed'])}. " if r["freed"] else "Nothing to free. ")
                  + "; ".join(r["notes"]))

    def _open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.root / ".aio_qc")))

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.wait(30000)
        super().closeEvent(e)


def _sig_title(sig: str) -> str:
    """Readable name of a cache signature."""
    parts = sig.split("|")
    head = parts[0]
    if head == "integrity":
        return "Image integrity checks"
    if head == "iq":
        return "Image quality metrics (blur / exposure)"
    if head == "ocr":
        return "OCR results · " + " · ".join(p for p in parts[1:5] if p)
    if len(parts) >= 4 and ("/" in head or "\\" in head or "." in head):
        return f"Model results · {Path(head).name} · imgsz {parts[3]}" + (" · SAHI" if "sahi" in sig else "")
    return sig if len(sig) < 80 else sig[:77] + "…"
