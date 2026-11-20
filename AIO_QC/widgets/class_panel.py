# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The Annotations panel of a classification dataset: the class of the image on screen (or of the images selected in
the file list), changed with one click.

    every class with its colour, its number key (1 … 9) and how many images it has; the current class ticked —
    for several selected images a dash marks a class only some of them have
    one class per image: click a class to move the image(s) there (class folders: the file moves into that folder)
    several classes per image (multi-label): click toggles the class on / off for all of them
    type to filter (Enter takes the first match); "New class…" adds one; every change is one Ctrl+Z
"""
from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
                             QWidget)

from AIO_QC.utils import style as apply_style


def _dot(color) -> QIcon:
    """A class's swatch: a bold filled dot with a thin darker ring — never the box icon, which means a box shape."""
    dpr = 2.0
    pm = QPixmap(int(14 * dpr), int(14 * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    c = QColor(color) if color is not None else QColor(128, 128, 128)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(c.darker(150), 1.0))
    p.setBrush(c)
    p.drawEllipse(QRectF(2.5, 2.5, 9, 9))
    p.end()
    return QIcon(pm)


class ClassPanel(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setObjectName("classPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        self.head.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.head)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter classes — Enter sets the first match")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        self.filter.returnPressed.connect(self._take_first)
        lay.addWidget(self.filter)
        self.list = QListWidget()
        self.list.setIconSize(QSize(14, 14))
        self.list.setUniformItemSizes(True)
        from AIO_QC.widgets.panels import BadgeDelegate
        self.list.setItemDelegate(BadgeDelegate(self.list))         # the image count as the Labels panel's pill
        self.list.itemClicked.connect(self._clicked)
        self.list.itemActivated.connect(self._clicked)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.new_btn = QPushButton("New class…")
        self.new_btn.setToolTip("Give the image(s) a class that does not exist yet")
        self.new_btn.clicked.connect(self._new_class)
        self.more_btn = QPushButton("Classes…")
        self.more_btn.setToolTip("Rename, merge, remove and order the classes")
        self.more_btn.clicked.connect(lambda: self.app.open_tool("remove_class"))
        self.issues_btn = QPushButton("To issues")
        self.issues_btn.setToolTip("I — set the image(s) aside in the issues folder, by class (issues/<class>/); "
                                   "Ctrl+Z puts them back")
        self.issues_btn.clicked.connect(lambda: self.app.send_to_issues(keys=self.keys()))
        self.issues_btn.setVisible(False)
        row.addWidget(self.new_btn)
        row.addWidget(self.more_btn)
        row.addWidget(self.issues_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.hint = QLabel("1 … 9 on the image set a class · C opens the class picker · Ctrl+Z undoes")
        self.hint.setWordWrap(True)
        apply_style(self.hint, "color:@muted;")
        lay.addWidget(self.hint)

    # ------------------------------------------------------------------ what it is about
    def keys(self) -> list[str]:
        """The images a click changes: the ones selected in the file list when the shown image is among them (or
        nothing is shown), else the shown image."""
        app, s = self.app, self.app.session
        if s is None:
            return []
        sel = [k for k in app.file_dock.selected_keys() if k in s.by_key]
        cur = app.current_key if app.current_key in s.by_key else None
        if len(sel) > 1 and (cur is None or cur in sel):
            return sel
        return [cur] if cur else sel[:1]

    def refresh(self):
        s = self.app.session
        self.list.clear()
        if s is None or not s.is_classification:
            self.head.setText("")
            self.issues_btn.setVisible(False)
            return
        keys = self.keys()
        have = Counter(t for k in keys for t in s.tags_of(k))
        multi = s.cls_mode == "multi"
        folders = s.type.value == "cls_folders"
        self.issues_btn.setVisible(folders)
        self.hint.setText("1 … 9 on the image set a class · C opens the class picker · " +
                          ("I sets it aside in issues/ · " if folders else "") + "Ctrl+Z undoes")
        totals = s.label_totals()
        names = list(s.cls_classes())
        n = len(keys)
        if not n:
            self.head.setText("<i>No image shown</i>")
        else:
            now = ", ".join(sorted(have)) or "no class"
            self.head.setText((f"<b>{n:,} images selected</b> · " if n > 1 else "This image: ") +
                              (f"<b>{now}</b>" if n == 1 else f"{len(have)} class(es) among them") +
                              ("<br><small>several classes per image — a click adds / removes one</small>" if multi
                               else ""))
        self.list.blockSignals(True)
        for i, c in enumerate(names):
            it = QListWidgetItem(c)
            it.setData(Qt.ItemDataRole.UserRole, c)
            k = have.get(c, 0)
            it.setCheckState(Qt.CheckState.Checked if n and k == n else Qt.CheckState.PartiallyChecked if k else
                             Qt.CheckState.Unchecked)
            col = self.app.label_colors.get(c) if hasattr(self.app, "label_colors") else None
            if col is None:
                from AIO_QC import utils
                utils.update_label_colors(c, self.app.label_colors)
                col = self.app.label_colors.get(c)
            it.setIcon(_dot(col))
            from AIO_QC.widgets.panels import COUNT_ROLE
            it.setText(f"{i + 1}   {c}" if i < 9 else f"     {c}")          # the number key in front
            it.setData(COUNT_ROLE, f"{totals.get(c, 0):,}")
            it.setToolTip(f"{c}: {totals.get(c, 0):,} image(s)" + (f" · key {i + 1}" if i < 9 else "") +
                          ("\nClick: move the image(s) into this class folder" if folders and not multi else
                           "\nClick: add / remove this class" if multi else "\nClick: set this class"))
            if k and n and k == n:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._apply_filter(self.filter.text())
        self.list.setEnabled(bool(n))
        self.new_btn.setEnabled(bool(n))

    # ------------------------------------------------------------------ actions
    def _apply_filter(self, text):
        t = text.strip().lower()
        for r in range(self.list.count()):
            it = self.list.item(r)
            it.setHidden(bool(t) and t not in str(it.data(Qt.ItemDataRole.UserRole)).lower())

    def _take_first(self):
        for r in range(self.list.count()):
            it = self.list.item(r)
            if not it.isHidden():
                self._clicked(it)
                self.filter.clear()
                return

    def _clicked(self, it):
        s, app = self.app.session, self.app
        name = it.data(Qt.ItemDataRole.UserRole)
        keys = self.keys()
        if s is None or not keys or not name:
            return
        if s.cls_mode == "multi":
            all_have = all(name in s.tags_of(k) for k in keys)
            app._apply_classes(keys, add=() if all_have else [name], remove=[name] if all_have else (),
                               what=("−" if all_have else "+") + name)
        else:
            if all(s.tags_of(k) == [name] for k in keys):
                return
            app._apply_classes(keys, tags=[name], what=f"class → {name}")
        self.refresh()

    def _new_class(self):
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        s, app = self.app.session, self.app
        keys = self.keys()
        if s is None or not keys:
            return
        name, ok = QInputDialog.getText(self, "New class", "Class name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if not s.base_adapter.can_use_label(name):
            QMessageBox.information(self, "New class", f"“{name}” cannot be a class name.")
            return
        if s.cls_mode == "multi":
            app._apply_classes(keys, add=[name], what=f"+{name}")
        else:
            app._apply_classes(keys, tags=[name], what=f"class → {name}")
        self.refresh()
