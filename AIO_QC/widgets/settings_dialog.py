# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One Settings window for each mode (Image QC, Frame Extractor, Data Studio) - laid out like Labelme 7.8's:
a section list with icons and a "Search settings" box (Ctrl+F) on the left, one scrolling page of grouped settings on
the right that the list follows, and changes that apply at once (Close, no OK / Cancel). On top of that: reset to the
default per setting (↺) and per section, Reset all, and Import / Export of the settings as a JSON file.

A mode describes its settings as data - Section(title, icon, [Setting ...]) - and the window builds itself from that:

    Setting(key, title, kind, get=..., set=..., default=..., description=..., aliases=(...), choices=[(v, label)])
    kind: bool · choice · int · float · text · file · folder · button (runs set(None)) · info (read-only text)

Helpers bind a setting to what already exists, so nothing is implemented twice:
    from_action(key, action, ...)          a checkable QAction (its own handler runs: action.trigger())
    from_actions(key, {value: action}, ...) an exclusive group of checkable QActions
    from_store(key, settings, name, ...)   a QSettings value (+ an optional apply(value) for the live window)
    command(key, title, text, fn, ...)     a button that opens a sub-dialog or runs something

Menus keep commands (things you do); preferences (things you set) live here. The window is opened by the gear icon in
every mode's menu bar corner, File ▸ Settings… (Ctrl+,) and the command palette (shell.AppShell._add_switch).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from AIO_QC.utils.qt_utils import fit_to_screen, newIcon

NO_DEFAULT = object()
KINDS = ("bool", "choice", "int", "float", "text", "file", "folder", "button", "info")


@dataclass
class Setting:
    key: str                                  # unique and stable (export files use it): "fx.playback.engine"
    title: str
    kind: str
    get: Callable[[], Any] | None = None
    set: Callable[[Any], Any] | None = None
    default: Any = NO_DEFAULT
    description: str = ""
    aliases: tuple = ()                       # extra search words
    choices: list | None = None               # [(value, label)] for kind == "choice"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    suffix: str = ""
    decimals: int = 2
    button: str = ""                          # button text (kind == "button")
    file_filter: str = "All files (*)"
    enabled: Callable[[], bool] | None = None
    note: str = ""                            # small print: "applies to the next video", "after a restart" …
    exportable: bool = True

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.key}: unknown kind {self.kind!r}")

    @property
    def has_default(self):
        return self.default is not NO_DEFAULT and self.kind not in ("button", "info")

    def value(self):
        return self.get() if self.get else None

    def is_default(self):
        return not self.has_default or _same(self.value(), self.default)

    def valid(self, v):
        """v coerced to this setting's type, or raise ValueError (imports, resets)."""
        k = self.kind
        if k == "bool":
            if isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        if k == "choice":
            vals = [c[0] for c in (self.choices or [])]
            if v not in vals:
                hit = [c for c in vals if str(c) == str(v)]
                if not hit:
                    raise ValueError(f"{v!r} is not one of {vals}")
                v = hit[0]
            return v
        if k in ("int", "float"):
            v = int(round(float(v))) if k == "int" else float(v)
            if self.minimum is not None and v < self.minimum:
                raise ValueError(f"below {self.minimum}")
            if self.maximum is not None and v > self.maximum:
                raise ValueError(f"above {self.maximum}")
            return v
        if k in ("text", "file", "folder"):
            return "" if v is None else str(v)
        raise ValueError("not a value setting")


@dataclass
class Section:
    title: str
    icon: str
    settings: list = field(default_factory=list)
    description: str = ""


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


# ------------------------------------------------------------------ binding helpers
def from_action(key, action, title=None, description=None, default=NO_DEFAULT, **kw) -> Setting:
    """A checkable QAction as an on / off setting; its own handler does the work (and persists it)."""
    def put(v):
        if action.isChecked() != bool(v):
            action.trigger()
    return Setting(key, title or action.text().replace("&", ""), "bool", get=lambda: bool(action.isChecked()),
                   set=put, default=default, description=description if description is not None else
                   (action.toolTip() if action.toolTip() != action.text() else ""),
                   enabled=lambda: bool(action.isEnabled()), **kw)


def from_actions(key, actions: dict, title, labels: dict | None = None, default=NO_DEFAULT, description="",
                 **kw) -> Setting:
    """An exclusive group of checkable QActions as one choice."""
    def get():
        return next((k for k, a in actions.items() if a.isChecked()), None)

    def put(v):
        a = actions[v]
        if not a.isChecked():
            a.trigger()
    choices = [(k, (labels or {}).get(k) or a.text().replace("&", "")) for k, a in actions.items()]
    return Setting(key, title, "choice", get=get, set=put, choices=choices, default=default,
                   description=description,
                   enabled=lambda: any(a.isEnabled() for a in actions.values()), **kw)


def from_store(key, settings, name, kind, default, title, apply: Callable | None = None, **kw) -> Setting:
    """A QSettings value; apply(value) updates the live window after it is stored."""
    typ = {"bool": bool, "int": int, "float": float}.get(kind)

    def get():
        try:
            v = settings.value(name, default, type=typ) if typ else settings.value(name, default)
        except (TypeError, ValueError):
            v = default
        return default if v is None else v

    def put(v):
        settings.setValue(name, v)
        if apply is not None:
            apply(v)
    return Setting(key, title, kind, get=get, set=put, default=default, **kw)


def command(key, title, text, fn, description="", **kw) -> Setting:
    return Setting(key, title, "button", set=lambda _v=None: fn(), button=text, description=description,
                   exportable=False, **kw)


# ------------------------------------------------------------------ the window
class _Row(QWidget):
    """One setting: title, description, editor, ↺ reset."""

    def __init__(self, dlg, s: Setting):
        super().__init__()
        self.dlg, self.s = dlg, s
        self.setObjectName(f"setting:{s.key}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        from PyQt6.QtWidgets import QGridLayout
        lay = QGridLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(2)
        lay.setColumnStretch(0, 1)
        self.title = QLabel(s.title)
        f = self.title.font()
        f.setWeight(f.Weight.DemiBold)
        self.title.setFont(f)
        lay.addWidget(self.title, 0, 0)
        wide = s.kind in ("text", "file", "folder")             # long editors go under the text, full width
        if s.description or s.note:
            d = QLabel(s.description + (f"<br><i>{s.note}</i>" if s.note else ""))
            d.setWordWrap(True)
            d.setTextFormat(Qt.TextFormat.RichText)
            d.setOpenExternalLinks(True)
            d.setStyleSheet("color: palette(placeholder-text);")
            lay.addWidget(d, 1, 0, 1, 1 if not wide else 2)
        self.editor = self._editor()
        if self.editor is not None:
            self.editor.setAccessibleName(s.title)
            if wide:
                lay.addWidget(self.editor, 2, 0, 1, 2)
            else:
                lay.addWidget(self.editor, 0, 1, 2, 1, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self.reset = QToolButton()
        self.reset.setIcon(newIcon("reset-default"))            # sharp icon (was a ↺ character)
        self.reset.setIconSize(QSize(18, 18))
        self.reset.setAutoRaise(True)
        self.reset.setToolTip(f"Back to the default ({self._label(s.default)})" if s.has_default else "")
        self.reset.setAccessibleName(f"Reset {s.title}")
        self.reset.clicked.connect(lambda _c=False: self.dlg.apply(s, s.default))
        sp = self.reset.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.reset.setSizePolicy(sp)
        lay.addWidget(self.reset, 0, 2, 3 if wide else 2, 1, Qt.AlignmentFlag.AlignVCenter)

    def _label(self, v):
        if self.s.kind == "choice":
            return next((lbl for val, lbl in (self.s.choices or []) if val == v), str(v))
        if self.s.kind == "bool":
            return "on" if v else "off"
        return f"{v}{self.s.suffix}"

    def _editor(self):
        s, dlg = self.s, self.dlg
        k = s.kind
        if k == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda on: dlg.apply(s, bool(on)))
            return w
        if k == "choice":
            w = QComboBox()
            for val, label in s.choices or []:
                w.addItem(str(label), val)
            w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            w.activated.connect(lambda i: dlg.apply(s, w.itemData(i)))
            return w
        if k in ("int", "float"):
            w = QSpinBox() if k == "int" else QDoubleSpinBox()
            if k == "float":
                w.setDecimals(s.decimals)
            lo = s.minimum if s.minimum is not None else (-2 ** 31 + 1 if k == "int" else -1e12)
            hi = s.maximum if s.maximum is not None else (2 ** 31 - 1 if k == "int" else 1e12)
            w.setRange(int(lo), int(hi)) if k == "int" else w.setRange(float(lo), float(hi))
            if s.step:
                w.setSingleStep(int(s.step) if k == "int" else float(s.step))
            if s.suffix:
                w.setSuffix(s.suffix)
            w.setKeyboardTracking(False)
            w.setGroupSeparatorShown(True)
            w.setMinimumWidth(110)
            w.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            w.valueChanged.connect(lambda val: dlg.apply(s, val))
            return w
        if k in ("text", "file", "folder"):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit()
            e.setMinimumWidth(260)
            e.editingFinished.connect(lambda: e.isModified() and (e.setModified(False), dlg.apply(s, e.text())))
            h.addWidget(e)
            if k != "text":
                b = QPushButton("Browse…")
                b.clicked.connect(lambda _c=False: self._browse(e))
                h.addWidget(b)
            box.line = e
            return box
        if k == "button":
            b = QPushButton((s.button or "Open…").replace("&", "&&"))
            b.clicked.connect(lambda _c=False: dlg.run(s))
            return b
        if k == "info":
            lab = QLabel()
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return lab
        return None

    def _browse(self, line):
        s = self.s
        if s.kind == "folder":
            p = QFileDialog.getExistingDirectory(self, s.title, line.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, s.title, line.text(), s.file_filter)
        if p:
            line.setText(p)
            self.dlg.apply(s, p)

    def refresh(self):
        s, w = self.s, self.editor
        try:
            v = s.value()
        except Exception:
            v = None
        if w is not None:
            w.blockSignals(True)
            try:
                if s.kind == "bool":
                    w.setChecked(bool(v))
                elif s.kind == "choice":
                    i = w.findData(v)
                    if i < 0 and v is not None:
                        i = next((j for j in range(w.count()) if str(w.itemData(j)) == str(v)), -1)
                    w.setCurrentIndex(i)
                elif s.kind in ("int", "float") and v is not None:
                    w.setValue(int(v) if s.kind == "int" else float(v))
                elif s.kind in ("text", "file", "folder"):
                    if not w.line.hasFocus():
                        w.line.setText("" if v is None else str(v))
                elif s.kind == "info":
                    w.setText("" if v is None else str(v))
            finally:
                w.blockSignals(False)
            on = True
            if s.enabled is not None:
                try:
                    on = bool(s.enabled())
                except Exception:
                    on = False
            w.setEnabled(on)
        self.reset.setVisible(bool(s.has_default and not s.is_default()))


class SettingsDialog(QDialog):
    def __init__(self, parent, sections: list, title="Settings", app_name="AIO QC"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("settingsDialog")
        self.app_name = app_name
        self.sections = [s for s in sections if s.settings]
        self.rows: dict[str, _Row] = {}
        self.section_resets = []
        self.groups: list[QGroupBox] = []
        self._syncing = False
        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        # ---- sidebar
        side = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search settings")
        self.search.setAccessibleName("Search settings")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._update_search)
        self.search.installEventFilter(self)
        side.addWidget(self.search)
        self.nav = QListWidget()
        self.nav.setAccessibleName("Settings sections")
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setMinimumWidth(210)
        self.nav.setMaximumWidth(260)
        self.nav.setWordWrap(True)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.nav.itemActivated.connect(self._activate)
        self.nav.itemClicked.connect(self._activate)
        side.addWidget(self.nav, 1)
        body.addLayout(side)
        # ---- page
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.content = content
        page = QVBoxLayout(content)
        for sec in self.sections:
            g = QGroupBox(sec.title.replace("&", "&&"))
            g.setObjectName(f"section:{sec.title}")
            gl = QVBoxLayout(g)
            gl.setSpacing(0)
            if sec.description:
                d = QLabel(sec.description)
                d.setWordWrap(True)
                d.setStyleSheet("color: palette(placeholder-text); padding: 2px 8px 6px 8px;")
                gl.addWidget(d)
            for s in sec.settings:
                if s.key in self.rows:
                    raise ValueError(f"duplicate setting key {s.key}")
                r = _Row(self, s)
                self.rows[s.key] = r
                gl.addWidget(r)
            if any(s.has_default for s in sec.settings):
                rs = QPushButton("Reset section to defaults")
                rs.setIcon(newIcon("reset-default"))
                rs.setFlat(True)
                rs.setObjectName(f"reset:{sec.title}")
                rs.setProperty("section_settings", [x.key for x in sec.settings])
                self.section_resets.append((rs, sec))
                rs.clicked.connect(lambda _c=False, sec=sec: self.reset_section(sec))
                hl = QHBoxLayout()
                hl.addStretch(1)
                hl.addWidget(rs)
                gl.addLayout(hl)
            page.addWidget(g)
            self.groups.append(g)
        page.addStretch(1)
        self.scroll.setWidget(content)
        self.scroll.verticalScrollBar().valueChanged.connect(self._sync_nav)
        body.addWidget(self.scroll, 1)
        # ---- footer
        foot = QHBoxLayout()
        for name, text, slot in (("settings_import", "Import…", self.import_file),
                                 ("settings_export", "Export…", self.export_file),
                                 ("settings_reset_all", "Reset all…", self.reset_all)):
            b = QPushButton(text)
            b.setObjectName(name)
            if name == "settings_reset_all":
                b.setIcon(newIcon("reset-default"))
            b.clicked.connect(slot)
            foot.addWidget(b)
        foot.addStretch(1)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        foot.addWidget(close)
        outer.addLayout(foot)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self, activated=self.focus_search)
        self._highlight = QTimer(self, singleShot=True, interval=1400, timeout=self._unhighlight)
        self._lit = None
        self._show_sections()
        self.refresh()
        self.resize(860, 620)

    # ---- values
    def refresh(self):
        for r in self.rows.values():
            r.refresh()
        for b, sec in self.section_resets:
            b.setEnabled(any(x.has_default and not x.is_default() for x in sec.settings))

    def apply(self, s: Setting, v):
        if s.set is None:
            return
        try:
            s.set(s.valid(v))
        except Exception as e:
            QMessageBox.warning(self, self.windowTitle(), f"{s.title}: {e}")
        self.refresh()

    def run(self, s: Setting):
        try:
            s.set(None)
        except Exception as e:
            QMessageBox.warning(self, self.windowTitle(), f"{s.title}: {e}")
        self.refresh()

    def reset_section(self, sec):
        for s in sec.settings:
            if s.has_default and not s.is_default():
                self.apply(s, s.default)

    def reset_all(self):
        r = QMessageBox.question(self, "Reset all settings",
                                 "Put every setting on this page back to its default?")
        if r == QMessageBox.StandardButton.Yes:
            for sec in self.sections:
                self.reset_section(sec)

    def values(self) -> dict:
        out = {}
        for sec in self.sections:
            for s in sec.settings:
                if s.exportable and s.kind not in ("button", "info") and s.get is not None:
                    try:
                        out[s.key] = s.value()
                    except Exception:
                        pass
        return out

    def load_values(self, data: dict) -> tuple[int, list]:
        """Apply {key: value}; (how many applied, [problems])."""
        n, problems = 0, []
        for key, v in data.items():
            r = self.rows.get(key)
            if r is None or not r.s.exportable or r.s.set is None:
                problems.append(f"{key}: not a setting here")
                continue
            try:
                vv = r.s.valid(v)
            except ValueError as e:
                problems.append(f"{key}: {e}")
                continue
            if not _same(r.s.value(), vv):
                try:
                    r.s.set(vv)
                    n += 1
                except Exception as e:
                    problems.append(f"{key}: {e}")
        self.refresh()
        return n, problems

    def export_file(self):
        from AIO_QC.utils.save_dirs import ask_save
        p, _ = ask_save(self, "Export settings", "aio_qc_settings.json", "Settings (*.json)", kind="exports")
        if not p:
            return
        data = {"app": self.app_name, "page": self.windowTitle(), "settings": self.values()}
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            QMessageBox.warning(self, "Export settings", str(e))

    def import_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import settings", "", "Settings (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            vals = data.get("settings", data) if isinstance(data, dict) else None
            if not isinstance(vals, dict):
                raise ValueError("not a settings file")
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Import settings", f"Can't read {p}:\n{e}")
            return
        n, problems = self.load_values(vals)
        msg = f"{n} setting(s) changed."
        if problems:
            msg += "\n\nSkipped:\n" + "\n".join(problems[:15]) + ("\n…" if len(problems) > 15 else "")
        QMessageBox.information(self, "Import settings", msg)

    # ---- navigation / search
    def _show_sections(self):
        self.nav.clear()
        self.nav.setAccessibleName("Settings sections")
        for i, sec in enumerate(self.sections):
            it = QListWidgetItem(newIcon(sec.icon), sec.title)
            it.setData(Qt.ItemDataRole.UserRole, ("section", i))
            self.nav.addItem(it)
        if self.sections:
            self.nav.setCurrentRow(0)

    def _matches(self, text):
        terms = text.casefold().split()
        out = []
        for i, sec in enumerate(self.sections):
            for s in sec.settings:
                hay = " ".join((s.title, s.description, sec.title, " ".join(s.aliases),
                                " ".join(str(c[1]) for c in (s.choices or [])), s.button)).casefold()
                if all(t in hay for t in terms):
                    score = (0 if all(t in s.title.casefold() for t in terms) else 1)
                    out.append((score, i, s))
        out.sort(key=lambda x: (x[0], x[1]))
        return out

    def _update_search(self, text):
        if not text.strip():
            self._show_sections()
            return
        self.nav.clear()
        self.nav.setAccessibleName("Matching settings")
        hits = self._matches(text)
        for _score, i, s in hits:
            it = QListWidgetItem(f"{s.title}\n{self.sections[i].title}")
            it.setData(Qt.ItemDataRole.UserRole, ("setting", s.key))
            it.setToolTip(s.description)
            self.nav.addItem(it)
        if not hits:
            it = QListWidgetItem("No matching settings")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.nav.addItem(it)
        else:
            self.nav.setCurrentRow(0)

    def _activate(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data[0] == "section":
            self._scroll_to(self.groups[data[1]])
            return
        row = self.rows[data[1]]
        sec_i = next(i for i, sec in enumerate(self.sections) if row.s in sec.settings)
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._show_sections()
        self._syncing = True
        self.nav.setCurrentRow(sec_i)
        self._syncing = False
        self.show_setting(data[1])

    def show_setting(self, key):
        row = self.rows[key]
        self._scroll_to(row, margin=40)
        self._unhighlight()
        row.setStyleSheet("#%s { background: palette(highlight); border-radius: 4px; }"
                          % row.objectName().replace(":", "\\:").replace(".", "\\."))
        self._lit = row
        self._highlight.start()
        if row.editor is not None:
            target = getattr(row.editor, "line", row.editor)
            if target.isEnabled():
                target.setFocus(Qt.FocusReason.OtherFocusReason)

    def _unhighlight(self):
        if self._lit is not None:
            self._lit.setStyleSheet("")
            self._lit = None

    def _scroll_to(self, w, margin=0):
        y = w.mapTo(self.content, w.rect().topLeft()).y()
        self._syncing = True
        self.scroll.verticalScrollBar().setValue(max(0, y - margin))
        self._syncing = False

    def _sync_nav(self, value):
        if self._syncing or self.search.text().strip():
            return
        view_h = self.scroll.viewport().height()
        reading = value + min(value, view_h // 2)
        active = 0
        for i, g in enumerate(self.groups):
            if g.y() <= reading + 4:
                active = i
        bar = self.scroll.verticalScrollBar()
        if value >= bar.maximum() and bar.maximum() > 0:
            active = len(self.groups) - 1
        self.nav.blockSignals(True)
        self.nav.setCurrentRow(active)
        self.nav.blockSignals(False)

    def focus_search(self):
        self.search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.search.selectAll()

    def eventFilter(self, obj, ev):
        if obj is self.search and ev.type() == QEvent.Type.KeyPress:
            if ev.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.search.text().strip():
                self.nav.setFocus()
                return False
            if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.search.text().strip():
                it = self.nav.currentItem() or self.nav.item(0)
                if it is not None:
                    self._activate(it)
                return True
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape and self.search.text().strip():
            self.search.clear()
            self.focus_search()
            return
        super().keyPressEvent(ev)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()
        fit_to_screen(self)
        self.focus_search()


def general_section() -> Section:
    """Settings every mode shares (one value for the whole app)."""
    from AIO_QC.widgets import motion

    def set_anim(m):
        motion.set_mode(m)
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():         # the old menu items (palette / shortcuts) follow
            for act in w.findChildren(QAction, f"animations_{m}"):
                act.setChecked(True)
            legacy = getattr(getattr(w, "a", None), "animations", None)
            if legacy is not None:
                legacy.blockSignals(True)
                legacy.setChecked(m != "off")
                legacy.blockSignals(False)
    return Section("General", "palette", [
        Setting("app.animations", "Animations", "choice", get=motion.mode, set=set_anim, default="full",
                choices=[("full", "Full"), ("reduced", "Reduced"), ("off", "Off")],
                description="Smooth motion for panels, drawers, dialogs, mode switches and the tours - in every "
                            "mode. Reduced: shorter and gentler; Off: everything changes at once.",
                aliases=("motion", "transitions", "fade")),
    ], "These apply to Image QC, the Frame Extractor and Data Studio alike.")


def open_settings(win, sections, title=None, key: str | None = None):
    """Show `win`'s settings (modal); key = jump to that setting."""
    dlg = SettingsDialog(win, sections, title or "Settings")
    if key and key in dlg.rows:
        QTimer.singleShot(0, lambda: dlg.show_setting(key))
    dlg.exec()
    return dlg


def install_settings(win):
    """The gear button in the menu-bar corner (next to the mode switch) and File ▸ Settings… (Ctrl+,).
    Safe to call again after the corner widget changed (the shell adds the mode switch to the Frame Extractor)."""
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QHBoxLayout as _H
    mb = win.menuBar()
    if getattr(win, "_settings_action", None) is None:
        act = QAction(newIcon("settings"), "Settings…", win)
        act.setObjectName("open_settings")
        act.setShortcut(QKeySequence("Ctrl+,"))
        act.setMenuRole(QAction.MenuRole.PreferencesRole)
        act.setToolTip("All preferences of this mode, searchable (Ctrl+,)")
        act.triggered.connect(lambda _c=False: win.open_settings())
        file_menu = mb.actions()[0].menu() if mb.actions() else None
        if file_menu is not None:
            acts = [x for x in file_menu.actions()]
            last = acts[-1] if acts else None                   # Exit / Quit stays last
            file_menu.insertAction(last, act)
            file_menu.insertSeparator(last)
        else:
            win.addAction(act)
        win._settings_action = act
    corner = mb.cornerWidget(Qt.Corner.TopRightCorner)
    if corner is not None and corner.objectName() == "settingsCorner":
        return
    # A corner box from an earlier call that something else replaced as the corner widget (the shell puts the mode
    # switch into the Frame Extractor's corner after the window installed its gear): QMenuBar does not delete a
    # replaced corner widget, so it would stay at (0, 0) over the File menu. Its gear goes; anything else it held
    # moves into the new box.
    extras = []
    for old in mb.findChildren(QWidget, "settingsCorner", Qt.FindChildOption.FindDirectChildrenOnly):
        for w in old.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if w.objectName() != "settingsButton" and w is not corner:
                extras.append(w)
        old.hide()
        old.setObjectName("")
        old.deleteLater()
    box = QWidget(mb)
    box.setObjectName("settingsCorner")
    h = _H(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    btn = QToolButton(box)
    btn.setObjectName("settingsButton")
    btn.setIcon(newIcon("settings"))
    btn.setProperty("aio_icon", "settings")
    btn.setIconSize(QSize(18, 18))
    btn.setAutoRaise(True)
    btn.setToolTip("Settings (Ctrl+,)")
    btn.setAccessibleName("Settings")
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.clicked.connect(lambda _c=False: win.open_settings())
    h.addWidget(btn)
    for w in ([corner] if corner is not None else []) + extras:
        w.setParent(box)
        h.addWidget(w)
        w.show()
    mb.setCornerWidget(box, Qt.Corner.TopRightCorner)
    box.show()
    win._settings_button = btn
