# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The AIO QC 3 design system: one set of tokens (colours, radii, spacing) turned into the Qt palette and one
app-wide style sheet, for light and dark. Every mode (Image QC, Frame Extractor, Data Studio) and every dialog
gets the same look; widgets keep their own small style sheets (utils.style with @role tokens) on top.

    design.apply(app)          # once at start (the shell) and whenever the appearance changes
    design.tokens()            # the active token dict (light or dark)
    design.mode()              # "system" | "light" | "dark"  (Settings ▸ General ▸ Appearance)

Base style: Fusion (the same on Windows, macOS and Linux, fully driven by the palette), wrapped by ui_style.AppStyle
(no pressed shift, one menu arrow). Buttons marked `variant="primary"` — and every dialog's default button — are
the accent-coloured main action.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication
from AIO_QC.core.diag import ignored as _ignored

SETTING = "ui/appearance"

LIGHT = {
    "accent": "#0B7A84", "accent_hover": "#096A73", "accent_pressed": "#075A61", "accent_soft": "#E6F4F5",
    "accent_mid": "#8FCBD0", "accent_text": "#0B6E77", "on_accent": "#FFFFFF",
    "text": "#1B1F24", "muted": "#4A5360", "faint": "#6B7480",
    "window": "#F6F7F9", "surface": "#FFFFFF", "surface2": "#F9FAFB", "hover": "#EEF1F4", "pressed": "#E3E7EB",
    "border": "#E3E6EA", "border_soft": "#EEF1F4", "control": "#D5DAE0", "track": "#E3E7EB",
    "scroll": "#C3C9D0", "scroll_hover": "#A3ABB5", "alt_row": "#FAFBFC",
    "canvas": "#24262A", "tooltip": "#FFFFFF", "on_tooltip": "#1B1F24",
    "error": "#C62828", "warn": "#D97706", "info": "#2F6FDB", "ok": "#2E7D32",
}
DARK = {
    "accent": "#2BA3AD", "accent_hover": "#3FB8C2", "accent_pressed": "#23909A", "accent_soft": "#163437",
    "accent_mid": "#1F6E75", "accent_text": "#5FC9D2", "on_accent": "#0B1214",
    "text": "#E6E9ED", "muted": "#A7B0BB", "faint": "#7E8792",
    "window": "#141518", "surface": "#1B1D21", "surface2": "#202227", "hover": "#262930", "pressed": "#2E323A",
    "border": "#2E3238", "border_soft": "#25282E", "control": "#3A3F47", "track": "#2E3238",
    "scroll": "#454B54", "scroll_hover": "#5A616B", "alt_row": "#1E2025",
    "canvas": "#0E0F11", "tooltip": "#25282E", "on_tooltip": "#E6E9ED",
    "error": "#F07171", "warn": "#F2B04C", "info": "#6EA2F5", "ok": "#5CC072",
}
RADIUS = 6
_ACTIVE = {"dark": None}


def _settings():
    from AIO_QC.utils import app_settings
    return app_settings()


def mode() -> str:
    try:
        m = str(_settings().value(SETTING, "system") or "system")
    except Exception as _exc:
        _ignored(_exc)
        m = "system"
    return m if m in ("system", "light", "dark") else "system"


def system_dark() -> bool:
    hints = QGuiApplication.styleHints() if QGuiApplication.instance() else None
    try:
        from PyQt6.QtCore import Qt
        return hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark
    except Exception as _exc:                                    # Qt < 6.5: no colour scheme
        _ignored(_exc)
        return False


def want_dark() -> bool:
    m = mode()
    return system_dark() if m == "system" else m == "dark"


def tokens(dark: bool | None = None) -> dict:
    if dark is None:
        dark = _ACTIVE["dark"] if _ACTIVE["dark"] is not None else want_dark()
    return DARK if dark else LIGHT


def palette(t: dict) -> QPalette:
    p = QPalette()
    R, G = QPalette.ColorRole, QPalette.ColorGroup
    c = lambda k: QColor(t[k])                          # noqa: E731
    for role, key in ((R.Window, "window"), (R.WindowText, "text"), (R.Base, "surface"), (R.AlternateBase, "alt_row"),
                      (R.Text, "text"), (R.Button, "surface"), (R.ButtonText, "text"), (R.BrightText, "on_accent"),
                      (R.Highlight, "accent"), (R.HighlightedText, "on_accent"), (R.ToolTipBase, "tooltip"),
                      (R.ToolTipText, "on_tooltip"), (R.Link, "accent_text"), (R.LinkVisited, "accent_text"),
                      (R.PlaceholderText, "faint"), (R.Light, "surface"), (R.Midlight, "border"),
                      (R.Mid, "control"), (R.Dark, "scroll"), (R.Shadow, "scroll_hover")):
        p.setColor(role, c(key))
    try:
        p.setColor(R.Accent, c("accent"))
    except AttributeError:
        pass
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(G.Disabled, role, c("faint"))
    p.setColor(G.Disabled, R.Highlight, c("control"))
    return p


def _asset_dir() -> Path:
    from AIO_QC.core.appdirs import data_home
    d = Path(data_home()) / "ui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chevron(color: str, up: bool = False) -> str:
    """A small chevron SVG for combo boxes / spin boxes (style sheets need an image there)."""
    d = "M6 15l6-6 6 6" if up else "M6 9l6 6 6-6"
    name = f"chev_{'up' if up else 'down'}_{color.lstrip('#')}.svg"
    p = _asset_dir() / name
    if not p.is_file():
        p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" '
                     f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
                     f'<path d="{d}"/></svg>', encoding="utf-8")
    return p.as_posix()


def _mark(color: str, kind: str) -> str:
    """Menu check marks: a clean tick (check boxes in menus) and a dot (one-of-many items)."""
    name = f"menu_{kind}_{color.lstrip('#')}.svg"
    p = _asset_dir() / name
    if not p.is_file():
        body = ('<path d="M5 12.5l4.2 4.2L19 7" fill="none" stroke="{c}" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round"/>' if kind == "tick" else '<circle cx="12" cy="12" r="4" fill="{c}"/>')
        p.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">'
                     + body.format(c=color) + '</svg>', encoding="utf-8")
    return p.as_posix()


def stylesheet(t: dict) -> str:
    down, up = _chevron(t["muted"]), _chevron(t["muted"], up=True)
    tick, dot = _mark(t["accent_text"], "tick"), _mark(t["accent_text"], "dot")
    ink_tick, ink_dot = _mark(t["text"], "tick"), _mark(t["text"], "dot")   # check marks in lists and tables
    r = RADIUS
    return f"""
QMainWindow, QDialog {{ background: {t['window']}; }}
#shelfChip {{ background: {t['surface']}; border: 1px solid {t['control']}; border-radius: 8px; }}
#shelfChip:hover {{ border-color: {t['accent']}; background: {t['hover']}; }}
#shelfChipState {{ color: {t['muted']}; }}
#shelfChipBar {{ background: {t['track']}; border: none; border-radius: 1px; }}
#shelfChipBar::chunk {{ background: {t['accent']}; border-radius: 1px; }}
#shelfChipClose {{ color: {t['muted']}; border: none; padding: 0 4px; font-size: 14px; }}
#shelfChipClose:hover {{ color: {t['error']}; }}
QToolTip {{ color: {t['on_tooltip']}; background: {t['tooltip']}; border: 1px solid {t['control']}; padding: 3px 7px;
           border-radius: 6px; }}

QMenuBar {{ background: {t['surface']}; border-bottom: 1px solid {t['border']}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 9px; border-radius: 5px; background: transparent; color: {t['text']}; }}
QMenuBar::item:selected, QMenuBar::item:pressed {{ background: {t['hover']}; }}
QMenu {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 28px 6px 10px; border-radius: 5px; color: {t['text']}; }}
QMenu::item:selected {{ background: {t['accent_soft']}; color: {t['text']}; }}
QMenu::item:disabled {{ color: {t['faint']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 6px; }}
QMenu::icon {{ padding-left: 8px; }}
QMenu::indicator {{ width: 16px; height: 16px; left: 8px; background: transparent; border: 0; }}
QAbstractItemView::indicator {{ width: 14px; height: 14px; border: 1px solid {t['control']}; border-radius: 3px;
                               background: {t['surface']}; }}
QAbstractItemView::indicator:checked {{ image: url("{ink_tick}"); }}
QAbstractItemView::indicator:indeterminate {{ image: url("{ink_dot}"); }}
QMenu::indicator:non-exclusive:checked {{ image: url("{tick}"); }}
QMenu::indicator:exclusive:checked {{ image: url("{dot}"); }}
QMenu::indicator:unchecked {{ image: none; }}
QMenu::icon:checked {{ background: {t['accent_soft']}; border: 1px solid {t['accent_mid']}; border-radius: 4px;
    padding: 2px; }}

QToolBar {{ background: {t['surface']}; border: 0; border-bottom: 1px solid {t['border']}; spacing: 3px;
           padding: 4px 8px; }}
QToolBar::separator {{ width: 1px; background: {t['border']}; margin: 7px 6px; }}
QToolButton {{ border: 0; border-radius: {r}px; padding: 4px; background: transparent; color: {t['text']}; }}
QToolButton:hover {{ background: {t['hover']}; }}
QToolButton:pressed {{ background: {t['pressed']}; }}
QToolButton:checked {{ background: {t['accent_soft']}; color: {t['accent_text']}; }}
QToolButton:disabled {{ color: {t['faint']}; }}
QToolButton[popupMode="2"] {{ padding-right: 14px; }}
QToolButton::menu-indicator {{ image: url("{down}"); subcontrol-origin: padding; subcontrol-position: center right;
    width: 10px; height: 10px; right: 2px; }}
QToolButton::menu-button {{ border: 0; width: 14px; }}
QToolButton::menu-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}

QPushButton {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['control']};
              border-radius: {r}px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:pressed {{ background: {t['pressed']}; }}
QPushButton:checked {{ background: {t['accent_soft']}; border-color: {t['accent_mid']}; color: {t['accent_text']}; }}
QPushButton:disabled {{ color: {t['faint']}; background: {t['window']}; border-color: {t['border']}; }}
QPushButton:flat {{ border: 0; background: transparent; }}
QPushButton::menu-indicator {{ image: url("{down}"); subcontrol-origin: padding; subcontrol-position: center right;
    width: 10px; height: 10px; right: -4px; }}
QPushButton:flat:hover {{ background: {t['hover']}; }}
QPushButton:default, QPushButton[variant="primary"] {{ background: {t['accent']}; color: {t['on_accent']};
              border: 1px solid {t['accent']}; font-weight: 600; }}
QPushButton:default:hover, QPushButton[variant="primary"]:hover {{ background: {t['accent_hover']};
              border-color: {t['accent_hover']}; }}
QPushButton:default:pressed, QPushButton[variant="primary"]:pressed {{ background: {t['accent_pressed']}; }}
QPushButton:default:disabled, QPushButton[variant="primary"]:disabled {{ background: {t['control']};
              border-color: {t['control']}; color: {t['surface']}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QTimeEdit {{
    background: {t['surface']}; color: {t['text']}; border: 1px solid {t['control']}; border-radius: {r}px;
    padding: 4px 8px; selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {t['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {t['faint']};
    background: {t['window']}; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: 0; }}
QComboBox::down-arrow {{ image: url("{down}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 6px;
    selection-background-color: {t['accent_soft']}; selection-color: {t['text']}; outline: 0; padding: 2px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; border: 0; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
    width: 18px; border: 0; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{up}"); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}

QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 7px 12px; border: 0;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTabBar::tab:selected {{ color: {t['accent_text']}; border-bottom: 2px solid {t['accent']}; }}
QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 8px; top: -1px; background: {t['surface']}; }}

QDockWidget {{ color: {t['text']}; }}
QDockWidget QListView, QDockWidget QListWidget, QDockWidget QTreeView {{ border: 0; border-radius: 0; }}
QDockWidget > QWidget {{ background: {t['surface']}; }}
QDockWidget::title {{ background: {t['surface']}; padding: 7px 10px; border-bottom: 1px solid {t['border']};
                     text-align: left; }}

QHeaderView::section {{ background: {t['surface2']}; color: {t['muted']}; border: 0;
    border-bottom: 1px solid {t['border']}; border-right: 1px solid {t['border_soft']}; padding: 6px 8px;
    font-weight: 600; }}
QTableView, QTreeView, QListView, QTableWidget, QTreeWidget, QListWidget {{ background: {t['surface']};
    color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px; gridline-color: {t['border_soft']};
    selection-background-color: {t['accent_soft']}; selection-color: {t['text']};
    alternate-background-color: {t['alt_row']}; outline: 0; }}
QTableView::item:hover, QTreeView::item:hover, QListView::item:hover {{ background: {t['hover']}; }}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {{ background: {t['accent_soft']};
    color: {t['text']}; }}
QTableCornerButton::section {{ background: {t['surface2']}; border: 0; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['scroll']}; border-radius: 4px; min-height: 28px; margin: 2px 3px; }}
QScrollBar::handle:horizontal {{ background: {t['scroll']}; border-radius: 4px; min-width: 28px; margin: 3px 2px; }}
QScrollBar::handle:hover {{ background: {t['scroll_hover']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: {t['track']}; border: 0; border-radius: 4px; color: {t['text']}; text-align: center;
               min-height: 14px; max-height: 18px; }}
QProgressBar::chunk {{ background: {t['accent_mid']}; border-radius: 4px; }}

QGroupBox {{ border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 12px 10px 8px 10px;
            background: {t['surface']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t['muted']}; }}

QStatusBar {{ background: {t['surface']}; border-top: 1px solid {t['border']}; color: {t['muted']}; }}
QStatusBar::item {{ border: 0; }}
QSplitter::handle {{ background: {t['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QMainWindow::separator {{ background: {t['border']}; width: 1px; height: 1px; }}
"""


UI_FAMILY, MONO_FAMILY = "Inter", "AIO Geist"
_FONTS = {"done": False}
# MONO_FAMILY is the font for numbers, file names, sizes, counters, timecodes and key hints (the file dock, the status
# bar, the canvas read-outs, badges …): "AIO Geist" — Geist (SIL OFL) with tabular figures by default and a slashed
# zero, the look of a phone's recording timer (00:00:01). Built by packaging/build_aio_geist.py. Inter stays the UI
# font. (Logs and code keep the system's fixed-width font: their columns are aligned with spaces.)
NUM_FILES = ("AIOGeist-Regular.ttf", "AIOGeist-Medium.ttf", "AIOGeist-Bold.ttf")


def load_fonts(app: QApplication | None = None):
    """Register the bundled Inter (UI) and AIO Geist (numbers, file names, key hints) once and make Inter the
    app font (13 px at 100 %, like the mock-up)."""
    global UI_FAMILY, MONO_FAMILY
    if _FONTS["done"]:
        return
    _FONTS["done"] = True
    from PyQt6.QtGui import QFont, QFontDatabase
    d = Path(__file__).resolve().parent.parent / "fonts"
    fams = {}
    for f in ("Inter-Regular.otf", "Inter-Medium.otf", "Inter-SemiBold.otf", "Inter-Bold.otf") + NUM_FILES:
        if (d / f).is_file():
            i = QFontDatabase.addApplicationFont(str(d / f))
            for fam in (QFontDatabase.applicationFontFamilies(i) if i != -1 else []):
                fams[f] = fam
    UI_FAMILY = fams.get("Inter-Regular.otf", UI_FAMILY)
    MONO_FAMILY = fams.get("AIOGeist-Regular.ttf", MONO_FAMILY)
    app = app or QApplication.instance()
    if app is not None and "Inter-Regular.otf" in fams:
        f = QFont(UI_FAMILY)
        f.setPointSizeF(9.75)                        # 13 px at 96 dpi
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        app.setFont(f)


# ---------------------------------------------------------------------------------------------------- tooltips
# 3.0.1: tooltips in the app's colours (light / dark), compact, laid out like a professional app:
#     Title  Ctrl+O            (the name in semi-bold, the shortcut muted in the mono font)
#     what it does             (muted, wraps when long)
# Done once for the whole app by formatting the text when a tooltip is SET (QWidget / QAction.setToolTip); reading it
# back (toolTip()) still returns the plain text, so code and searches never see the markup.
import re as _re

_KEY = (r"(?:(?:Ctrl|Alt|Shift|Meta|Cmd|Win|⌘|⌥|⇧)\+)*(?:F\d{1,2}|Esc|Space|Del|Delete|Tab|Enter|Return|Home|End|"
        r"PgUp|PgDn|Page ?Up|Page ?Down|Backspace|Ins|Insert|[A-Z0-9]|[,./=\-\[\]`;'+]|←|→|↑|↓|Wheel|Click)")
_KEYS_RE = _re.compile(r"\s*\((" + _KEY + r"(?:\s*(?:/|,|or|·)\s*" + _KEY + r")*)\)")
_PLAIN: dict = {}
_MUTED = "#7A838E"               # readable on the light and the dark tooltip


def format_tip(text: str) -> str:
    if not text or len(text) > 700 or text.lstrip().startswith("<") or "\t" in text:
        return text
    from html import escape
    lines = text.split("\n")
    head, rest = lines[0].strip(), [x for x in lines[1:]]
    m = _KEYS_RE.search(head)
    keys = ""
    if m is not None:
        keys = m.group(1)
        head = (head[:m.start()] + head[m.end():]).strip()
    title, _, desc = head.partition(" — ")
    title, desc = title.strip(), desc.strip()
    if len(title) > 44 and ": " in title and not desc:            # "Command palette: run any tool …"
        title, _, desc = title.partition(": ")
        desc = desc[:1].upper() + desc[1:]
    if not keys and not desc and not any(x.strip() for x in rest):
        return text                                  # a plain one-liner stays plain (styled by the style sheet)
    if len(title) > 64 and not keys:
        return text
    nowrap = "white-space:pre" if len(title) <= 56 else ""
    out = (f"<p style='margin:0; {nowrap}'><span style='font-weight:600'>{escape(title)}</span>"
           + (f"&nbsp;&nbsp;<span style='color:{_MUTED}; font-family:\"{MONO_FAMILY}\"; font-size:11px'>"
              f"{escape(keys)}</span>" if keys else "") + "</p>")
    body = [desc] if desc else []
    body += rest
    body = "\n".join(body).strip("\n")
    if body:
        out += (f"<p style='margin:3px 0 0 0; color:{_MUTED}'>" + escape(body).replace("\n", "<br>") + "</p>")
    if len(_PLAIN) > 6000:
        for k in list(_PLAIN)[:2000]:
            _PLAIN.pop(k, None)
    _PLAIN[out] = text
    return out


class _RoundTips:
    """Tooltips with really round corners: the tooltip window gets a see-through background (set while it is being
    polished — before it has a native window), so the style sheet's rounded border is all that shows. Without it the
    corners outside the radius stay filled: a grey round outline on a square box."""
    obj = None

    @classmethod
    def install(cls):
        from PyQt6.QtCore import QEvent, QObject, Qt
        if cls.obj is not None or QApplication.instance() is None:
            return

        class F(QObject):
            def eventFilter(self, o, e):
                if e.type() == QEvent.Type.Polish and o.isWidgetType() \
                        and o.metaObject().className() == "QTipLabel" and not o.property("aio_round"):
                    o.setProperty("aio_round", True)
                    o.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                    o.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
                return False
        cls.obj = F(QApplication.instance())
        QApplication.instance().installEventFilter(cls.obj)


def install_tooltips():
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QWidget
    _RoundTips.install()
    if getattr(QWidget, "_aio_tips", False):
        return
    for cls in (QWidget, QAction):
        set_, get_ = cls.setToolTip, cls.toolTip

        def setter(self, text, _set=set_):
            _set(self, format_tip(text or ""))

        def getter(self, _get=get_):
            cur = _get(self)
            return _PLAIN.get(cur, cur)
        cls.setToolTip = setter
        cls.toolTip = getter
    QWidget._aio_tips = True


def mono(px: float | None = None, weight=None):
    """The font for numbers, file names and key hints (AIO Geist: every digit the same width, slashed zero)."""
    from PyQt6.QtGui import QFont
    load_fonts()
    f = QFont(MONO_FAMILY)
    f.setStyleHint(QFont.StyleHint.SansSerif)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)     # like Inter: the designed shapes
    if px:
        f.setPixelSize(int(round(px)))
    if weight is not None:
        f.setWeight(weight)
    return f


def apply(app: QApplication | None = None):
    """Palette + style sheet for the current appearance; icons and @role styles follow."""
    app = app or QApplication.instance()
    if app is None:
        return
    load_fonts(app)
    install_tooltips()
    try:
        from AIO_QC.widgets.window_frame import install_tool_window_chrome
        install_tool_window_chrome()                 # feature windows: merged title bar (all three modes)
        from AIO_QC.widgets import window_shelf
        window_shelf.install(app)                    # minimise to the shelf; attention when a job ends unseen
    except Exception as _exc:
        _ignored(_exc)
    dark = want_dark()
    t = tokens(dark)
    _ACTIVE["dark"] = dark
    app.setPalette(palette(t))
    try:
        app.setStyleSheet(stylesheet(t))
    except Exception as _exc:                                    # never block start-up on a style problem
        _ignored(_exc)
        app.setStyleSheet("")
    try:
        from AIO_QC.utils.qt_utils import refresh_icons
        refresh_icons()                                  # theme ink + @role style sheets
    except Exception as _exc:
        _ignored(_exc)


def set_mode(m: str):
    _settings().setValue(SETTING, m if m in ("system", "light", "dark") else "system")
    apply()
    app = QApplication.instance()
    sh = getattr(app, "shell", None) if app else None
    for w in (getattr(sh, "image", None), getattr(sh, "video", None), getattr(sh, "studio", None)):
        fn = getattr(w, "_on_theme_changed", None) or getattr(w, "refresh_theme", None)
        if fn is not None:
            try:
                fn()
            except Exception as _exc:
                _ignored(_exc)


def watch(app: QApplication):
    """Follow the system's light / dark switch while Appearance is "System"."""
    hints = QGuiApplication.styleHints()
    if hasattr(hints, "colorSchemeChanged") and not getattr(app, "_aio_design_watch", False):
        app._aio_design_watch = True
        hints.colorSchemeChanged.connect(lambda _s: set_mode(mode()) if mode() == "system" else None)


def appearance_setting():
    from AIO_QC.widgets.settings_dialog import Setting
    return Setting("app.appearance", "Appearance", "choice", get=mode, set=set_mode, default="system",
                   choices=[("system", "Follow the system"), ("light", "Light"), ("dark", "Dark")],
                   description="Light or dark look for every mode; “Follow the system” switches with Windows / "
                               "macOS.", aliases=("theme", "dark mode", "light mode", "colors", "colours"))
