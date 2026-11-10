# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Help center: the User's Guide and the keyboard-shortcut reference in one searchable window.

  * chapters are HTML files in FrameExtractor/docs/guide (title / keywords in a comment on the first lines)
  * reference tables are generated from the code at display time - naming tokens, image formats, duplicate
    presets, mining criteria, command-line help, and the shortcuts as they are configured right now - so
    the guide can never drift from the app
  * full-text search over every chapter, with matches highlighted on the page
  * Back / Forward / Home, text size, "Open in browser" (one printable file with the whole guide)
"""
import html as _html
import os
import re
import tempfile

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QSize
from PyQt6.QtGui import QKeySequence, QShortcut, QTextCursor, QTextCharFormat, QColor, QDesktopServices, QPalette
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QTextBrowser,
    QSplitter, QToolButton, QLabel, QTextEdit,
)

from AIO_QC.widgets import motion as anim

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "guide")
SHORTCUTS_PAGE = "shortcuts"


# ------------------------------------------------------------------ content
TESTING_CHAPTERS = {"09-model-lab.html", "13-metrics.html"}
EXTRA_CHAPTERS = []          # [(name, title, keywords, html)] set by AIO QC (Image QC, Data Studio, modes) - first


def load_chapters():
    """[(name, title, keywords, raw html)]: AIO QC's chapters, then the Frame Extractor's (file order); the Model Lab
    chapters are Testing's now."""
    out = list(EXTRA_CHAPTERS)
    if not os.path.isdir(DOCS):
        return out
    for f in sorted(os.listdir(DOCS)):
        if not f.endswith(".html"):
            continue
        with open(os.path.join(DOCS, f), encoding="utf-8") as fh:
            raw = fh.read()
        title = re.search(r"<!--title:\s*(.*?)-->", raw)
        kw = re.search(r"<!--keywords:\s*(.*?)-->", raw)
        t = title.group(1).strip() if title else f
        t = (f"Testing · {t}" if f in TESTING_CHAPTERS else f"Frame Extractor · {t}")
        out.append((f, t, kw.group(1).strip() if kw else "", raw))
    return out


def _table(head, rows):
    h = "".join(f"<th>{_html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="t"><tr>{h}</tr>{b}</table>'


def _code(s):
    return f"<code>{_html.escape(str(s))}</code>"


def generated(name, shortcuts=None, mouse_help=None):
    """HTML for a <!--@name--> placeholder, built from the code (single source of truth)."""
    try:
        if name == "formats":
            from AIO_QC.tools.frame_extractor.core.imageio import FORMATS
            return _table(["Format", "Lossy", "Notes"],
                          [[f"<b>{_html.escape(lab)}</b> {_code(ext)}", "yes" if lossy else "no (lossless)", _html.escape(note)]
                           for ext, (lab, lossy, note) in FORMATS.items()])
        if name == "tokens":
            from AIO_QC.tools.frame_extractor.core import naming
            ctx = naming.NameContext(video="cam01", frame=2503, sec=83.45, fps=29.97, counter=17, kind="frame",
                                     reason="uncertain", roi=(100, 50, 740, 410), w=640, h=360)

            def example(k):
                try:
                    return ("{" + k + "}").format(**ctx.values()) or "20250502"
                except Exception:
                    return ""
            return _table(["Token", "Meaning", "Example"],
                          [[_code("{" + k + "}"), _html.escape(v), _code(example(k))] for k, v in naming.TOKENS.items()])
        if name == "naming_presets":
            from AIO_QC.tools.frame_extractor.core.naming import PRESETS
            return _table(["Preset", "Template"], [[_html.escape(k), _code(v)] for k, v in PRESETS.items()])
        if name == "dedup":
            from AIO_QC.tools.frame_extractor.core.dedup import PRESETS, PRESET_HELP
            rows = [["<b>exact</b>", "identical pixels", _html.escape(PRESET_HELP.get("exact", ""))]]
            rows += [[f"<b>{k}</b>", f"{g:g} % overall · {loc:g} % in any region", _html.escape(PRESET_HELP.get(k, ""))]
                     for k, (g, loc) in PRESETS.items()]
            return _table(["Preset", "A frame is a duplicate below", "Use it when"], rows)
        if name == "mining":
            from AIO_QC.tools.frame_extractor.core.detector import MINING_CRITERIA
            return _table(["Criterion", "Keeps frames with"], [[f"<b>{k}</b>", _html.escape(v)]
                                                               for k, v in MINING_CRITERIA.items()])
        if name == "cli":
            from AIO_QC.tools.frame_extractor.core.cli import build_parser
            p = build_parser()
            sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
            parts = []
            for cmd, sp in sub.choices.items():
                txt = sp.format_help()
                parts.append(f"<h3>{_html.escape(cmd)}</h3><pre>{_html.escape(txt)}</pre>")
            return "".join(parts)
        if name == "version":
            from AIO_QC.tools.frame_extractor import __version__
            return _html.escape(__version__)
        if name == "shortcuts":
            return shortcuts_html(shortcuts or [], mouse_help or [])
    except Exception as e:                 # never break the guide over a reference table
        return f"<p><i>(could not build this table: {_html.escape(str(e))})</i></p>"
    return ""


def key_html(seq):
    """'Ctrl+Shift+E' -> key caps (native names: ⌘ on macOS)."""
    if not seq:
        return '<span class="muted">—</span>'
    text = QKeySequence(seq).toString(QKeySequence.SequenceFormat.NativeText) or seq
    out = []
    for chord in text.split(", "):
        keys = re.split(r"(?<!^)\+(?!$)", chord) if "+" in chord and len(chord) > 1 else [chord]
        out.append("+".join(f'<span class="k">{_html.escape(k)}</span>' for k in keys))
    return " , ".join(out)


def shortcuts_html(entries, mouse_help):
    """entries: [(category, command, [key sequences])] as configured now; mouse_help: [(section, [(how, what)])]."""
    cats = {}
    for cat, name, seqs in entries:
        cats.setdefault(cat.split(" › ")[0], []).append((name, seqs))
    parts = ['<h1>Keyboard shortcuts</h1><p class="lead">Every command with its current shortcut. '
             '<a href="action:customize">Customize shortcuts…</a> · <a href="action:browser">Open as a printable '
             'cheat sheet</a></p>']
    for cat, rows in cats.items():
        rows = [(n, s) for n, s in rows if any(s)]
        if not rows:
            continue
        parts.append(f"<h2>{_html.escape(cat)}</h2>")
        parts.append(_table(["Command", "Shortcut"],
                            [[_html.escape(n), " &nbsp;or&nbsp; ".join(key_html(x) for x in s if x)] for n, s in rows]))
    if mouse_help:
        parts.append("<h1>Mouse, timeline &amp; position display</h1>")
        for section, rows in mouse_help:
            parts.append(f"<h2>{_html.escape(section)}</h2>")
            parts.append(_table(["Action", "Effect"], [[f"<b>{_html.escape(a)}</b>", _html.escape(b)] for a, b in rows]))
    return "".join(parts)


def css(dark):
    if dark:
        ink, ink2, muted, line, code, kbd, tip, note, link = ("#ecebe8", "#c3c2b7", "#8f8d86", "#3a3936", "#2a2a28",
                                                              "#34332f", "#14301f", "#3a2e10", "#79aef2")
    else:
        ink, ink2, muted, line, code, kbd, tip, note, link = ("#141413", "#45443f", "#87857f", "#dddbd3", "#f1f0ec",
                                                              "#f4f3ef", "#e7f4ec", "#fbf1dc", "#2a6fd0")
    return f"""
    body {{ color:{ink}; font-size:10.5pt; line-height:150%; }}
    h1 {{ font-size:20pt; font-weight:600; margin:4px 0 6px 0; }}
    h2 {{ font-size:13.5pt; font-weight:600; margin:18px 0 6px 0; }}
    h3 {{ font-size:11pt; font-weight:600; margin:14px 0 4px 0; color:{ink2}; }}
    p.lead {{ font-size:11.5pt; color:{ink2}; }}
    a {{ color:{link}; text-decoration:none; }}
    code {{ font-family:Consolas,'Cascadia Mono',monospace; background-color:{code}; }}
    pre {{ font-family:Consolas,'Cascadia Mono',monospace; font-size:9pt; background-color:{code}; padding:8px; }}
    table.t {{ border-collapse:collapse; margin:6px 0 10px 0; }}
    table.t th {{ text-align:left; font-weight:600; color:{ink2}; padding:5px 12px 5px 0; border-bottom:1px solid {line}; }}
    table.t td {{ padding:5px 12px 5px 0; border-bottom:1px solid {line}; vertical-align:top; }}
    table.cards td {{ padding:8px 14px 10px 0; vertical-align:top; width:50%; }}
    span.k {{ font-family:Consolas,'Cascadia Mono',monospace; font-size:9pt; background-color:{kbd};
              border:1px solid {line}; padding:0 4px; }}
    span.muted {{ color:{muted}; }}
    div.tip {{ background-color:{tip}; padding:8px 10px; margin:10px 0; }}
    div.note {{ background-color:{note}; padding:8px 10px; margin:10px 0; }}
    .muted {{ color:{muted}; }}
    .key {{ font-weight:600; }}
    .box {{ border:1px solid {line}; padding:8px; }}
    td {{ vertical-align:top; }}
    """


def _plain(html):
    t = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t)).strip()


# ------------------------------------------------------------------ window
class _HelpBrowser(QTextBrowser):
    """QTextBrowser that renders AIO QC's 'aio-icon:<name>' images (theme-aware icons, crisp at this scale)."""

    def __init__(self, hc):
        super().__init__()
        self._hc = hc

    def loadResource(self, typ, url):
        if url.scheme() == "aio-icon" and self._hc.icon_loader is not None:
            try:
                img = self._hc.icon_loader(url.path() or url.toString()[9:], self.devicePixelRatioF())
                if img is not None:
                    return img
            except Exception:
                pass
        return super().loadResource(typ, url)


class HelpCenter(QMainWindow):
    customize_shortcuts = pyqtSignal()

    def __init__(self, parent=None, shortcuts_getter=None, mouse_help=None, actions=None, icon_loader=None):
        super().__init__(parent)
        self.setWindowTitle("AIO QC — Help")
        self.actions = dict(actions or {})         # "action:<name>" links -> callables (set by the app)
        self.icon_loader = icon_loader             # "aio-icon:<name>" images in AIO QC's chapters
        self.setObjectName("HelpCenter")
        self.shortcuts_getter = shortcuts_getter or (lambda: [])
        self.mouse_help = mouse_help or []
        self.chapters = load_chapters()
        self._cache = {}
        self._hist, self._pos = [], -1
        self._zoom = 0
        self._query = ""

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        lay.addWidget(split)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 10, 6, 10)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the guide…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._search)
        self.search.returnPressed.connect(self._open_first_result)
        sl.addWidget(self.search)
        self.toc = QListWidget()
        self.toc.setFrameShape(QListWidget.Shape.NoFrame)
        self.toc.setSpacing(1)
        self.toc.itemClicked.connect(self._toc_clicked)
        self.toc.itemActivated.connect(self._toc_clicked)
        sl.addWidget(self.toc, 1)
        self.hits = QLabel("")
        self.hits.setEnabled(False)
        sl.addWidget(self.hits)
        split.addWidget(side)

        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 6, 10, 6)
        bar = QHBoxLayout()
        self.back = self._tool("◀", "Back (Alt+Left)", self.go_back)
        self.fwd = self._tool("▶", "Forward (Alt+Right)", self.go_forward)
        home = self._tool("⌂", "Contents", lambda: self.show_page(self.chapters[0][0] if self.chapters else SHORTCUTS_PAGE))
        bar.addWidget(self.back)
        bar.addWidget(self.fwd)
        bar.addWidget(home)
        bar.addStretch(1)
        bar.addWidget(self._tool("A−", "Smaller text (Ctrl+-)", lambda: self._zoom_by(-1)))
        bar.addWidget(self._tool("A+", "Larger text (Ctrl++)", lambda: self._zoom_by(1)))
        bar.addWidget(self._tool("Open in browser", "The whole guide as one printable page", self.open_in_browser))
        ml.addLayout(bar)
        self.view = _HelpBrowser(self)
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._link)
        self.view.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.view.document().setDocumentMargin(18)
        ml.addWidget(self.view, 1)
        split.addWidget(main)
        split.setSizes([260, 760])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(central)

        self._fill_toc()
        QShortcut(QKeySequence("Alt+Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Alt+Right"), self, activated=self.go_forward)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=lambda: (self.search.setFocus(), self.search.selectAll()))
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)
        QShortcut(QKeySequence.StandardKey.ZoomIn, self, activated=lambda: self._zoom_by(1))
        QShortcut(QKeySequence.StandardKey.ZoomOut, self, activated=lambda: self._zoom_by(-1))
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1000, 700)
        anim.fade_on_show(self, 180)
        self._update_nav()

    def _tool(self, text, tip, fn):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(fn)
        return b

    # ---- table of contents / search
    def _fill_toc(self, results=None):
        self.toc.clear()
        if results is None:
            it = QListWidgetItem("⌨  Keyboard shortcuts")
            it.setData(Qt.ItemDataRole.UserRole, SHORTCUTS_PAGE)
            self.toc.addItem(it)
            for i, (name, title, _, _) in enumerate(self.chapters, 1):
                it = QListWidgetItem(f"{i:>2}   {title}")
                it.setData(Qt.ItemDataRole.UserRole, name)
                it.setSizeHint(QSize(200, 28))
                self.toc.addItem(it)
            self.hits.setText("")
            return
        for name, title, snippet, n in results:
            it = QListWidgetItem(f"{title}  ({n})\n{snippet}")
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setToolTip(snippet)
            self.toc.addItem(it)
        self.hits.setText(f"{sum(r[3] for r in results)} matches in {len(results)} pages" if results else "No matches")

    def _page_text(self, name):
        if name == SHORTCUTS_PAGE:
            return "keyboard shortcuts " + _plain(self._render(SHORTCUTS_PAGE))
        title, kw, raw = next(((t, k, r) for n, t, k, r in self.chapters if n == name), ("", "", ""))
        return f"{title} {kw} " + _plain(self._render(name))

    def _search(self, text):
        q = text.strip().lower()
        self._query = q
        if len(q) < 2:
            self._fill_toc()
            self._highlight("")
            return
        words = q.split()
        results = []
        for name in [SHORTCUTS_PAGE] + [c[0] for c in self.chapters]:
            t = self._page_text(name)
            low = t.lower()
            if not all(w in low for w in words):
                continue
            n = sum(low.count(w) for w in words)
            i = low.find(words[0])
            snippet = ("…" if i > 40 else "") + t[max(0, i - 40):i + 70].strip() + "…"
            title = "Keyboard shortcuts" if name == SHORTCUTS_PAGE else next(c[1] for c in self.chapters if c[0] == name)
            kw = "keyboard shortcuts keys" if name == SHORTCUTS_PAGE else next(c[2] for c in self.chapters if c[0] == name)
            boost = (50 if any(w in title.lower() for w in words) else 0) + \
                (25 * sum(w in kw.lower().split() for w in words))
            results.append((name, title, snippet, n, n + boost))
        results.sort(key=lambda r: -r[4])
        self._fill_toc([r[:4] for r in results])

    def _open_first_result(self):
        if self.toc.count():
            self._toc_clicked(self.toc.item(0))

    def _toc_clicked(self, it):
        name = it.data(Qt.ItemDataRole.UserRole) if it else None
        if name:
            self.show_page(name)

    # ---- pages
    def _dark(self):
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _render(self, name):
        key = (name, self._dark())
        if key in self._cache and name != SHORTCUTS_PAGE:
            return self._cache[key]
        if name == SHORTCUTS_PAGE:
            body = generated("shortcuts", self.shortcuts_getter(), self.mouse_help)
        else:
            raw = next((r for n, _, _, r in self.chapters if n == name), "<p>Page not found.</p>")
            body = re.sub(r"<!--@(\w+)-->", lambda m: generated(m.group(1)), raw)
        page = f"<html><head><style>{css(self._dark())}</style></head><body>{body}</body></html>"
        self._cache[key] = page
        return page

    def show_page(self, name, record=True):
        if name != SHORTCUTS_PAGE and not any(c[0] == name for c in self.chapters):
            return
        old = self.view.grab() if self.view.isVisible() and self._hist else None
        cur = self._hist[self._pos] if 0 <= self._pos < len(self._hist) else None
        self.view.setHtml(self._render(name))          # text size (zoomIn) is a widget property: it persists
        if record and name != cur:
            self._hist = self._hist[:self._pos + 1] + [name]
            self._pos = len(self._hist) - 1
        self._select_toc(name)
        self._update_nav()
        if self._query:
            self._highlight(self._query)
        if old is not None and name != cur:
            anim.transition(self.view, old, 1, "y", 200)

    def _select_toc(self, name):
        for i in range(self.toc.count()):
            if self.toc.item(i).data(Qt.ItemDataRole.UserRole) == name:
                self.toc.setCurrentRow(i)
                return

    def _highlight(self, q):
        sels = []
        if q:
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#f2c94c") if not self._dark() else QColor("#7a5d00"))
            doc = self.view.document()
            for w in q.split():
                cur = QTextCursor(doc)
                while True:
                    cur = doc.find(w, cur)
                    if cur.isNull():
                        break
                    sel = QTextEdit.ExtraSelection()
                    sel.cursor, sel.format = cur, fmt
                    sels.append(sel)
                    if len(sels) > 500:
                        break
            if sels:
                self.view.setTextCursor(sels[0].cursor)
                self.view.ensureCursorVisible()
        self.view.setExtraSelections(sels)

    def _link(self, url):
        s = url.toString()
        if s.startswith("#"):                                  # in-page anchor (AIO QC's chapters)
            self.view.scrollToAnchor(s[1:])
            return
        if s.startswith("action:") and s[7:] in self.actions:
            self.actions[s[7:]]()
            return
        if s == "action:customize":
            self.customize_shortcuts.emit()
        elif s == "action:browser":
            self.open_in_browser(only=SHORTCUTS_PAGE)
        elif url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
        else:
            name = os.path.basename(url.path() or s).split("#")[0]
            self.show_page(name)

    def go_back(self):
        if self._pos > 0:
            self._pos -= 1
            self.show_page(self._hist[self._pos], record=False)

    def go_forward(self):
        if self._pos < len(self._hist) - 1:
            self._pos += 1
            self.show_page(self._hist[self._pos], record=False)

    def _update_nav(self):
        self.back.setEnabled(self._pos > 0)
        self.fwd.setEnabled(self._pos < len(self._hist) - 1)

    def _zoom_by(self, d):
        self._zoom = max(-3, min(8, self._zoom + d))
        self.view.zoomIn(d)

    def refresh(self):
        """Shortcuts or theme changed: rebuild what is cached."""
        self._cache.clear()
        if 0 <= self._pos < len(self._hist):
            self.show_page(self._hist[self._pos], record=False)

    # ---- export
    def open_in_browser(self, only=None):
        """The whole guide (or one page) as a single, printable HTML file in the default browser."""
        from AIO_QC.tools.frame_extractor import __version__
        names = [only] if only else [SHORTCUTS_PAGE] + [c[0] for c in self.chapters]
        bodies = []
        for n in names:
            page = self._render(n)
            body = re.search(r"<body>(.*)</body>", page, re.S).group(1)
            body = re.sub(r'<a href="action:[^"]*">(.*?)</a>', r"\1", body)
            body = re.sub(r'href="(\d\d-[\w-]+)\.html"', lambda m: f'href="#ch-{m.group(1)}"', body)
            bodies.append(f'<section id="ch-{n.replace(".html", "")}">{body}</section>')
        toc = "" if only else "<nav><h1>AIO QC " + _html.escape(__version__) + " — User's Guide</h1><ol>" + "".join(
            f'<li><a href="#ch-{c[0].replace(".html", "")}">{_html.escape(c[1])}</a></li>' for c in self.chapters) + \
            f'<li><a href="#ch-{SHORTCUTS_PAGE}">Keyboard shortcuts</a></li></ol></nav>'
        doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
               f"content='width=device-width,initial-scale=1'><title>AIO QC {_html.escape(__version__)} — "
               f"{'Keyboard shortcuts' if only else 'User’s Guide'}</title><style>{_BROWSER_CSS}</style></head>"
               f"<body><main>{toc}{''.join(bodies)}</main></body></html>")
        path = os.path.join(tempfile.gettempdir(), "AIO QC-shortcuts.html" if only else "AIO QC-guide.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        return path


_BROWSER_CSS = """
:root{color-scheme:light dark;--ink:#141413;--ink2:#45443f;--line:#dddbd3;--code:#f1f0ec;--tip:#e7f4ec;--note:#fbf1dc;
--link:#2a6fd0;--bg:#fcfcfb}
@media (prefers-color-scheme:dark){:root{--ink:#ecebe8;--ink2:#c3c2b7;--line:#3a3936;--code:#2a2a28;--tip:#14301f;
--note:#3a2e10;--link:#79aef2;--bg:#1a1a19}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:28px;margin:48px 0 8px}h2{font-size:19px;margin:28px 0 8px}h3{font-size:15px;color:var(--ink2)}
section{border-top:1px solid var(--line);padding-top:8px}nav ol{columns:2}
a{color:var(--link);text-decoration:none}p.lead{font-size:17px;color:var(--ink2)}
code,pre,span.k{font-family:ui-monospace,Consolas,monospace;background:var(--code)}
pre{padding:12px;overflow:auto;font-size:12.5px;border-radius:6px}
span.k{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:8px 0 14px}th,td{text-align:left;vertical-align:top;
padding:7px 12px 7px 0;border-bottom:1px solid var(--line)}th{color:var(--ink2);font-weight:600}
table.cards td{width:50%;border:0}div.tip,div.note{padding:10px 14px;border-radius:8px;margin:12px 0}
div.tip{background:var(--tip)}div.note{background:var(--note)}
@media (max-width:640px){nav ol{columns:1}main{padding:20px 16px 60px}}
@media print{body{background:#fff;color:#000}section{break-before:page}a{color:#000}}
"""
