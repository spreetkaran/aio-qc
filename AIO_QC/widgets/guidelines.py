# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Labelling guidelines in the app (core.guidelines):

GuideCard           compact card — the rule text and ✓ / ✗ example crops. Floats over the image (bottom-left) when a
                    shape whose class has a guideline is selected, and sits above the results of a QC tool that has a
                    guideline for its rule. Click an example to see it large; "Edit" opens the Guidelines window.
GuidelinesDialog    Labels ▸ Guidelines (Ctrl+Alt+G): every class, every Relationship Rule (one guideline per rule,
                    shown above that rule's results) and the checks that find labelling problems — write the rule,
                    add the selected shape as a good / bad example, notes per example, export one HTML page for
                    annotators (also put into every hand-off batch).
add_selected(app, kind)   the selected shape on the image -> example of its class (annotation menu, G / Shift+G).
shape_example(...)        the one way a shape becomes an example (box from its real extent, its outline drawn).
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QPlainTextEdit, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style
from . import anim

ROLE = Qt.ItemDataRole.UserRole


def _pix(path, side) -> QPixmap:
    pm = QPixmap(str(path))
    if pm.isNull():
        return pm
    return pm.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


# QC tools whose findings are labelling mistakes an annotator can learn from (the others — stats, exports,
# reports, review tools … — have nothing to explain). Relationship Rules get one guideline per rule instead.
GUIDE_TOOLS = ("validation", "empty", "ambiguity", "text_labels", "duplicates", "outliers", "sequence", "dim_area",
               "train_size", "ocr_check", "similarity", "model_check")


def scope_title(scope: str, tools: dict | None = None, titles: dict | None = None, store=None) -> str:
    if titles and scope in titles:
        return titles[scope]
    if scope.startswith("tool:"):
        k = scope[5:]
        return f"Check · {(tools or {}).get(k, k)}"
    if scope.startswith("rule:") and store is not None:
        return store.title(scope)
    return scope


def shape_example(store, scope: str, kind: str, shape: dict, image_path, note: str = "", src: str = ""):
    """A shape -> a ✓ / ✗ example crop of `scope` (its real extent — circles by their radius — and its outline)."""
    pts = shape.get("points") or []
    if not pts:
        return None
    st = shape.get("shape_type") or "polygon"
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    poly = None
    if st == "circle" and len(pts) == 2:
        import math
        r = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
        poly = [(xs[0] + r * math.cos(a * math.pi / 16), ys[0] + r * math.sin(a * math.pi / 16)) for a in range(32)]
        xs, ys = [xs[0] - r, xs[0] + r], [ys[0] - r, ys[0] + r]
    elif st == "polygon" and len(pts) >= 3:
        poly = list(zip(xs, ys))
    elif st in ("line", "linestrip") and len(pts) >= 2:
        poly = list(zip(xs, ys))
    return store.add_example(scope, kind, image_path, (min(xs), min(ys), max(xs), max(ys)), note, src=src,
                             poly=poly)


# ====================================================================================================== card
class GuideCard(QFrame):
    editRequested = pyqtSignal(str)

    def __init__(self, parent=None, compact=True):
        super().__init__(parent)
        self.scope = None
        self.compact = compact
        self.setObjectName("guideCard")
        apply_style(self, "QFrame#guideCard{background:@surface; border:1px solid @line; border-radius:8px;}")
        self.title = QLabel("")
        apply_style(self.title, "font-weight:600;")
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.PlainText)
        apply_style(self.text, "color:@muted;")
        self.row = QHBoxLayout()
        self.row.setSpacing(4)
        edit = QToolButton()
        edit.setText("Edit")
        edit.setAutoRaise(True)
        edit.clicked.connect(lambda: self.editRequested.emit(self.scope or ""))
        close = QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.setToolTip("Hide (Settings ▸ Interface ▸ Guideline cards turns them off)")
        close.clicked.connect(lambda: anim.fade(self, False))
        head = QHBoxLayout()
        head.addWidget(QLabel("📘"))
        head.addWidget(self.title, 1)
        head.addWidget(edit)
        if compact:
            head.addWidget(close)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(4)
        lay.addLayout(head)
        lay.addWidget(self.text)
        lay.addLayout(self.row)
        self.hide()

    def set_scope(self, store, scope: str | None, tools=None, titles=None) -> bool:
        g = store.get(scope) if (store is not None and scope) else None
        self.scope = scope
        while self.row.count():
            it = self.row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if g is None:
            return False
        self.title.setText(scope_title(scope, tools, titles, store))
        t = g.get("text") or ""
        lines = t.splitlines()
        short = "\n".join(lines[:4]) + (" …" if len(lines) > 4 else "") if self.compact else t
        self.text.setText(short)
        self.text.setToolTip(t)
        self.text.setVisible(bool(t))
        side = 64 if self.compact else 96
        shown = 0
        for kind in ("good", "bad"):
            for ex in [e for e in g["examples"] if e["kind"] == kind][: (3 if self.compact else 8)]:
                lbl = QLabel()
                pm = _pix(store.image(ex), side)
                if pm.isNull():
                    continue
                lbl.setPixmap(pm)
                col = hexc("ok") if kind == "good" else hexc("error")
                lbl.setStyleSheet(f"border:2px solid {col}; border-radius:3px;")
                lbl.setToolTip(f"{'✓ Do' if kind == 'good' else '✗ Don’t'}: {ex.get('note') or ''}")
                lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                lbl.mousePressEvent = (lambda _e, p=store.image(ex), n=ex.get("note", ""), k=kind:
                                       _show_large(self, p, n, k))
                self.row.addWidget(lbl)
                shown += 1
        self.row.addStretch(1)
        self.adjustSize()
        return True


def _show_large(parent, path, note, kind):
    d = QDialog(parent.window())
    d.setWindowTitle(("✓ Good example" if kind == "good" else "✗ Bad example") + (f" — {note}" if note else ""))
    lbl = QLabel()
    lbl.setPixmap(_pix(path, 520))
    lay = QVBoxLayout(d)
    lay.addWidget(lbl)
    if note:
        n = QLabel(note)
        n.setWordWrap(True)
        lay.addWidget(n)
    d.show()
    anim.window_fade_in(d)


class CanvasGuide:
    """Keeps a GuideCard over the image for the selected shape's class (debounced, faded)."""

    def __init__(self, app):
        self.app = app
        self.card = GuideCard(app.canvas)
        self.card.setMaximumWidth(360)
        self.card.editRequested.connect(lambda sc: app.open_guidelines(sc))
        self.enabled = app.settings.value("ui/guide_cards", True, type=bool)
        self._t = QTimer(app, singleShot=True, interval=120, timeout=self._update)
        app.canvas.shapeSelected.connect(lambda _i: self._t.start())
        app.canvas.selectionCleared.connect(lambda: self._t.start())

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.app.settings.setValue("ui/guide_cards", self.enabled)
        self._update()

    def _update(self):
        app, c = self.app, self.app.canvas
        g = c.selected_group
        s = app.session
        scope = str(g.shape_data.get("label")) if g is not None else None
        if not self.enabled or s is None or scope is None or not self.card.set_scope(s.guides, scope):
            if self.card.isVisible():
                anim.fade(self.card, False, 140)
            return
        self.card.adjustSize()
        vp = c.viewport().geometry()
        self.card.move(vp.left() + 10, vp.bottom() - self.card.height() - 10)
        if not self.card.isVisible():
            anim.fade(self.card, True, 180)
        self.card.raise_()


def add_selected(app, kind: str):
    """The shape selected on the image -> a good / bad example of its class."""
    c, s = app.canvas, app.session
    g = c.selected_group
    if s is None or g is None or app.current_ann is None:
        app.statusBar().showMessage("Select a shape on the image first", 4000)
        return
    if not (g.shape_data.get("points") or []):
        return
    label = str(g.shape_data.get("label"))
    smp = s.by_key.get(app.current_key)
    img = app.current_ann.image_path or (smp.image_path if smp else None)
    if img is None or not Path(img).is_file():
        app.statusBar().showMessage("✋ This image has no image file (embedded data only) — cannot make an example", 6000)
        return
    note, ok = QInputDialog.getText(app, f"{'Good' if kind == 'good' else 'Bad'} example of '{label}'",
                                    "Note (what makes it right / wrong) — optional:")
    if not ok:
        return
    eid = shape_example(s.guides, label, kind, g.shape_data, img, note, src=s.history.rel(app.current_key))
    if eid is None:
        app.statusBar().showMessage("✋ Could not read the image", 5000)
        return
    app.statusBar().showMessage(f"📘 Added a {'✓ good' if kind == 'good' else '✗ bad'} example of '{label}' "
                                "(Labels ▸ Guidelines)", 5000)
    if getattr(app, "canvas_guide", None) is not None:
        app.canvas_guide._update()


# ====================================================================================================== dialog
class GuidelinesDialog(QDialog):
    TITLE = "Guidelines"
    SUPPORTED = None

    def __init__(self, app, scope=None):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.store = self.session.guides
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-guidelines"))
        self.resize(940, 620)
        self.scopes = QListWidget()
        self.scopes.currentItemChanged.connect(lambda *_: self._show())
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("The rule, in a few lines: what to include, what not, edge cases …")
        self._save_t = QTimer(self, singleShot=True, interval=600, timeout=self._save_text)
        self.text.textChanged.connect(self._save_t.start)
        self.examples = QListWidget()
        self.examples.setViewMode(QListWidget.ViewMode.IconMode)
        self.examples.setIconSize(QSize(150, 150))
        self.examples.setGridSize(QSize(170, 200))
        self.examples.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.examples.setMovement(QListWidget.Movement.Static)
        self.examples.setWordWrap(True)
        self.examples.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.examples.customContextMenuRequested.connect(self._ex_menu)
        self.examples.itemDoubleClicked.connect(self._open_ex)
        self.add_good = newButton("✓ Add selected shape as good", slot=lambda: self._add("good"),
                                  tip="The shape selected on the image becomes a good example (G)")
        self.add_bad = newButton("✗ Add selected shape as bad", slot=lambda: self._add("bad"),
                                 tip="The shape selected on the image becomes a bad example (Shift+G)")
        export = newButton("Export for annotators…", icon="export", slot=self._export,
                           tip="One HTML page with every rule and its examples (images embedded)")
        self.header = QLabel("")
        apply_style(self.header, "font-size:13pt; font-weight:600;")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.header)
        rl.addWidget(QLabel("Rule"))
        rl.addWidget(self.text, 1)
        rl.addWidget(QLabel("Examples — double-click to enlarge, right-click to edit the note, switch good/bad or "
                            "remove"))
        rl.addWidget(self.examples, 3)
        b = QHBoxLayout()
        b.addWidget(self.add_good)
        b.addWidget(self.add_bad)
        b.addStretch(1)
        rl.addLayout(b)
        split = QSplitter()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Classes · relationship rules · checks · ● = has a guideline")
        apply_style(hint, "color:@muted;")
        ll.addWidget(hint)
        ll.addWidget(self.scopes, 1)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([260, 680])
        foot = QHBoxLayout()
        foot.addWidget(export)
        foot.addStretch(1)
        foot.addWidget(newButton("Close", slot=self.close))
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(foot)
        self._fill(scope)

    def _tools(self) -> dict:
        from AIO_QC.app import QC_TOOLS
        return {t.key: t.title for t in QC_TOOLS if t.dialog is not None}

    def _rules(self) -> list:
        from AIO_QC.core import rules as RL
        from AIO_QC.core.state import state_dir
        return RL.load_rules(state_dir(self.session.root) / "rules.json")

    def _titles(self) -> dict:
        from AIO_QC.core import rules as RL
        return {f"rule:{r['id']}": f"Rule · {RL.describe(r)}" for r in self._rules()}

    def _fill(self, want=None):
        s = self.session
        names = list(s.true_classes or []) or sorted(s.label_totals(), key=str.lower)
        tools, titles = self._tools(), self._titles()
        items = [(n, n) for n in names]
        items += [(sc, title) for sc, title in titles.items()]                 # one per relationship rule
        items += [(f"tool:{k}", scope_title(f"tool:{k}", tools)) for k in GUIDE_TOOLS if k in tools]
        have = dict(items)
        items += [(sc, scope_title(sc, tools, titles, self.store)) for sc in self.store.scopes() if sc not in have]
        cur = want or (self.scopes.currentItem().data(ROLE) if self.scopes.currentItem() else None)
        self.scopes.blockSignals(True)
        self.scopes.clear()
        for sc, title in items:
            it = QListWidgetItem(("● " if self.store.has(sc) else "    ") + title)
            it.setData(ROLE, sc)
            self.scopes.addItem(it)
            if sc == cur:
                self.scopes.setCurrentItem(it)
        self.scopes.blockSignals(False)
        if self.scopes.currentItem() is None and self.scopes.count():
            self.scopes.setCurrentRow(0)
        self._show()

    def select(self, scope):
        self._fill(scope)

    def scope(self):
        it = self.scopes.currentItem()
        return it.data(ROLE) if it else None

    def _show(self):
        sc = self.scope()
        self.header.setText(scope_title(sc, self._tools(), self._titles(), self.store) if sc else "")
        g = self.store.get(sc) if sc else None
        self.text.blockSignals(True)
        self.text.setPlainText((g or {}).get("text", ""))
        self.text.blockSignals(False)
        self.examples.clear()
        for ex in (g or {}).get("examples", []):
            it = QListWidgetItem(QIcon(_pix(self.store.image(ex), 150)),
                                 ("✓ " if ex["kind"] == "good" else "✗ ") + (ex.get("note") or ""))
            it.setData(ROLE, ex)
            it.setToolTip(f"{ex.get('src', '')}\n{time.strftime('%Y-%m-%d', time.localtime(ex.get('ts', 0)))}")
            self.examples.addItem(it)
        is_class = bool(sc) and not sc.startswith(("tool:", "rule:"))
        for b in (self.add_good, self.add_bad):
            b.setEnabled(bool(sc))
            b.setToolTip(b.toolTip() if is_class else "Select a shape on the image: it becomes an example of this rule")

    def _save_text(self):
        sc = self.scope()
        if sc:
            had = self.store.has(sc)
            self.store.set_text(sc, self.text.toPlainText())
            if had != self.store.has(sc):
                self._mark()
            self._card_refresh()

    def _mark(self):
        it = self.scopes.currentItem()
        if it:
            sc = it.data(ROLE)
            it.setText(("● " if self.store.has(sc) else "    ") + scope_title(sc, self._tools(), self._titles(),
                                                                               self.store))

    def _card_refresh(self):
        cg = getattr(self.app, "canvas_guide", None)
        if cg is not None:
            cg._update()

    def _add(self, kind):
        sc = self.scope()
        app = self.app
        g = app.canvas.selected_group
        if sc is None:
            return
        if g is None:
            QMessageBox.information(self, self.TITLE, "Select a shape on the image first (Select mode, click it).")
            return
        label = str(g.shape_data.get("label"))
        if not sc.startswith(("tool:", "rule:")) and label != sc:
            if QMessageBox.question(self, self.TITLE, f"The selected shape is '{label}', not '{sc}'. Add it as a "
                                    f"{kind} example of '{sc}' anyway?") != QMessageBox.StandardButton.Yes:
                return
        pts = g.shape_data.get("points") or []
        smp = self.session.by_key.get(app.current_key)
        img = (app.current_ann.image_path if app.current_ann else None) or (smp.image_path if smp else None)
        if not pts or img is None or not Path(img).is_file():
            QMessageBox.information(self, self.TITLE, "This image has no readable image file.")
            return
        note, ok = QInputDialog.getText(self, "Example note", "What makes it right / wrong (optional):")
        if not ok:
            return
        shape_example(self.store, sc, kind, g.shape_data, img, note, src=self.session.history.rel(app.current_key))
        self._show()
        self._mark()
        self._card_refresh()

    def _ex_menu(self, pos):
        it = self.examples.itemAt(pos)
        sc = self.scope()
        if it is None or sc is None:
            return
        ex = it.data(ROLE)
        m = QMenu(self)
        m.addAction("Edit note…", lambda: self._note(sc, ex))
        m.addAction("Make it a ✗ bad example" if ex["kind"] == "good" else "Make it a ✓ good example",
                    lambda: self._flip(sc, ex))
        if ex.get("src"):
            m.addAction("Open its image", lambda: self._open_src(ex))
        m.addSeparator()
        m.addAction("Remove", lambda: self._remove(sc, ex))
        m.exec(self.examples.mapToGlobal(pos))

    def _note(self, sc, ex):
        t, ok = QInputDialog.getText(self, "Example note", "Note:", text=ex.get("note", ""))
        if ok:
            self.store.update_example(sc, ex["id"], note=t)
            self._show()

    def _flip(self, sc, ex):
        self.store.update_example(sc, ex["id"], kind="bad" if ex["kind"] == "good" else "good")
        self._show()
        self._card_refresh()

    def _remove(self, sc, ex):
        self.store.remove_example(sc, ex["id"])
        self._show()
        self._mark()
        self._card_refresh()

    def _open_src(self, ex):
        k = str(self.session.root / ex["src"])
        key = next((x for x in (k, str(Path(k))) if x in self.session.by_key), None)
        if key:
            self.app.open_sample(key)

    def _open_ex(self, it):
        ex = it.data(ROLE)
        _show_large(self, self.store.image(ex), ex.get("note", ""), ex["kind"])

    def _export(self):
        if not self.store.scopes():
            QMessageBox.information(self, self.TITLE, "No guidelines yet.")
            return
        from AIO_QC.utils.save_dirs import ask_save
        f, _ = ask_save(self, "Export guidelines", f"{self.session.root.name}_guidelines.html", "HTML (*.html)",
                        kind="exports", fallback=self.session.root.parent)
        if f:
            tools, titles = self._tools(), self._titles()
            n = self.store.export_html(f, title=f"Labelling guidelines — {self.session.root.name}",
                                       titles={sc: scope_title(sc, tools, titles, self.store)
                                               for sc in self.store.scopes()})
            self.app.statusBar().showMessage(f"Exported {n} guideline(s) to {f}", 6000)

    def closeEvent(self, e):
        if self._save_t.isActive():
            self._save_t.stop()
            self._save_text()
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
