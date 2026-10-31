# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🔖 Saved Views — named file queries with live counts and a plain-language explanation (see core.views).

Build a view from conditions (labels and counts, review state / note, split, folder, file name, Issues-Inbox
findings, unreadable files — or a fixed list of files), read back what it means in one sentence, and keep it: the
count next to each view follows every edit, mark and check result. Show a view in the file list, review it in the
Review Grid, or export it for a similar dataset.
"""
from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from AIO_QC.core.views import (
    OPS, REVIEW, TYPES, ViewStore, evaluate, explain, new_view, snapshot, snapshot_cond, validate,
)
from AIO_QC.utils import newButton, style as apply_style
from .base_scan import ispin
from .inference_service import FuncWorker

ID_ROLE = Qt.ItemDataRole.UserRole


class CondRow(QWidget):
    """One condition: type · its parameters · NOT · remove."""

    def __init__(self, dlg, cond: dict):
        super().__init__()
        self.dlg, self.c = dlg, dict(cond)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.type = QComboBox()
        for k, v in TYPES.items():
            self.type.addItem(v, k)
        self.type.setCurrentIndex(max(0, self.type.findData(self.c.get("t", "label"))))
        self.type.currentIndexChanged.connect(self._type_changed)
        self.neg = QCheckBox("not")
        self.neg.setToolTip("Invert this condition")
        self.neg.setChecked(bool(self.c.get("not")))
        self.neg.toggled.connect(lambda on: self._set("not", on))
        self.params = QWidget()
        self.pl = QHBoxLayout(self.params)
        self.pl.setContentsMargins(0, 0, 0, 0)
        rm = newButton("✕", slot=lambda: self.dlg.remove_cond(self), tip="Remove this condition")
        rm.setMaximumWidth(32)
        for w in (self.neg, self.type, self.params):
            self.lay.addWidget(w)
        self.lay.addWidget(rm)
        self._build()

    def _set(self, k, v):
        self.c[k] = v
        self.dlg.changed()

    def _type_changed(self, _i):
        t = self.type.currentData()
        self.c = {"t": t, "not": self.neg.isChecked()}
        self._build()
        self.dlg.changed()

    def _clear(self):
        while self.pl.count():
            w = self.pl.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _num(self):
        op = QComboBox()
        for k, v in OPS.items():
            op.addItem(v, k)
        op.setCurrentIndex(max(0, op.findData(self.c.setdefault("op", "ge"))))
        n = ispin(0, 1_000_000, int(self.c.setdefault("n", 1)))
        n2 = ispin(0, 1_000_000, int(self.c.setdefault("n2", self.c["n"])))
        n2.setVisible(self.c["op"] == "between")
        op.currentIndexChanged.connect(lambda _i: (self._set("op", op.currentData()),
                                                   n2.setVisible(op.currentData() == "between")))
        n.valueChanged.connect(lambda v: self._set("n", v))
        n2.valueChanged.connect(lambda v: self._set("n2", v))
        for w in (op, n, n2):
            self.pl.addWidget(w)

    def _text(self, key, placeholder):
        e = QLineEdit(self.c.get(key, ""))
        e.setPlaceholderText(placeholder)
        e.textChanged.connect(lambda t: self._set(key, t))
        self.pl.addWidget(e, 1)
        return e

    def _build(self):
        self._clear()
        t = self.c.get("t", "label")
        s = self.dlg.session
        if t == "label":
            lab = QComboBox()
            lab.setEditable(True)
            for name, _n in s.label_totals().most_common():
                lab.addItem(name)
            lab.setCurrentText(self.c.setdefault("label", lab.itemText(0) if lab.count() else ""))
            lab.currentTextChanged.connect(lambda v: self._set("label", v))
            self.pl.addWidget(lab, 1)
            self._num()
        elif t in ("total", "labels_n"):
            self._num()
        elif t == "review":
            states = set(self.c.setdefault("states", ["none"]))
            for k, v in REVIEW.items():
                cb = QCheckBox(v)
                cb.setChecked(k in states)
                cb.toggled.connect(lambda on, k=k: self._toggle_state(k, on))
                self.pl.addWidget(cb)
        elif t == "note":
            self._text("text", "e.g. blurry")
        elif t in ("split", "folder"):
            box = QComboBox()
            box.setEditable(True)
            box.addItems(s.splits if t == "split" else s.folders)
            box.setCurrentText(self.c.setdefault("value", box.itemText(0) if box.count() else ""))
            box.currentTextChanged.connect(lambda v: self._set("value", v))
            self.pl.addWidget(box, 1)
        elif t == "name":
            self._text("text", "text, or a regular expression")
            rx = QCheckBox("regex")
            rx.setChecked(bool(self.c.get("regex")))
            rx.toggled.connect(lambda on: self._set("regex", on))
            self.pl.addWidget(rx)
        elif t == "issue":
            tool = QComboBox()
            tool.addItem("any tool", "")
            iss = getattr(s, "issues", None)
            for k in sorted(getattr(iss, "by_tool", {}) or {}):
                tool.addItem(getattr(iss, "titles", {}).get(k, k), k)
            tool.setCurrentIndex(max(0, tool.findData(self.c.setdefault("tool", ""))))
            tool.currentIndexChanged.connect(lambda _i: self._set("tool", tool.currentData()))
            risk = ispin(0, 100, int(self.c.setdefault("risk", 0)))
            risk.setSpecialValueText("any risk")
            risk.setToolTip("Minimum risk score (Issues Inbox, 0–100)")
            risk.valueChanged.connect(lambda v: self._set("risk", v))
            self.pl.addWidget(tool, 1)
            self.pl.addWidget(QLabel("risk ≥"))
            self.pl.addWidget(risk)
        elif t == "keys":
            lab = QLabel(f"{len(self.c.get('rels') or []):,} files saved {self.c.get('when', '')}")
            apply_style(lab, "color:@muted;")
            self.pl.addWidget(lab, 1)

    def _toggle_state(self, k, on):
        st = set(self.c.get("states") or [])
        st.add(k) if on else st.discard(k)
        self._set("states", sorted(st))


class ViewsDialog(QDialog):
    TITLE, ICON = "Saved Views", "🔖"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.resize(int(app.width() * 0.7), int(app.height() * 0.7))
        self.store = ViewStore(self.session.root)
        self.counts: dict[str, int] = {}
        self.cur: dict | None = None
        self._snap = None
        self._worker = None
        self._dirty_counts = False

        self.list = QListWidget()
        self.list.currentItemChanged.connect(lambda *_: self._select())
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Views</b> — counts stay live"))
        ll.addWidget(self.list, 1)
        lb = QHBoxLayout()
        for text, slot, tip in (("＋ New", self._new, "New empty view"),
                                ("⧉ Copy", self._dup, "Duplicate the selected view"),
                                ("🗑", self._delete, "Delete the selected view")):
            lb.addWidget(newButton(text, slot=slot, tip=tip))
        ll.addLayout(lb)
        lb2 = QHBoxLayout()
        lb2.addWidget(newButton("📌 Current list", slot=self._snapshot,
                                tip="Save the files in the main file list right now as a fixed-list view"))
        lb2.addWidget(newButton("Export…", slot=self._export, tip="Save the selected view(s) to a .json file"))
        lb2.addWidget(newButton("Import…", slot=self._import, tip="Load views from a .json file (e.g. another dataset)"))
        ll.addLayout(lb2)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.name = QLineEdit()
        self.name.textChanged.connect(lambda t: self._edit("name", t))
        self.match = QComboBox()
        self.match.addItem("ALL conditions (and)", "all")
        self.match.addItem("ANY condition (or)", "any")
        self.match.currentIndexChanged.connect(lambda _i: self._edit("match", self.match.currentData()))
        self.desc = QPlainTextEdit()
        self.desc.setPlaceholderText("Why this view exists / what to do with its files (optional)")
        self.desc.setMaximumHeight(60)
        self.desc.textChanged.connect(lambda: self._edit("desc", self.desc.toPlainText()))
        f.addRow("Name:", self.name)
        f.addRow("Match:", self.match)
        f.addRow("Notes:", self.desc)
        rl.addLayout(f)
        self.conds_host = QWidget()
        self.conds_lay = QVBoxLayout(self.conds_host)
        self.conds_lay.setContentsMargins(0, 0, 0, 0)
        self.conds_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.conds_host)
        rl.addWidget(sc, 1)
        rl.addWidget(newButton("＋ Condition", slot=self._add_cond, tip="Add a condition"), 0,
                     Qt.AlignmentFlag.AlignLeft)
        self.explain = QLabel("")
        self.explain.setWordWrap(True)
        apply_style(self.explain, "QLabel{background:@surface; border:1px solid @line; border-radius:6px; "
                                  "padding:8px;}")
        rl.addWidget(self.explain)
        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        apply_style(self.problems, "color:@warn;")
        rl.addWidget(self.problems)
        acts = QHBoxLayout()
        self.count_lbl = QLabel("")
        acts.addWidget(self.count_lbl, 1)
        acts.addWidget(newButton("👁 Show in file list", slot=self._show, tip="Restrict the main file list to this view"))
        acts.addWidget(newButton("▦ Review in grid", slot=self._grid, tip="Open these files in the Review Grid"))
        rl.addLayout(acts)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        help_lbl = QLabel("A view is a question about your files, saved: e.g. “at least 3 'plate' and not reviewed "
                          "yet”. Conditions use the label index, review marks and check results already in memory, "
                          "so counts follow every change instantly — without re-reading files.")
        help_lbl.setWordWrap(True)
        apply_style(help_lbl, "color:@muted;")
        lay.addWidget(help_lbl)
        lay.addWidget(split, 1)

        self._count_timer = QTimer(self, singleShot=True, interval=700, timeout=self._recount)
        self._save_timer = QTimer(self, singleShot=True, interval=500, timeout=self.store.save)
        for sig in (self.session.indexChanged, self.session.reviewChanged, self.session.issuesChanged):
            sig.connect(self._data_changed)
        if not self.store.views:
            self._seed()
        self._fill(select=self.store.views[0]["id"] if self.store.views else None)
        self._recount()

    # ------------------------------------------------------------------ list
    def _seed(self):
        """Useful starting points, shown once for a dataset without views."""
        a = new_view("Not reviewed yet")
        a["conds"] = [{"t": "review", "states": ["none"]}]
        b = new_view("Flagged with a note")
        b["conds"] = [{"t": "review", "states": ["flag"]}]
        c = new_view("Empty files")
        c["conds"] = [{"t": "total", "op": "eq", "n": 0}]
        d = new_view("Has check findings, not reviewed")
        d["conds"] = [{"t": "issue", "tool": "", "risk": 0}, {"t": "review", "states": ["none"]}]
        self.store.views = [a, b, c, d]
        self.store.save()

    def _fill(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for v in self.store.views:
            it = QListWidgetItem(self._item_text(v))
            it.setData(ID_ROLE, v["id"])
            it.setToolTip(explain(v) + (f"\n\n{v['desc']}" if v.get("desc") else ""))
            self.list.addItem(it)
        self.list.blockSignals(False)
        for i in range(self.list.count()):
            if self.list.item(i).data(ID_ROLE) == select:
                self.list.setCurrentRow(i)
                break
        else:
            if self.list.count():
                self.list.setCurrentRow(0)

    def _item_text(self, v):
        n = self.counts.get(v["id"])
        return f"{v['name']}    " + ("…" if n is None else f"{n:,}")

    def _select(self):
        it = self.list.currentItem()
        self.cur = self.store.get(it.data(ID_ROLE)) if it else None
        self._load_editor()

    def _load_editor(self):
        v = self.cur
        for w in (self.name, self.match, self.desc):
            w.blockSignals(True)
        self.name.setText(v["name"] if v else "")
        self.match.setCurrentIndex(max(0, self.match.findData(v.get("match", "all") if v else "all")))
        self.desc.setPlainText(v.get("desc", "") if v else "")
        for w in (self.name, self.match, self.desc):
            w.blockSignals(False)
        while self.conds_lay.count() > 1:
            w = self.conds_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for c in (v.get("conds") if v else []) or []:
            self.conds_lay.insertWidget(self.conds_lay.count() - 1, CondRow(self, c))
        self._refresh_text()

    # ------------------------------------------------------------------ editing
    def _rows(self) -> list[CondRow]:
        return [self.conds_lay.itemAt(i).widget() for i in range(self.conds_lay.count() - 1)
                if isinstance(self.conds_lay.itemAt(i).widget(), CondRow)]

    def changed(self):
        if self.cur is None:
            return
        self.cur["conds"] = [dict(r.c) for r in self._rows()]
        self._after_edit()

    def _edit(self, key, value):
        if self.cur is not None:
            self.cur[key] = value
            self._after_edit()

    def _after_edit(self):
        self._save_timer.start()
        self._refresh_text()
        it = self.list.currentItem()
        if it is not None and self.cur is not None:
            it.setText(self._item_text(self.cur))
            it.setToolTip(explain(self.cur))
        self._count_timer.start()

    def _refresh_text(self):
        v = self.cur
        if v is None:
            self.explain.setText("")
            self.problems.setText("")
            self.count_lbl.setText("")
            return
        self.explain.setText("<b>Means:</b> " + explain(v))
        probs = validate(v, set(self.session.label_totals()), set(self.session.splits))
        self.problems.setText(("⚠ " + " · ".join(probs)) if probs else "")
        n = self.counts.get(v["id"])
        self.count_lbl.setText("" if n is None else f"<b>{n:,}</b> file(s) match right now")

    def _add_cond(self):
        if self.cur is None:
            self._new()
        self.conds_lay.insertWidget(self.conds_lay.count() - 1, CondRow(self, {"t": "label", "op": "ge", "n": 1}))
        self.changed()

    def remove_cond(self, row):
        row.setParent(None)
        row.deleteLater()
        self.changed()

    def _new(self):
        v = new_view()
        self.store.views.append(v)
        self.store.save()
        self._fill(v["id"])
        self.name.setFocus()
        self.name.selectAll()
        self._count_timer.start()

    def _dup(self):
        if self.cur is None:
            return
        v = copy.deepcopy(self.cur)
        v.update(new_view(self.cur["name"] + " (copy)"), conds=copy.deepcopy(self.cur.get("conds", [])),
                 match=self.cur.get("match", "all"), desc=self.cur.get("desc", ""))
        self.store.views.append(v)
        self.store.save()
        self._fill(v["id"])
        self._count_timer.start()

    def _delete(self):
        if self.cur is None:
            return
        if QMessageBox.question(self, self.TITLE, f"Delete the view “{self.cur['name']}”? (Files are not "
                                                  "touched.)") != QMessageBox.StandardButton.Yes:
            return
        self.store.views = [v for v in self.store.views if v["id"] != self.cur["id"]]
        self.store.save()
        self._fill()

    def _snapshot(self):
        keys = self.app.file_dock.visible_keys()
        if not keys:
            return
        v = new_view(f"Fixed list ({len(keys):,} files)")
        v["conds"] = [snapshot_cond(self.session, keys)]
        v["desc"] = "The files the main list showed when this view was saved — it does not change by itself."
        self.store.views.append(v)
        self.store.save()
        self._fill(v["id"])
        self._count_timer.start()

    def _export(self):
        if self.cur is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export view", f"{self.cur['name']}.aioqc-views.json",
                                              "Views (*.json)")
        if path:
            self.store.export({self.cur["id"]}, path)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import views", "", "Views (*.json)")
        if not path:
            return
        try:
            added = self.store.import_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.TITLE, f"Could not read the file: {exc}")
            return
        labels, splits = set(self.session.label_totals()), set(self.session.splits)
        probs = [f"“{v['name']}”: " + "; ".join(validate(v, labels, splits)) for v in added
                 if validate(v, labels, splits)]
        self._fill(added[0]["id"] if added else None)
        self._count_timer.start()
        if probs:
            QMessageBox.information(self, self.TITLE, "Imported. Some conditions refer to things this dataset does "
                                                      "not have (they match nothing until edited):\n\n" +
                                    "\n".join(probs))

    # ------------------------------------------------------------------ counting
    def _data_changed(self, *_):
        self._count_timer.start()

    def _samples(self):
        """(key, name, split, folder) for every file — rebuilt only when the file set changes."""
        n = len(self.session.samples)
        c = getattr(self, "_samples_cache", None)
        if c is None or c[0] != n or c[1] is not self.session.samples:
            c = self._samples_cache = (n, self.session.samples,
                                       [(s.key, s.name, s.split, s.folder) for s in self.session.samples])
        return c[2]

    def _recount(self):
        if self._worker is not None:
            self._dirty_counts = True
            return
        snap = snapshot(self.session, self._samples())
        views = copy.deepcopy(self.store.views)

        def run():
            return {v["id"]: evaluate(v, snap) for v in views}

        w = FuncWorker(run, self)
        self._worker = w

        def done(res):
            self._worker = None
            self._results = res
            self.counts = {k: len(v) for k, v in res.items()}
            for i in range(self.list.count()):
                it = self.list.item(i)
                v = self.store.get(it.data(ID_ROLE))
                if v is not None:
                    it.setText(self._item_text(v))
            self._refresh_text()
            if self._dirty_counts:
                self._dirty_counts = False
                self._count_timer.start()
        w.done.connect(done)
        w.failed.connect(lambda e: (setattr(self, "_worker", None), self.count_lbl.setText(f"⚠ {e}")))
        w.finished.connect(w.deleteLater)
        w.start()

    def keys_of(self, v) -> list[str]:
        res = getattr(self, "_results", {})
        if v["id"] in res and not self._count_timer.isActive() and self._worker is None:
            return res[v["id"]]
        return evaluate(v, snapshot(self.session, self._samples()))

    def _show(self):
        if self.cur is None:
            return
        keys = self.keys_of(self.cur)
        if not keys:
            QMessageBox.information(self, self.TITLE, "No file matches this view right now.")
            return
        self.app.apply_file_filter(keys, f"🔖 {self.cur['name']}")

    def _grid(self):
        if self.cur is None:
            return
        keys = self.keys_of(self.cur)
        if not keys:
            QMessageBox.information(self, self.TITLE, "No file matches this view right now.")
            return
        self.app.open_review_grid(keys, f"🔖 {self.cur['name']}")

    def closeEvent(self, e):
        self._save_timer.stop()
        self.store.save()
        for sig in (self.session.indexChanged, self.session.reviewChanged, self.session.issuesChanged):
            try:
                sig.disconnect(self._data_changed)
            except TypeError:
                pass
        if self._worker is not None:
            self._worker.wait(5000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
