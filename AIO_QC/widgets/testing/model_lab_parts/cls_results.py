# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — the results page for classifier runs (core.testing.cls_eval.ClsLabRun).

The same tabs as for detectors, each showing what a classifier is judged by: top-1 / top-5 accuracy and macro
precision / recall / F1 in the leaderboard; per-class F1 and the class-by-class confusion matrix; the mistakes as
pictures with the model's answer under them (surest first — often a wrong label); every image's answer; A vs B
(per-class recall difference, the images only one of them gets right); and on videos, class flips over time. The
operating threshold becomes a confidence below which an answer counts as "unsure". Images of a classification test
set can be given another class from the viewers (the run re-scores).

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab, used by ResultsMixin's methods when the run is a classifier's.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt

from AIO_QC.widgets.testing.model_lab_common import T_FAILURES, _item, _num_item, _release_thread_reader, \
    _swatch_icon, slot_hex

DET_BOARD = ["#", "", "Model", "mAP50-95", "mAP50", "Precision", "Recall", "F1", "Conf", "ms / img", "img / s",
             "False det.", "Missed"]
CLS_BOARD = ["#", "", "Model", "Top-1", "Top-5", "Precision", "Recall", "F1", "Answered", "ms / img", "img / s",
             "Right when sure", "Wrong"]
DET_CLASSES = ["Class", "Objects", "AP50", "AP50-95", "Precision", "Recall", "F1", "Correct", "False det.", "Missed"]
CLS_CLASSES = ["Class", "Images", "Precision", "Recall", "F1", "Right", "Taken for it", "Missed",
               "Mostly confused with", "Unsure"]
DET_KINDS = (("All errors", "all"), ("False detections", "fp"), ("Missed objects", "fn"), ("Wrong class", "cls"),
             ("Poor box", "loc"), ("Duplicates", "dup"), ("Detections on background", "bkg"),
             ("Missed (found below threshold)", "missed_low_conf"))
CLS_KINDS = (("All mistakes (surest first)", "all"), ("Sure mistakes — often a wrong label", "sure"),
             ("Unsure answers", "unsure"), ("Right only in its top 5", "top5"))
DET_SHOW = (("Predictions", "pred"), ("Predictions + ground truth", "gt"), ("Errors highlighted", "err"))
CLS_SHOW = (("Answers", "pred"), ("Answers + image's class", "gt"))
DET_SORT = (("Test order", "order"), ("Most detections", "most"), ("Fewest detections", "fewest"))
CLS_SORT = (("Test order", "order"), ("Wrong first", "wrong"), ("Least sure first", "unsure"),
            ("Surest first", "sure"))
DET_FAIL_HINT = ("Dashed = ground truth · model colour = correct · red = error · red dashed = missed. "
                 "Double-click to fix labels or view full size; right-click to compare models or save the image.")
CLS_FAIL_HINT = ("The model's answer is under each image: true class → answer and its confidence. Double-click to see "
                 "it full size and give the image another class (1-9, Enter); right-click to compare models.")
DET_PLOT_HINT = ("Each test image with the model's detections drawn (labels in Inter), at the operating threshold "
                 "above. Double-click to view full size; right-click or use the buttons to save.")
CLS_PLOT_HINT = ("Each test image with the model's answer under it (best class, confidence, runner-ups; grey = below "
                 "the confidence above). Double-click to view full size; right-click or the buttons to save.")


def _fill(combo, entries):
    cur = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for label, k in entries:
        combo.addItem(label, k)
    k = combo.findData(cur)
    combo.setCurrentIndex(k if k >= 0 else 0)
    combo.blockSignals(False)


class ClsResultsMixin:

    def _is_cls(self, r=None):
        r = self.run if r is None else r
        return getattr(r, "task", "") == "classify"

    def _cls_mode(self):
        """Relabel the results page for a classifier run (or back for a detector run)."""
        cls = self._is_cls()
        if getattr(self, "_res_mode_cls", None) == cls:
            return
        self._res_mode_cls = cls
        self.board.setHorizontalHeaderLabels(CLS_BOARD if cls else DET_BOARD)
        self.board.setToolTip(
            "Top-1: the model's best class is the image's class; top-5: it is among the five best. Precision / "
            "recall / F1: macro averages over the classes. Answered / right when sure: at the confidence above. "
            "Double-click a model for its mistakes." if cls else
            "P / R / F1 at the best-F1 confidence (YOLO val method). False detections and missed objects at the "
            "operating threshold. Click a header to sort; double-click a model for its failures.")
        heads = CLS_CLASSES if cls else DET_CLASSES
        self.pc_table.setColumnCount(len(heads))
        self.pc_table.setHorizontalHeaderLabels(heads)
        from PyQt6.QtWidgets import QComboBox
        _fill(self.f_kind, CLS_KINDS if cls else DET_KINDS)
        self.f_kind.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.f_kind.adjustSize()
        _fill(self.pl_show, CLS_SHOW if cls else DET_SHOW)
        if cls:
            self.pl_show.setCurrentIndex(1)
        _fill(self.pl_sort, CLS_SORT if cls else DET_SORT)
        self.f_zoom.setVisible(not cls)
        self.f_hint.setText(CLS_FAIL_HINT if cls else DET_FAIL_HINT)
        self.pl_hint.setText(CLS_PLOT_HINT if cls else DET_PLOT_HINT)
        self.res_best.setText("Answer every image" if cls else "Best F1 per model")
        self.res_best.setToolTip(
            "A classifier always has a best class. Untick and move the slider to treat answers below that confidence "
            "as \"unsure\": the leaderboard then shows how many images each model answers and how often those are "
            "right." if cls else
            "Operating threshold: each model at its best-F1 confidence, or one fixed value for all (move the "
            "slider). Errors, confusion matrices and galleries follow it.")
        self.tabs.setTabText(T_FAILURES, "Mistakes" if cls else "Failures")
        self.tabs.setTabText(T_FAILURES + 1, "Answers" if cls else "Plots")

    # ---- overview
    def _cls_overview_rows(self, r):
        slots = self._run_slots(r)
        for x in r.summary_table():
            k = self.board.rowCount()
            self.board.insertRow(k)
            self.board.setItem(k, 0, _num_item(x["rank"], "{:.0f}"))
            sw = _item("")
            sw.setIcon(_swatch_icon(slot_hex(slots[x["idx"]])))
            self.board.setItem(k, 1, sw)
            name = _item(x["model"], tip=r.specs[x["idx"]].path)
            name.setData(Qt.ItemDataRole.UserRole, x["idx"])
            self.board.setItem(k, 2, name)
            for c, (key, fmt) in enumerate((("top1", "{:.1%}"), ("top5", "{:.1%}"), ("precision", "{:.3f}"),
                                            ("recall", "{:.3f}"), ("f1", "{:.3f}"), ("answered", "{:.0%}"),
                                            ("speed_ms", "{:.1f}"), ("fps", "{:.0f}"),
                                            ("accuracy_answered", "{:.1%}"), ("wrong", "{:,.0f}")), 3):
                self.board.setItem(k, c, _num_item(x[key], fmt))

    # ---- per class
    def _cls_classes(self, r, res):
        unsure = {}
        for _i, g, p, _rt in res.per_image:
            if p and p[0][1] < res.op_conf:
                for t in g:
                    unsure[t] = unsure.get(t, 0) + 1
        for x in sorted(res.class_rows(), key=lambda x: x["f1"]):
            k = self.pc_table.rowCount()
            self.pc_table.insertRow(k)
            name = _item(x["name"])
            name.setData(Qt.ItemDataRole.UserRole, x["name"])
            self.pc_table.setItem(k, 0, name)
            for c, (key, fmt) in enumerate((("support", "{:,.0f}"), ("precision", "{:.3f}"), ("recall", "{:.3f}"),
                                            ("f1", "{:.3f}"), ("tp", "{:,.0f}"), ("fp", "{:,.0f}"),
                                            ("fn", "{:,.0f}")), 1):
                self.pc_table.setItem(k, c, _num_item(x[key], fmt))
            row = {p: n for p, n in (res.confusion.get(x["name"]) or {}).items() if p != x["name"]}
            if row:
                p, n = max(row.items(), key=lambda kv: kv[1])
                self.pc_table.setItem(k, 8, _item(f"{p} ({n:,})"))
            else:
                self.pc_table.setItem(k, 8, _item("–"))
            self.pc_table.setItem(k, 9, _num_item(unsure.get(x["name"], 0), "{:,.0f}"))
        note = ("Sorted weakest first. Taken for it: images of other classes the model called this; missed: its images "
                "called something else; unsure: its images answered below the confidence above "
                f"({res.op_conf:.2f}). Double-click a class for its mistakes.")
        un = res.extra.get("unmatched") or []
        if un:
            note += " Model classes the test set does not have (always wrong): " + ", ".join(un[:12])
        self.pc_note.setText(note)

    # ---- mistakes gallery
    @staticmethod
    def _cls_failure_rows(r, mi, kind, cls, cm):
        """[(score, item index, true names, [(name, conf)])], computed on the view thread."""
        if r is None or mi is None or r.results[mi] is None:
            return []
        res = r.results[mi]
        rows = []
        for i, g, p, right in res.per_image:
            if not p:
                continue
            top, c = p[0]
            sure = c >= res.op_conf
            if kind == "unsure":
                keep = not sure
            elif kind == "sure":
                keep = not right and sure and c >= max(0.5, res.op_conf)
            elif kind == "top5":
                keep = not right and any(n in g for n, _ in p[:5])
            else:
                keep = not right
            if not keep:
                continue
            if cm is not None and not (cm[0] in g and top == cm[1]):
                continue
            if cls not in (None, -1, "") and cls not in g and top != cls:
                continue
            rows.append((c if kind != "unsure" else 1 - c, r.eval_idx[i], g, p))
        rows.sort(key=lambda x: -x[0])
        return rows

    @staticmethod
    def _cls_row_text(g, p):
        if not p:
            return "no answer"
        return f"{', '.join(g)} → {p[0][0]} {p[0][1]:.0%}"

    # ---- answers (plots) tab
    @staticmethod
    def _cls_plot_rows(r, mi, cls, sort, conf):
        if r is None or mi is None or r.preds[mi] is None:
            return []
        order = list(r.eval_idx) if r.scored else list(range(len(r.items)))
        rows = []
        for ii in order:
            p = r.preds[mi][ii]
            if not p:
                continue
            g = list(r.items[ii].tags or [])
            if cls not in (None, -1, "") and cls not in g and p[0][0] != cls:
                continue
            wrong = bool(g) and p[0][0] not in g
            txt = f"{p[0][0]} {p[0][1]:.0%}" + (" · unsure" if p[0][1] < conf else "") + (" · wrong" if wrong else "")
            rows.append((ii, txt, p[0][1], wrong))
        if sort == "wrong":
            rows.sort(key=lambda x: (not x[3], -x[2]))
        elif sort == "unsure":
            rows.sort(key=lambda x: x[2])
        elif sort == "sure":
            rows.sort(key=lambda x: -x[2])
        return [(ii, txt) for ii, txt, _c, _w in rows]

    # ---- viewers and class changes
    def _cls_can_edit(self, ii):
        r = self.run
        ds = getattr(self.data, "dataset", None) if self.data is not None else None
        return (r is not None and r.ground == "labels" and ds is not None and getattr(ds, "cls_type", None) is not None
                and r.items[ii].meta.get("source") != "video")

    def _cls_set_class(self, ii, new):
        from AIO_QC.core.testing.dataset import set_item_class
        it = self.run.items[ii]
        old = list(it.tags or [])
        if self.qc is not None and self.qc.is_this(self.data):      # through Image QC: backups, history, Undo
            path = self.qc.set_class((it.label_path, it.path), new)
            if path:
                it.path = path
            it.tags = [new]
            if new not in self.data.dataset.names.values():
                self.data.dataset.names[max(self.data.dataset.names, default=-1) + 1] = new
        else:
            set_item_class(self.data.dataset, it, new)
        self._cls_follow_move(ii, it)
        self._log(f"{os.path.basename(it.path)}: class {', '.join(old) or '–'} → {new}")

    def _cls_follow_move(self, ii, it):
        """Class folders: the image moved, so its cache key changed — the cached answers move with it (a project
        reopened later still shows its results without running the models again)."""
        from AIO_QC.core.testing.runner import image_key
        d, cache = self.data, getattr(self, "cache", None)
        try:
            k = self.data.dataset.items.index(it)
        except ValueError:
            return
        if cache is None or k >= len(d.keys):
            return
        old = d.keys[k]
        if getattr(d, "qc", False):
            from AIO_QC.core.model_registry import qc_image_key
            new = qc_image_key(it.path, d.root)
        else:
            new = image_key(it.path, d.root)
        if new == old:
            return
        for s in self.run.specs:
            got = cache.get_many(s.key, [old]).get(old)
            if got is not None:
                cache.put_many(s.key, [(new, got[0], got[1])])
        d.keys[k] = new

    def _cls_open(self, items, start, models):
        from AIO_QC.widgets.testing.cls_view import ClsViewer
        if self.run is None or not items:
            return
        v = ClsViewer(self, self.run, items, models, start, self._plotter(), self._cls_can_edit, self._cls_set_class)
        changed = []
        v.labels_changed.connect(changed.extend)
        try:
            v.exec()
        finally:
            _release_thread_reader()
        if changed:
            self._labels_changed(sorted(set(changed)))

    # ---- compare
    def _cls_compare_build(self, r, a, b, rbld):
        from AIO_QC.core.testing import charts as C
        from AIO_QC.core.testing.report import extract_figures
        c = r.comparison(a, b)
        if not c:
            return None
        ra, rb_ = r.results[a], r.results[b]
        pc = sorted((x for x in c["per_class"] if x["a"] is not None or x["b"] is not None), key=lambda x: x["delta"])
        svg = C.diverging_bars([x["name"] for x in pc], [x["delta"] for x in pc], 560, rbld.ink,
                               "Recall difference per class", value_fmt=lambda v: f"{v:+.0%}",
                               labels=(f"{ra.name} better", f"{rb_.name} better"))
        figs = [("Per-class difference", f"Share of each class {rb_.name} gets right minus {ra.name}.", svg)]
        figs += extract_figures(rbld._confusion(a, ra) + rbld._confusion(b, rb_))
        return c, figs

    def _cls_compare_tiles(self, c, ra, rb_):
        from html import escape

        def d(v):
            col = "#199e70" if v > 1e-9 else ("#d03b3b" if v < -1e-9 else "inherit")
            return f"<b style='color:{col}'>{v * 100:+.1f} points</b>"
        sp = ""
        if ra.speed_ms and rb_.speed_ms:
            pct = (rb_.speed_ms - ra.speed_ms) / ra.speed_ms * 100
            sp = (f" · speed {rb_.speed_ms:.1f} vs {ra.speed_ms:.1f} ms "
                  f"(<b style='color:{'#199e70' if pct < 0 else '#d03b3b'}'>{pct:+.0f}%</b>)")
        return (f"<b>{escape(rb_.name)}</b> vs <b>{escape(ra.name)}</b>: top-1 {d(c['top1_delta'])} · top-5 "
                f"{d(c['map50_delta'])} · macro F1 {c['f1_delta']:+.3f}{sp} · right where the other is wrong on "
                f"<b>{len(c['b_better']):,}</b> images, wrong where the other is right on <b>{len(c['a_better']):,}</b>")

    def _cls_compare_reference(self, r, ref, m, rbld):
        """A classifier next to the reference model: how often it gives the reference's answer, per class, and
        the images where they differ."""
        from html import escape
        from AIO_QC.core.testing.report import extract_figures
        res = r.results[m]
        if res is None:
            self.cmp_grid.set_empty_text("Pick a model to compare with the reference.")
            self.cmp_grid.set_figures([])
            return
        epoch = self._cmp_epoch

        def build():
            figs = extract_figures(rbld._confusion(m, res, f"{res.name} vs reference {r.specs[ref].label}"))
            figs += extract_figures(rbld._per_class())
            differ = [(ii, p[0][1]) for ii, _g, p in r.mistakes(m)]
            unsure = sorted(((r.eval_idx[i], p[0][1]) for i, _g, p, rt in res.per_image
                             if rt and p and p[0][1] < 0.5), key=lambda x: x[1])
            return figs, differ, unsure

        def fill(out):
            if epoch != self._cmp_epoch:
                return
            figs, differ, unsure = out
            rn, mn = escape(r.specs[ref].label), escape(res.name)
            self.cmp_tiles.setText(f"<b>{mn}</b> gives the answer of reference <b>{rn}</b> on "
                                   f"<b>{res.top1:.1%}</b> of {len(r.eval_idx):,} images (top-5 {res.top5:.1%}) · "
                                   f"differs on <b>{len(differ):,}</b>")
            self.cmp_grid.set_figures(figs)
            self._cls_fill_compare_lists(r, ref, m, ((f"{res.name} answers differently from {r.specs[ref].label}",
                                                       differ),
                                                      (f"Same answer, but {res.name} is unsure", unsure)), epoch)
        self._bg("compare", build, fill)

    def _cls_fill_compare_lists(self, r, a, b, lists, epoch):
        from html import escape
        from PyQt6.QtCore import QSize, QThread
        from PyQt6.QtGui import QIcon
        from AIO_QC.widgets.testing.lab_common import Job, fit_pixmap
        from AIO_QC.core.diag import ignored as _ignored
        rbld = self._ink_builder(r)
        jobs = []
        for (lab, lst), gal, (title, rows) in zip(self.cmp_lists, self.cmp_galleries, lists):
            lab.setText(f"<b>{escape(title)}</b> ({len(rows):,}) - left {escape(r.specs[a].label)}, right "
                        f"{escape(r.specs[b].label)}. Double-click to look closer.")
            lst.setProperty("items", [ii for ii, _ in rows])
            gal.empty.setText("None.")
            gal.show_empty(not rows)
            for ii, c in rows[:24]:
                jobs.append((gal.add(self._loc_text(r.items[ii]) + f" · {c:.0%}", ii, r.items[ii].path), ii))

        def work(job):
            for it, ii in jobs:
                if job.cancelled() or self._cmp_epoch != epoch:
                    break
                try:
                    img = rbld.shot_image(ii, a, focus=False, compare=(a, b), tw=320)
                except Exception as _exc:                    # noqa: BLE001
                    _ignored(_exc)
                    img = None
                job.progress.emit((it, img))

        def prog(p):
            it, img = p
            if self._cmp_epoch == epoch and img is not None:
                it.setIcon(QIcon(fit_pixmap(img, it.listWidget().iconSize() if it.listWidget() else QSize(300, 112))))
        j = Job(work, self)
        j.progress.connect(prog)
        j.finished.connect(j.deleteLater)
        self._cmp_job = j
        j.start(QThread.Priority.LowPriority)
