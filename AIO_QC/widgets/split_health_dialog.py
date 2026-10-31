# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Dataset ▸ Split Health — are train / val / test fit to train and evaluate on? (core.split_health)

Health tab     per-split images / instances / empty images, per-class image counts per split (rare cells marked),
               drift of each split's class mix vs train, and every finding with why it matters.
Moves tab      suggested moves (leakage groups reunited, val / test topped up from train, classes missing from
               train brought in) — tick, Apply; the files move between the split folders in the dataset's own layout,
               review marks / findings follow, and Undo puts them back. CSV of the moves from the table toolbar.
"""
from __future__ import annotations

import html
import json
import time

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QSpinBox, QTabWidget, QTextBrowser

from AIO_QC.core import profile as P
from AIO_QC.core import split_health as SH
from AIO_QC.core.state import state_dir
from .base_scan import BaseScanDialog, ResultRow


class SplitHealthDialog(BaseScanDialog):
    TITLE, ICON = "Split Health", "⚖"
    HELP = ("Checks that every class is in every split with enough examples, that val / test look like train, and "
            "that no near-identical image sits in two splits (leakage inflates scores). Suggests file moves that "
            "fix it — nothing moves until you apply them, and moves can be undone.")
    COLUMNS = ["Move", "Reason"]
    FILTER_COLUMN = 0
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.min_img = QSpinBox()
        self.min_img.setRange(1, 10_000)
        self.min_img.setValue(int(self.app.settings.value("split_health/min", 10)))
        self.min_img.setToolTip("A class with fewer images than this in a split is reported (its score is noise)")
        self.leak = QCheckBox("Check leakage (near-identical images across splits — reads each image once, cached)")
        self.leak.setChecked(self.app.settings.value("split_health/leak", True, type=bool))
        self.thr = QSpinBox()
        self.thr.setRange(0, 40)
        self.thr.setValue(int(self.app.settings.value("split_health/thr", 10)))
        self.thr.setToolTip("Max differing bits of the 256-bit perceptual hash (0 = identical look; 10 = re-encoded / "
                            "slightly shifted copies)")
        form.addRow("Min images per class per split:", self.min_img)
        form.addRow(self.leak)
        form.addRow("Leakage tolerance (bits):", self.thr)
        self.add_action("⇄ Apply ticked moves", self.apply_moves,
                        "Move the ticked files to the suggested split (files keep their place in the layout)")
        self.undo_btn = self.add_action("↶ Undo last moves", self.undo_moves, "Put the files of the last Apply back")
        self.undo_btn.setEnabled(bool(self._last_log()))

    def post_init(self):
        self.health = QTextBrowser()
        self.health.setOpenLinks(False)
        self.health.anchorClicked.connect(self._link)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(self.health, "Health")
        self.tabs.addTab(self.table, "Suggested moves")
        self.table_host.addWidget(self.tabs)
        self.rep = None

    # ------------------------------------------------------------------ scan
    def make_context(self):
        s = self.session
        if not s.splits:
            raise ValueError("This dataset has no train / val / test split. Export & Split can build one.")
        if not getattr(s, "index_complete", True):
            raise ValueError("The label index is still being built (see the status bar) — try again in a moment.")
        st = self.app.settings
        st.setValue("split_health/min", self.min_img.value())
        st.setValue("split_health/leak", self.leak.isChecked())
        st.setValue("split_health/thr", self.thr.value())
        return {"adapter": s.base_adapter, "cache": s.cache, "leak": self.leak.isChecked(), "thr": self.thr.value(),
                "min": self.min_img.value(), "index": dict(s.index), "files": [], "hashes": {}}

    def scan_sample(self, ctx, s):
        ctx["files"].append((s.key, s.split, ctx["index"].get(s.key)))
        if not ctx["leak"] or not s.split:
            return []
        path = s.image_path
        if path is None or not path.exists():
            try:
                path = ctx["adapter"].load(s, light=True).image_path
            except Exception:
                path = None
        if path is None:
            return []
        cache = ctx["cache"]
        hit = cache.get_image(path) if cache is not None else None
        h = np.frombuffer(hit["phash"], dtype=np.uint8) if hit and hit.get("phash") else None
        if h is None:
            g = P.read_gray_reduced(path)
            if g is not None:
                h = P.phash(g)
                if cache is not None:
                    cache.put_image(path, phash=h.tobytes())
        if h is not None:
            ctx["hashes"][s.key] = h
        return []

    def finalize(self, ctx):
        if ctx["cache"] is not None:
            ctx["cache"].flush()
        rep = SH.analyze(ctx["files"], ctx["hashes"] if ctx["leak"] else None, min_images=ctx["min"],
                         hamming=ctx["thr"])
        ctx["rep"] = rep
        rows = []
        for m in rep["moves"]:
            tone = "error" if m.reason.startswith("near-duplicate") else "warning"
            rows.append(ResultRow(m.key, (f"{m.src} → {m.dst}", m.reason), payload=m, tone=tone))
        return rows

    def on_scan_complete(self, ctx):
        self.rep = ctx.get("rep")
        self.health.setHtml(self._html(self.rep) if self.rep else "")
        self.tabs.setTabText(1, f"Suggested moves ({len(self.rows):,})")
        self.tabs.setCurrentIndex(0)

    def summarize(self) -> str:
        if not self.rep:
            return ""
        F = self.rep["findings"]
        e = sum(f.severity == "error" for f in F)
        w = sum(f.severity == "warn" for f in F)
        return f"{e} problem(s) · {w} warning(s) · {len(self.rep['moves'])} suggested move(s)"

    # ------------------------------------------------------------------ health page
    def _html(self, rep) -> str:
        from AIO_QC.utils import hexc
        err, warn, mut = hexc("error"), hexc("warn"), "#888"
        sp = rep["splits"]
        T = rep["table"]
        h = ["<style>td,th{padding:2px 8px;} th{text-align:left;} .n{text-align:right;}</style>"]
        h.append("<h3>Splits</h3><table><tr><th>Split</th><th class=n>Images</th><th class=n>Share</th>"
                 "<th class=n>Instances</th><th class=n>Empty images</th><th class=n>Drift vs train</th></tr>")
        tot = sum(T[s]["images"] for s in sp) or 1
        for s in sp:
            t = T[s]
            d = rep["js"].get(s)
            dc = err if d is not None and d > 0.15 else warn if d is not None and d > 0.05 else ""
            h.append(f"<tr><td><b>{html.escape(s)}</b></td><td class=n>{t['images']:,}</td>"
                     f"<td class=n>{t['images'] / tot:.1%}</td><td class=n>{t['instances']:,}</td>"
                     f"<td class=n>{t['empty']:,}</td><td class=n style='color:{dc}'>"
                     f"{'—' if d is None else f'{d:.3f}'}</td></tr>")
        h.append("</table>")
        F = rep["findings"]
        if F:
            h.append("<h3>Findings</h3><table>")
            for f in sorted(F, key=lambda f: {"error": 0, "warn": 1, "info": 2}[f.severity]):
                c = {"error": err, "warn": warn}.get(f.severity, mut)
                icon = {"error": "●", "warn": "▲", "info": "○"}[f.severity]
                link = f" <a href='keys:{F.index(f)}'>show files</a>" if f.keys else ""
                h.append(f"<tr><td style='color:{c}'>{icon}</td><td><b>{html.escape(f.text)}</b>{link}<br>"
                         f"<span style='color:{mut}'>{html.escape(f.why)}</span></td></tr>")
            h.append("</table>")
        else:
            h.append("<p>✓ No split problems found.</p>")
        classes = sorted(set().union(*(T[s]["cls_img"] for s in sp)), key=str.lower)
        if classes:
            mn = self.min_img.value()
            h.append("<h3>Images per class</h3><table><tr><th>Class</th>" +
                     "".join(f"<th class=n>{html.escape(s)}</th>" for s in sp) + "</tr>")
            for c in classes:
                cells = []
                for s in sp:
                    n = T[s]["cls_img"].get(c, 0)
                    col = err if n == 0 else warn if n < mn else ""
                    share = n / max(1, T[s]["images"])
                    cells.append(f"<td class=n style='color:{col}'>{n:,} <span style='color:{mut}'>"
                                 f"({share:.0%})</span></td>")
                h.append(f"<tr><td>{html.escape(c)}</td>{''.join(cells)}</tr>")
            h.append("</table>")
        return "".join(h)

    def _link(self, url):
        t = url.toString()
        if t.startswith("keys:") and self.rep:
            f = self.rep["findings"][int(t[5:])]
            keys = [k for k in f.keys if k in self.session.by_key]
            if keys:
                self.app.apply_file_filter(dict.fromkeys(keys), f"Split health: {f.text}")

    def on_row_activated(self, row):
        if row is not None:
            self.app.open_sample(row.key)

    # ------------------------------------------------------------------ moves
    def _log_path(self):
        return state_dir(self.session.root) / "split_moves.json"

    def _last_log(self):
        try:
            return json.loads(self._log_path().read_text(encoding="utf-8")).get("log") or []
        except (OSError, ValueError):
            return []

    def apply_moves(self):
        rows = [r for r in self.target_rows() if isinstance(r.payload, SH.Move)]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) the moves to apply.")
            return
        if not self.confirm(f"Move {len(rows):,} file(s) to another split?\n\nThe files (image + annotation) move "
                            "between the split folders; review marks follow. Undo puts them back."):
            return
        s = self.session

        def image_of(smp):
            if smp.image_path is not None:
                return smp.image_path
            try:
                return s.base_adapter.load(smp, light=True).image_path
            except Exception:
                return None
        with s.as_tool("Split health"):
            new, errors, log = s.move_to_split([(r.key, r.payload.dst) for r in rows], image_of)
        if log:
            try:
                self._log_path().write_text(json.dumps({"when": time.time(), "log": log}), encoding="utf-8")
            except OSError:
                pass
        self.undo_btn.setEnabled(bool(log))
        self.report("Split moves", [f"Moved {len(new):,} file(s)."], errors)
        self.start_scan()

    def undo_moves(self):
        log = self._last_log()
        if not log or not self.confirm(f"Move the {len(log):,} file(s) of the last Apply back?"):
            return
        with self.session.as_tool("Split health"):
            back, errors = self.session.undo_split_moves(log)
        try:
            self._log_path().unlink()
        except OSError:
            pass
        self.undo_btn.setEnabled(False)
        self.report("Undo split moves", [f"Moved {len(back):,} file(s) back."], errors)
        self.start_scan()
