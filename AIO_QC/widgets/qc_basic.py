# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Dataset / Labels / Shapes tab tools (dialogs). The tab a tool appears in is defined in app.QC_TOOLS.

    📊 Dataset Stats · 🛡️ Validation · 📭 Empty · 🔍 Duplicate Shapes · 🖼️ Duplicate Images · 📝 Label Ambiguities
"""
from __future__ import annotations

import re

from AIO_QC.core.dataset import DETECTION_TYPES, DatasetType

from collections import Counter, defaultdict

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import AnnotationError
from AIO_QC.core.checks import (
    ERROR, ValidationOptions, classify_labels, find_duplicate_pairs, validate_dataset,
    validate_sample,
)
from AIO_QC.core import profile as P
from AIO_QC.core.geometry import to_polygon
from AIO_QC.core.fixes import FIX_DESCRIPTIONS, apply_fix, describe
from AIO_QC.utils import hexc, theme_color, style as apply_style
from AIO_QC.utils import newButton, newIcon, CSS_GRADIENT, balance_score, gradient_hex, imbalance_score, tone
from .base_scan import BAR_ROLE, BaseScanDialog, ResultRow, dspin, ispin, readable_table
from .card_gallery import Card, CardView, Panel, Shape

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    def natsorted(it, key=None):
        return sorted(it, key=key)


# =============================================================================
# 📊 Dataset Stats  (+ image profile from dataset_analysis_v2)
# =============================================================================
OBJ_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 9, "5–9"), (10, 10**9, "10+")]


class StatsDialog(BaseScanDialog):
    REVIEW_ACTIONS = False    # rows are labels, not files
    TITLE, ICON = "Dataset Stats", "📊"
    HELP = ("Labels: instances / images per label and split, coloured on a continuous scale by each label's share "
            "compared with an even split across all labels. Image profile: resolution, aspect family, orientation, "
            "format, small images, objects per image, lighting. Double-click any row to list its files.")
    COLUMNS = ["Label", "Instances", "Images", "% of instances", "vs. equal share"]
    FILE_COLUMN = CHECKABLE = False
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.profile_cb = QCheckBox("Image profile (reads image headers — fast)")
        self.profile_cb.setChecked(True)
        self.light_cb = QCheckBox("Lighting (decodes each image at reduced size — slower)")
        form.addRow(self.profile_cb)
        form.addRow(self.light_cb)

    def post_init(self):
        readable_table(self.table)
        # summary cards + legend above the tables
        self.cards = {}
        cards = QHBoxLayout()
        cards.setSpacing(8)
        for key, title in (("files", "Files"), ("inst", "Instances"), ("labels", "Labels"),
                           ("per", "Objects / file"), ("empty", "Without shapes"), ("bad", "Unreadable"),
                           ("imb", "Imbalance (max : min)"), ("rare", "Rarest label")):
            frame, value = self._card(title)
            self.cards[key] = (frame, value)
            cards.addWidget(frame)
        legend = QHBoxLayout()
        bar = QLabel()
        bar.setFixedSize(220, 12)
        bar.setStyleSheet(f"background:{CSS_GRADIENT}; border-radius:6px;")
        for w in (QLabel("Very rare"), bar, QLabel("Well represented")):
            if w is not bar:
                apply_style(w, "color: @muted;")
            legend.addWidget(w)
        note = QLabel("Each label is compared with an equal share for every label. "
                      "Green = has its fair share or more · Yellow = somewhat rare · Red = very rare.")
        apply_style(note, "color: @muted;")
        note.setToolTip("'vs. equal share' column: 1.00 = exactly an equal share; 0.25 = a quarter of it.")
        legend.addWidget(note)
        legend.addStretch(1)
        self.table_host.insertLayout(0, legend)
        self.table_host.insertLayout(0, cards)
        # wrap the label table + a profile table in tabs
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(self.table, "Labels")
        self.profile = QTableWidget(0, 4)
        self.profile.setHorizontalHeaderLabels(["Property", "Value", "Images", "% images"])
        self.profile.verticalHeader().setVisible(False)
        self.profile.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.profile.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.profile.horizontalHeader().setStretchLastSection(True)
        self.profile.cellDoubleClicked.connect(self._profile_activated)
        readable_table(self.profile)
        self.tabs.addTab(self.profile, "Image profile")
        self.table_host.addWidget(self.tabs)

    def make_context(self):
        return {"adapter": self.session.adapter, "splits": self.session.splits, "inst": Counter(),
                "imgs": Counter(), "split_inst": defaultdict(Counter), "split_n": Counter(),
                "label_keys": defaultdict(list), "empty_keys": [], "bad": 0, "n": 0,
                "profile": self.profile_cb.isChecked(), "light": self.light_cb.isChecked(),
                "prof": defaultdict(lambda: defaultdict(list)), "mapping": self.session.adapter.active,
                "cache": self.session.cache}

    def scan_sample(self, ctx, s):
        try:
            ann = ctx["adapter"].load(s, light=True)
        except AnnotationError:
            ctx["bad"] += 1
            return []
        ctx["n"] += 1
        ctx["split_n"][s.split] += 1
        counts = ann.label_counts()
        if not counts:
            ctx["empty_keys"].append(s.key)
        for label, n in counts.items():
            ctx["inst"][label] += n
            ctx["imgs"][label] += 1
            ctx["split_inst"][s.split][label] += n
            ctx["label_keys"][label].append(s.key)
        prof = ctx["prof"]
        n_obj = sum(counts.values())
        prof["Objects per image"][next(t for lo, hi, t in OBJ_BINS if lo <= n_obj <= hi)].append(s.key)
        if s.split:
            prof["Split"][s.split].append(s.key)
        if s.folder:
            prof["Folder"][s.folder].append(s.key)
        if ctx["profile"] or ctx["light"]:
            path = ann.image_path or s.image_path
            if path is None:
                prof["Image"]["missing"].append(s.key)
                return []
            prof["Format"][path.suffix.lower() or "?"].append(s.key)
        if ctx["profile"]:
            size = None
            if ann.width and ann.height and not ann.normalized:
                size = (int(ann.width), int(ann.height))
            elif ctx["cache"] is not None:
                hit = ctx["cache"].get_image(path)
                if hit and hit.get("w"):
                    size = (hit["w"], hit["h"])
            if size is None:
                size = ctx["adapter"].image_size(s, ann)
                if size and ctx["cache"] is not None:
                    ctx["cache"].put_image(path, w=size[0], h=size[1])
            if size:
                w, h = size
                prof["Resolution"][f"{w}×{h}"].append(s.key)
                prof["Aspect ratio"][P.aspect_family(w, h)].append(s.key)
                prof["Orientation"][P.orientation(w, h)].append(s.key)
                small = w < P.SMALL_DIMENSION_THRESHOLD or h < P.SMALL_DIMENSION_THRESHOLD
                prof[f"Small (<{P.SMALL_DIMENSION_THRESHOLD}px side)"]["yes" if small else "no"].append(s.key)
            else:
                prof["Image"]["unreadable header"].append(s.key)
        if ctx["light"]:
            hit = ctx["cache"].get_image(path) if ctx["cache"] is not None else None
            light = hit.get("light") if hit else None
            if light is None:
                g = P.read_gray_reduced(path)
                light = P.lighting(g)[0] if g is not None else None
                if light is not None and ctx["cache"] is not None:
                    ctx["cache"].put_image(path, light=light)      # also feeds Model Check slice analysis
            if light is not None:
                prof["Lighting"][light].append(s.key)
            else:
                prof["Image"]["undecodable"].append(s.key)
        return []

    def columns_for(self, ctx):
        return self.COLUMNS + [f"{sp} inst." for sp in ctx["splits"]]

    def finalize(self, ctx):
        total = sum(ctx["inst"].values())
        sp = ctx["splits"]
        n_cls = len(ctx["inst"])
        max_pct = max((v / total * 100 for v in ctx["inst"].values()), default=1.0) if total else 1.0
        rows = []
        for label in natsorted(ctx["inst"]):
            share = ctx["inst"][label] / total if total else 0.0
            score = balance_score(share, n_cls)
            pct = share * 100
            rows.append(ResultRow(
                "", (label, ctx["inst"][label], ctx["imgs"][label], round(pct, 2), round(share * n_cls, 2),
                     *[ctx["split_inst"][x][label] for x in sp]),
                payload=label,
                dot=gradient_hex(score, 0.55, 0.70),                            # status dot, readable on any theme
                bars={3: (pct / max_pct if max_pct else 0.0, gradient_hex(score, 0.58, 0.65))}))
        rows.append(ResultRow("", ("(no shapes)", 0, len(ctx["empty_keys"]), 0.0, "", *["" for _ in sp]),
                              payload="__empty__", tone="muted"))
        rows.append(ResultRow("", ("TOTAL", total, ctx["n"], 100.0, "",
                                   *[sum(ctx["split_inst"][x].values()) for x in sp]),
                              payload="__total__", emphasis=True))
        ctx["n_cls"] = n_cls
        return rows

    def on_scan_complete(self, ctx):
        if ctx.get("cache") is not None:
            ctx["cache"].flush()
        order = ["Split", "Folder", "Objects per image", "Resolution", "Aspect ratio", "Orientation",
                 f"Small (<{P.SMALL_DIMENSION_THRESHOLD}px side)", "Format", "Lighting", "Image"]
        n = max(ctx["n"], 1)
        rows = []
        for group in order:
            values = ctx["prof"].get(group)
            if not values:
                continue
            items = sorted(values.items(), key=lambda kv: -len(kv[1]))
            if group == "Objects per image":
                items = [(t, values[t]) for _, _, t in OBJ_BINS if t in values]
            if group == "Resolution" and len(items) > 20:
                rest = [k for _, ks in items[20:] for k in ks]
                items = items[:20] + [(f"{len(values) - 20} other sizes", rest)]
            rows += [(group, v, keys) for v, keys in items]
        self._fill_cards(ctx)
        group_max = defaultdict(int)
        for group, _v, keys in rows:
            group_max[group] = max(group_max[group], len(keys))
        self._profile_keys = []
        self.profile.setRowCount(len(rows))
        warn = {"lowlight", "very_lowlight", "yes", "missing", "unreadable header", "undecodable"}
        for r, (group, value, keys) in enumerate(rows):
            vals = (group, value, len(keys), round(len(keys) / n * 100, 2))
            for c, v in enumerate(vals):
                it = self._cell(v)
                if value in warn or str(value).startswith("other"):
                    it.setForeground(theme_color("warn"))
                if c == 3:
                    it.setData(BAR_ROLE, (len(keys) / max(group_max[group], 1),
                                          "#f5c16c" if value in warn else "#9cc3f5"))
                self.profile.setItem(r, c, it)
            self._profile_keys.append((f"{group}: {value}", keys))
        self.profile.resizeColumnsToContents()
        if not (ctx["profile"] or ctx["light"]):
            self.tabs.setTabText(1, "Image profile (partial — enable above)")
        else:
            self.tabs.setTabText(1, "Image profile")

    @staticmethod
    def _card(title):
        frame = QFrame()
        frame.setObjectName("card")
        apply_style(frame, "QFrame#card{border:1px solid @line; border-radius:8px; background:palette(base);}")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        value = QLabel("–")
        f = value.font()
        f.setPointSizeF(max(f.pointSizeF(), 9.0) + 5)
        f.setWeight(QFont.Weight.DemiBold)
        value.setFont(f)
        cap = QLabel(title)
        apply_style(cap, "color: @muted;")
        lay.addWidget(value)
        lay.addWidget(cap)
        return frame, value

    def _set_card(self, key, text, score=None, tip=""):
        frame, value = self.cards[key]
        value.setText(text)
        frame.setToolTip(tip)
        if score is None:
            apply_style(frame, "QFrame#card{border:1px solid @line; border-radius:8px; background:palette(base);}")
            value.setStyleSheet("")
        else:
            frame.setStyleSheet("QFrame#card{border:1px solid %s; border-left:4px solid %s; border-radius:8px;"
                                " background:%s;}" % (tone(score, "border"), tone(score, "border"), tone(score, "bg")))
            value.setStyleSheet(f"color:{tone(score, 'fg')};")

    def _fill_cards(self, ctx):
        inst = ctx["inst"]
        total = sum(inst.values())
        n = ctx["n"]
        self._set_card("files", f"{n:,}")
        self._set_card("inst", f"{total:,}")
        self._set_card("labels", f"{len(inst):,}", tip=", ".join(natsorted(inst)))
        self._set_card("per", f"{total / n:.2f}" if n else "–")
        empty = len(ctx["empty_keys"])
        self._set_card("empty", f"{empty:,}", 1.0 - min(empty / max(n, 1) * 5, 1.0),
                       "Files with no shapes (fine for background images)")
        self._set_card("bad", f"{ctx['bad']:,}", 1.0 if ctx["bad"] == 0 else 0.0,
                       "Annotation files that could not be parsed — see Validation")
        if inst:
            (big, nb), (small, ns) = inst.most_common(1)[0], inst.most_common()[-1]
            ratio = nb / max(ns, 1)
            self._set_card("imb", f"{ratio:,.0f} : 1" if ratio >= 10 else f"{ratio:.1f} : 1", imbalance_score(ratio),
                           f"{big} ({nb:,}) vs {small} ({ns:,}) — ≤3:1 green, ≥50:1 red")
            share = ns / total if total else 0
            self._set_card("rare", small if len(small) <= 18 else small[:16] + "…",
                           balance_score(share, len(inst)), f"{small}: {ns:,} instances ({share * 100:.2f} %)")
        else:
            self._set_card("imb", "–")
            self._set_card("rare", "–")

    def _profile_activated(self, row, _col):
        if 0 <= row < len(getattr(self, "_profile_keys", [])):
            title, keys = self._profile_keys[row]
            self.app.apply_file_filter(keys, title)

    def summarize(self):
        c = self.ctx
        if not c:
            return ""
        total = sum(c["inst"].values())
        per = total / c["n"] if c["n"] else 0
        split = ""
        if c["splits"]:
            split = " · " + ", ".join(f"{x}: {c['split_n'][x]:,}" for x in c["splits"])
        mapped = " · 🔀 label mapping active" if c.get("mapping") else ""
        return (f"{c['n']:,} files · {total:,} instances · {len(c['inst'])} labels · "
                f"{len(c['empty_keys']):,} without shapes · {c['bad']:,} unreadable · {per:.2f} objects/file"
                f"{split}{mapped}")

    def on_row_activated(self, row):
        if row is None or row.payload == "__total__":
            return
        if row.payload == "__empty__":
            keys, title = self.ctx["empty_keys"], "Files without shapes"
        else:
            keys, title = self.ctx["label_keys"].get(row.payload, []), f"Files with '{row.payload}'"
        if keys:
            self.app.apply_file_filter(keys, title)


# =============================================================================
# 🛡️ Validation
# =============================================================================
SHAPE_FIXABLE = {"Invalid geometry", "Class ID out of range", "Malformed label line", "Empty label",
                 "Coordinates outside image"}
FILE_TRASHABLE = {"Unreadable annotation", "Not LabelMe format", "Label without image"}


class ValidationDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES | {DatasetType.CLS_LABELME}
    TITLE, ICON = "Validation", "🛡️"
    HELP = ("File & geometry INTEGRITY — pairing (image ↔ annotation), schema, imagePath, image size, class map / "
            "class-id range, point counts, geometry (zero area, self-intersection, repeated vertices, outside the "
            "image). Label names → 📝 Label Ambiguities · empty files → 📭 Empty · duplicates → 🔍 / 🖼️ Duplicates.")
    COLUMNS = ["Severity", "Category", "Detail", "Shape", "Proposed fix"]
    FILTER_COLUMN = 1

    def build_params(self, form):
        self.deep = QCheckBox("Decode every image (slow; catches corrupt files and JSON/image size mismatch)")
        self.geometry_cb = QCheckBox("Geometry checks")
        self.geometry_cb.setChecked(True)
        self.min_area = dspin(0, 1e6, 2.0, 1, suffix=" px²")
        form.addRow(self.deep)
        form.addRow(self.geometry_cb)
        form.addRow("Tiny-area threshold:", self.min_area)
        self.add_action("🩹 Review fixes…", self.review_fixes,
                        "Preview the proposed fix for each checked issue; nothing changes until you confirm")
        self.add_action("🗑 Delete flagged shapes", self.delete_flagged_shapes,
                        f"Delete shapes for: {', '.join(sorted(SHAPE_FIXABLE))}")
        self.add_action("🗑 Trash flagged files", self.trash_flagged_files,
                        f"Move to .aio_qc_trash: {', '.join(sorted(FILE_TRASHABLE))}")

    def make_context(self):
        return {"adapter": self.session.adapter, "samples": list(self.session.samples),
                "opts": ValidationOptions(deep_image_check=self.deep.isChecked(),
                                          geometry=self.geometry_cb.isChecked(),
                                          min_area_px=self.min_area.value())}

    @staticmethod
    def _row(key, issue):
        fix = FIX_DESCRIPTIONS.get(issue.fix, "manual") if issue.fix else "manual"
        return ResultRow(key, (issue.severity, issue.category, issue.detail,
                               "" if issue.shape_index is None else issue.shape_index, fix),
                         issue.shape_index, issue, "error" if issue.severity == ERROR else "warning")

    def scan_sample(self, ctx, s):
        return [self._row(s.key, i) for i in validate_sample(ctx["adapter"], s, ctx["opts"])]

    def finalize(self, ctx):
        return [self._row(path, i) for path, i in validate_dataset(ctx["adapter"], ctx["samples"])]

    def summarize(self):
        c = Counter(r.cells[0] for r in self.rows if r)
        if not c:
            return "✅ No issues found"
        fixable = sum(1 for r in self.rows if r and r.payload is not None and r.payload.fix)
        return (f"❌ {c.get('Error', 0):,} error(s) · ⚠ {c.get('Warning', 0):,} warning(s) · "
                f"🩹 {fixable:,} with a proposed fix (tick rows → Review fixes…)")

    def review_fixes(self):
        rows = [r for r in self.target_rows() if r.payload is not None and r.payload.fix and r.key in self.session.by_key]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) issues whose 'Proposed fix' is not 'manual'.")
            return
        dlg = FixPreviewDialog(self, rows)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.chosen_rows()
        if not chosen:
            return
        by_key = defaultdict(list)
        for r in chosen:
            by_key[r.key].append(r.payload)
        applied = [0]

        def make(issues):
            def fn(ann):
                changed = False
                # deletions last and from the end, so the other fixes still address the right shapes
                for iss in sorted(issues, key=lambda i: (i.fix == "delete_shape", -(i.shape_index or 0))):
                    if apply_fix(ann, iss.fix, iss.shape_index, iss.fix_arg):
                        applied[0] += 1
                        changed = True
                return changed
            return fn
        changed, errors = self.session.modify_many(self, [(k, make(v)) for k, v in by_key.items()],
                                                   "Applying fixes...", raw=True)
        self.after_edit(by_key.keys())
        self.report("Fixes applied", [f"Applied {applied[0]} fix(es) in {len(changed)} file(s).",
                                      "Affected files were re-validated; remaining issues are still listed."], errors)

    def delete_flagged_shapes(self):
        rows = [r for r in self.target_rows() if r.cells[1] in SHAPE_FIXABLE]
        self.delete_shape_rows(rows, "flagged shape(s)")

    def trash_flagged_files(self):
        rows = [r for r in self.target_rows() if r.cells[1] in FILE_TRASHABLE]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Check rows of these categories first:\n• " +
                                    "\n• ".join(sorted(FILE_TRASHABLE)))
            return
        sample_keys = list(dict.fromkeys(r.key for r in rows if r.key in self.session.by_key))
        paths = list(dict.fromkeys(r.key for r in rows if r.key not in self.session.by_key))
        if not self.confirm(f"Move {len(sample_keys) + len(paths)} file(s) to .aio_qc_trash?\n"
                            "(Recoverable: they are moved, not deleted.)"):
            return
        _, e1 = self.session.trash_samples(sample_keys)
        done, e2 = self.session.trash_paths(paths)
        ds = set(done)
        self.remove_rows(lambda r: r.key in ds)
        self.after_edit([k for k in sample_keys if k in self.session.by_key])
        self.report("Done", [f"Moved {len(sample_keys) + len(done)} file(s) to trash."], e1 + e2)


class FixPreviewDialog(QDialog):
    """Every proposed change, one per row, each individually ticked. Nothing is written here."""

    def __init__(self, parent, rows):
        super().__init__(parent)
        self.setWindowTitle("Review proposed fixes")
        self.setWindowIcon(newIcon("edit"))
        self.resize(900, 480)
        self.rows = rows
        session = parent.session
        cache = {}
        self.table = QTableWidget(len(rows), 5)
        self.table.setHorizontalHeaderLabels(["Apply", "File", "Issue", "Fix", "Change"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, row in enumerate(rows):
            iss = row.payload
            if row.key not in cache:
                try:
                    cache[row.key] = session.base_adapter.load(session.by_key[row.key])
                except Exception:
                    cache[row.key] = None
            ann = cache[row.key]
            shape = ann.shapes[iss.shape_index] if ann and iss.shape_index is not None and \
                iss.shape_index < len(ann.shapes) else None
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(r, 0, chk)
            self.table.setItem(r, 1, QTableWidgetItem(session.by_key[row.key].display))
            self.table.setItem(r, 2, QTableWidgetItem(f"{iss.category}: {iss.detail}"))
            self.table.setItem(r, 3, QTableWidgetItem(FIX_DESCRIPTIONS.get(iss.fix, iss.fix)))
            self.table.setItem(r, 4, QTableWidgetItem(describe(iss.fix, shape, iss.fix_arg)))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        info = QLabel(f"{len(rows)} proposed change(s). Untick anything you do not want. "
                      "Files are rewritten atomically; shapes are addressed by index as scanned.")
        info.setWordWrap(True)
        buttons = QHBoxLayout()
        none_btn = newButton("Untick all", slot=lambda: self._set_all(False))
        all_btn = newButton("Tick all", slot=lambda: self._set_all(True))
        cancel = newButton("Cancel", slot=self.reject)
        ok = newButton("✓ Apply ticked", slot=self.accept)
        ok.setDefault(True)
        for b in (none_btn, all_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.table, 1)
        lay.addLayout(buttons)

    def _set_all(self, on):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def chosen_rows(self):
        return [self.rows[r] for r in range(self.table.rowCount())
                if self.table.item(r, 0).checkState() == Qt.CheckState.Checked]


# =============================================================================
# 📭 Empty annotations
# =============================================================================
class EmptyDialog(BaseScanDialog):
    TITLE, ICON = "Empty Annotations", "📭"
    HELP = ("LabelMe JSONs without shapes, YOLO images with an empty or missing label file. "
            "In YOLO these are usually intentional background/negative images — keep them unless unintended.")
    COLUMNS = ["Kind", "Split"]
    FILTER_COLUMN = 0
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.also_images = QCheckBox("also move images")
        self.actions_layout.addWidget(self.also_images)
        self.add_action("🗑 Move to trash", self.trash_selected,
                        "Moves annotation files (and optionally images) to <root>/.aio_qc_trash")

    def make_context(self):
        return {"adapter": self.session.adapter, "cls": self.session.is_classification}

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        if ctx["cls"]:                                   # classification: an image is empty without a class
            if not (ann.tags or []):
                return [ResultRow(s.key, ("no class", s.split))]
            return []
        if not ann.exists:
            return [ResultRow(s.key, ("no label file", s.split), tone="muted")]
        if not ann.shapes:
            return [ResultRow(s.key, ("no shapes", s.split))]
        return []

    def trash_selected(self):
        rows = self.target_rows()
        if not rows:
            QMessageBox.information(self, self.TITLE, "Check (or select) rows first.")
            return
        imgs = self.also_images.isChecked()
        keys = list(dict.fromkeys(r.key for r in rows))
        what = "annotation files and images" if imgs else "annotation files"
        if not self.confirm(f"Move {what} of {len(keys)} sample(s) to .aio_qc_trash?"):
            return
        done, errors = self.session.trash_samples(keys, include_images=imgs)
        ds = set(done)
        self.remove_rows(lambda r: r.key in ds)
        self._compact()
        self.summary.setText(self.summarize())
        self.report("Done", [f"Moved {len(done)} sample(s) to trash."], errors)


# =============================================================================
# 🔍 Duplicates
# =============================================================================
class DuplicateDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Duplicate Shapes", "🔍"
    HELP = ("Exact: identical geometry after rounding — independent of start vertex, drawing direction and "
            "rectangle-vs-4-point-polygon. Overlap: IoU ≥ threshold (exact for box pairs, rasterised for polygons). "
            "YOLO: identical label lines are exact duplicates. "
            "Every duplicate is paired with the first shape of its cluster. Untick 'same class only' to catch one "
            "object annotated with two labels — then choose which side to delete.")
    COLUMNS = ["Label (kept)", "Label (dup)", "Kept #", "Dup #", "IoU"]

    def build_params(self, form):
        self.mode = QComboBox()
        self.mode.addItems(["Exact points", "Overlap (IoU)"])
        self.iou = dspin(0.3, 1.0, 0.90, 2, 0.01)
        self.same = QCheckBox("Same class only")
        self.same.setChecked(True)
        self.decimals = ispin(0, 8, 6 if self.session.is_yolo else 2)
        self.mode.currentIndexChanged.connect(self._sync)
        form.addRow("Mode:", self.mode)
        form.addRow("IoU threshold:", self.iou)
        form.addRow("Round points to decimals:", self.decimals)
        form.addRow(self.same)
        self._sync()
        self.add_action("🗑 Delete duplicate (B)", lambda: self.delete_shape_rows(self.target_rows(), "duplicate(s)"),
                        "Keep the first shape of each pair")
        self.add_action("🗑 Delete kept (A)", self._delete_kept,
                        "For cross-class pairs where the first label is the wrong one")

    def post_init(self):
        """Pairs tab (visual, default) + Table tab; ticks shared."""
        self.init_marks()
        self.cards = CardView(self, n_panels=2)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Left: the shape that is kept (A) · right: its duplicate (B, red) with A dashed for comparison · "
                      "tick a pair, then “Delete duplicate (B)” or “Delete kept (A)”")
        apply_style(hint, "color:@muted;")
        lay.addWidget(hint)
        lay.addWidget(self.cards, 1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(w, "Pairs")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)

    def _refresh_cards(self):
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        ok, dup, cmp_ = hexc("ok"), hexc("error"), hexc("ok")
        cards = []
        for row in self.visible_rows():
            p = row.payload or {}
            a, b = p.get("a") or [], p.get("b") or []
            if not a or not b:
                continue
            xs = [q[0] for q in a + b]
            ys = [q[1] for q in a + b]
            focus = (min(xs), min(ys), max(xs), max(ys))
            same = row.cells[0] == row.cells[1]
            s = self.session.by_key.get(row.key)
            cards.append(Card(row, [
                Panel(row.key, focus, [Shape(a, ok)], p.get("norm", False), caption=f"A · kept #{row.cells[2]}"),
                Panel(row.key, focus, [Shape(a, cmp_, dashed=True, width=1.5), Shape(b, dup)], p.get("norm", False),
                      caption=f"B · duplicate #{row.cells[3]}")],
                chips=[(f"IoU {row.cells[4]:.2f}", "info"), ("same class" if same else "cross-class",
                                                              "muted" if same else "warn")],
                lines=[f"{row.cells[0]}" if same else f"{row.cells[0]}  vs  {row.cells[1]}",
                       s.display if s else row.key]))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Pairs ({len(cards):,})")

    def summarize(self):
        self._refresh_cards()
        return super().summarize()

    def _delete_kept(self):
        rows = [ResultRow(r.key, r.cells, r.cells[2]) for r in self.target_rows()]
        self.delete_shape_rows(rows, "kept shape(s)")

    def _sync(self):
        iou_mode = self.mode.currentIndex() == 1
        self.iou.setEnabled(iou_mode)
        self.decimals.setEnabled(not iou_mode)

    def make_context(self):
        return {"adapter": self.session.adapter, "mode": "iou" if self.mode.currentIndex() else "exact",
                "iou": self.iou.value(), "same": self.same.isChecked(), "dec": self.decimals.value()}

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        shapes = ann.shapes
        pairs = find_duplicate_pairs(shapes, ctx["mode"], ctx["iou"], ctx["same"], ctx["dec"])
        def outline(k):                   # polygon for the pair cards (rectangles as 4 corners)
            poly = to_polygon(shapes[k])
            return poly.tolist() if poly is not None else [list(p) for p in shapes[k].get("points", [])]
        return [ResultRow(s.key, (shapes[i].get("label"), shapes[j].get("label"), i, j, round(iou, 3)),
                          shape_index=j, tone=None if shapes[i].get("label") == shapes[j].get("label") else "warning",
                          payload={"a": outline(i), "b": outline(j), "norm": ann.normalized})
                for i, j, iou in pairs]


# =============================================================================
# 🖼️ Duplicate images (exact + perceptual, split leakage)
# =============================================================================
_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def _suffix_depth(stem: str) -> int:
    """img -> 0, img_1 -> 1, img_1_2 -> 2 (how "renamed" a name looks)."""
    d = 0
    while _SUFFIX.match(stem):
        stem = _SUFFIX.match(stem).group(1)
        d += 1
    return d


class ImageDuplicatesDialog(BaseScanDialog):
    TITLE, ICON = "Duplicate Images", "🖼️"
    HELP = ("Exact: identical file bytes (SHA-256). Near: perceptual hash (256-bit DCT) within the Hamming "
            "threshold — re-encodes, resizes, small brightness changes. Same name: one file name in several splits. "
            "Groups spanning splits are leakage (the same scene in train and val). "
            "Tick → 'Keep one per group' → trash the rest.")
    COLUMNS = ["Group", "Kind", "Distance", "Splits in group", "Leakage", "vs kept copy"]
    FILTER_COLUMN = 1
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.exact_cb = QCheckBox("Exact duplicates (SHA-256)")
        self.exact_cb.setChecked(True)
        self.near_cb = QCheckBox("Near duplicates (perceptual hash)")
        self.near_cb.setChecked(True)
        self.thr = ispin(0, 40, 8)
        self.leak_only = QCheckBox("Only groups spanning several splits (leakage)")
        self.leak_only.setVisible(bool(self.session.splits))
        self.same_name = QCheckBox("Same file name in several splits (cheap leakage check, no hashing)")
        self.same_name.setChecked(bool(self.session.splits))
        self.same_name.setVisible(bool(self.session.splits))
        form.addRow(self.exact_cb)
        form.addRow(self.near_cb)
        form.addRow("Hamming threshold (of 256 bits):", self.thr)
        form.addRow(self.leak_only)
        form.addRow(self.same_name)
        self.add_action("☑ Keep one per group", self._keep_one,
                        "Mark every image except one per group (keeps the train-split copy when there is one)")
        self.add_action("↺ Clear marks", lambda: self._toggle_all(False), "Unmark everything")
        self.add_action("🗑 Trash marked", self._trash, "Move the marked images and their annotations to .aio_qc_trash")

    def post_init(self):
        """Groups tab (visual, default) + Table tab; both share the table's check states as marks."""
        from .dup_gallery import DupGalleryView
        self.init_marks()
        self.gallery = DupGalleryView(self)
        self.on_marks_changed(self.gallery._repaint.start)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        groups = QWidget()
        gl = QVBoxLayout(groups)
        gl.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Click an image to open it · tick its box to mark it for trash · "
                      "“Keep one per group” marks all extra copies at once")
        apply_style(hint, "color:@muted;")
        gl.addWidget(hint)
        gl.addWidget(self.gallery, 1)
        self.tabs.addTab(groups, "Groups")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_groups())

    def start_scan(self):
        if hasattr(self, "gallery"):
            self.gallery.set_groups([])
        super().start_scan()

    def _refresh_groups(self):
        self.prune_marks()
        want = self.filter_box.currentData() if self.FILTER_COLUMN is not None else None
        by_gid: dict = {}
        for row in self.rows:
            if row is None or (want is not None and str(row.cells[1]) != want):
                continue
            by_gid.setdefault(row.payload[0], []).append(row)
        groups = []
        for gid, members in by_gid.items():
            members.sort(key=lambda r: r.payload[1])
            splits = sorted({self.session.by_key[m.key].split for m in members
                             if m.key in self.session.by_key and self.session.by_key[m.key].split})
            dists = [m.cells[2] for m in members if isinstance(m.cells[2], (int, float))]
            groups.append({"label": members[0].cells[0], "kind": members[0].cells[1], "members": members,
                           "splits": splits, "leak": bool(members[0].cells[4]), "max_dist": max(dists) if dists else 0})
        groups.sort(key=lambda g: (not g["leak"], -len(g["members"]), g["label"]))
        self.gallery.set_groups(groups)
        self.tabs.setTabText(0, f"Groups ({len(groups):,})")

    def make_context(self):
        if not (self.exact_cb.isChecked() or self.near_cb.isChecked() or self.same_name.isChecked()):
            raise ValueError("Enable at least one check.")
        return {"adapter": self.session.adapter, "exact": self.exact_cb.isChecked(), "cache": self.session.cache,
                "near": self.near_cb.isChecked(), "thr": self.thr.value(),
                "leak_only": self.leak_only.isChecked(), "items": [], "index": dict(self.session.index),
                "same_name": self.same_name.isChecked() and bool(self.session.splits)}

    def scan_sample(self, ctx, s):
        if not (ctx["exact"] or ctx["near"]):
            ctx["items"].append((s.key, s.split, None, None, s.name, None))
            return []
        path = s.image_path
        if path is None or not path.exists():
            path = ctx["adapter"].load(s, light=True).image_path
        if path is None:
            return []
        cache = ctx["cache"]
        hit = cache.get_image(path) if cache is not None else None
        sha = hit["sha"] if hit and hit.get("sha") else None
        h = np.frombuffer(hit["phash"], dtype=np.uint8) if hit and hit.get("phash") else None
        new = {}
        if ctx["exact"] and sha is None:
            sha = new["sha"] = P.sha256_file(path)
        if ctx["near"] and h is None:
            g = P.read_gray_reduced(path)
            if g is not None:
                h = P.phash(g)
                new["phash"] = h.tobytes()
        if new and cache is not None:
            cache.put_image(path, **new)
        try:
            st = path.stat()
            meta = ((hit or {}).get("w") or 0) * ((hit or {}).get("h") or 0), st.st_size, st.st_mtime
        except OSError:
            meta = (0, 0, 0)
        ctx["items"].append((s.key, s.split, sha, h, s.name, meta))
        return []

    def finalize(self, ctx):
        if ctx["cache"] is not None:
            ctx["cache"].flush()
        items = ctx["items"]
        groups = []                           # (kind, [indices], {index: distance})
        in_exact = {}
        if ctx["exact"]:
            by_sha = defaultdict(list)
            for i, it in enumerate(items):
                if it[2]:
                    by_sha[it[2]].append(i)
            for idxs in by_sha.values():
                if len(idxs) > 1:
                    groups.append(("exact", idxs, {i: 0 for i in idxs}))
                    for i in idxs[1:]:
                        in_exact[i] = idxs[0]
        if ctx["near"]:
            cand = [i for i, it in enumerate(items) if it[3] is not None and i not in in_exact]
            pairs = P.near_duplicate_pairs([items[i][3] for i in cand], ctx["thr"])
            for g in P.union_find_groups(len(cand), pairs):
                idxs = [cand[x] for x in g]
                first = items[idxs[0]][3]
                groups.append(("near", idxs, {i: P.hamming(first, items[i][3]) for i in idxs}))
        if ctx["same_name"]:
            by_stem = defaultdict(list)
            for i, it in enumerate(items):
                by_stem[it[4].rsplit(".", 1)[0]].append(i)
            for idxs in by_stem.values():
                if len({items[i][1] for i in idxs}) > 1:
                    groups.append(("same name", idxs, {i: "" for i in idxs}))
        rows = []
        for gid, (kind, idxs, dist) in enumerate(sorted(groups, key=lambda g: -len(g[1])), 1):
            splits = sorted({items[i][1] for i in idxs if items[i][1]})
            leak = len(splits) > 1
            if ctx["leak_only"] and not leak:
                continue
            # the copy to keep comes first: train split > annotated > original name (fewest _1 / _2 suffixes) >
            # most pixels > largest file > oldest > shortest path
            index = ctx.get("index") or {}

            def keep_rank(i):
                it = items[i]
                meta = it[5] or (0, 0, 0)
                return (it[1] not in ("train", ""), 0 if sum((index.get(it[0]) or {}).values()) else 1,
                        _suffix_depth(it[4].rsplit(".", 1)[0]), -meta[0], -meta[1], meta[2], len(it[0]), it[0])
            idxs = sorted(idxs, key=keep_rank)
            kept = index.get(items[idxs[0]][0]) or {}
            for rank, i in enumerate(idxs):
                lab = index.get(items[i][0]) or {}
                vs = ("kept" if rank == 0 else "no labels" if not lab else "kept copy has no labels" if not kept
                      else "same labels" if dict(lab) == dict(kept) else "⚠ DIFFERENT labels")
                rows.append(ResultRow(items[i][0], (f"{kind[0].upper()}{gid:04d}", kind, dist[i],
                                                    ", ".join(splits), "⚠ yes" if leak else "", vs),
                                      payload=(gid, rank), tone="error" if leak else
                                      ("warning" if vs.startswith("⚠") else None)))
        ctx["n_groups"] = len({r.payload[0] for r in rows})
        ctx["items"] = []
        return rows

    def summarize(self):
        if hasattr(self, "gallery"):
            self._refresh_groups()
        live = [r for r in self.rows if r]
        leak = len({r.payload[0] for r in live if r.cells[4]})
        return (f"{self.ctx.get('n_groups', 0):,} group(s) · {len(live):,} image(s) · "
                f"{leak} group(s) leak across splits")

    def _keep_one(self):
        self.table.setSortingEnabled(False)
        for r in range(self.table.rowCount()):
            row = self._row_at(r)
            if row is not None and not self.table.isRowHidden(r):
                self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if row.payload[1] > 0
                                                    else Qt.CheckState.Unchecked)
        self.table.setSortingEnabled(True)

    def _trash(self):
        rows = self.target_rows()
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick rows first (tip: ☑ Keep one per group).")
            return
        groups = defaultdict(int)
        for r in self.rows:
            if r:
                groups[r.payload[0]] += 1
        chosen = defaultdict(int)
        for r in rows:
            chosen[r.payload[0]] += 1
        whole = [g for g, n in chosen.items() if n >= groups[g]]
        warn = f"\n\n⚠ {len(whole)} group(s) would lose EVERY copy." if whole else ""
        keys = list(dict.fromkeys(r.key for r in rows))
        if not self.confirm(f"Move {len(keys)} image(s) and their annotations to .aio_qc_trash?{warn}"):
            return
        done, errors = self.session.trash_samples(keys, include_images=True)
        ds = set(done)
        self.remove_rows(lambda r: r.key in ds)
        self._compact()
        self.summary.setText(self.summarize())
        self.report("Done", [f"Moved {len(done)} sample(s) to trash."], errors)


# =============================================================================
# 📝 Label ambiguities
# =============================================================================
class AmbiguityDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    REVIEW_ACTIONS = False    # rows are labels, not files
    TITLE, ICON = "Label Ambiguities", "📝"
    HELP = ("Finds labels that are not in the class list: format variants (case / whitespace / '-' vs '_'), "
            "near-misspellings, and unknown labels. LabelMe without a class list: rarer labels are compared "
            "against more frequent ones. YOLO: the class list is dataset.yaml, else classes.txt; class ids "
            "outside it appear as 'class_N' and can be merged into a valid class. "
            "Double-click a row to list the files using that label.")
    COLUMNS = ["Label", "Kind", "Suggestion", "Similarity", "Instances", "Files"]
    FILE_COLUMN = False
    FILTER_COLUMN = 1
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.cutoff = dspin(0.5, 0.99, 0.80, 2, 0.05)
        self.source = QLabel()
        load = newButton("Load classes…", slot=self._load_classes)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.source, 1)
        h.addWidget(load)
        form.addRow("Similarity cutoff:", self.cutoff)
        form.addRow("Reference:", row)
        self._update_source()
        self.add_action("🔀 Fix in Label Mapping…", self.fix_in_mapping,
                        "Open Label Mapping with the ticked rows pre-filled (view-only mapping or write to files)")

    def _update_source(self):
        src = self.session.classes_source
        if src:
            self.source.setText(f"class list from {src}")
        elif self.session.is_yolo:
            self.source.setText("⚠ no class list — load classes.txt")
        else:
            self.source.setText("no class list → comparing labels to each other")

    def _load_classes(self):
        if self.app.load_classes_file():
            self._update_source()
            self.start_scan()

    def make_context(self):
        return {"adapter": self.session.adapter, "truth": self.session.true_classes,
                "cutoff": self.cutoff.value(), "counts": Counter(), "files": defaultdict(list)}

    def scan_sample(self, ctx, s):
        for label, n in ctx["adapter"].load(s, light=True).label_counts().items():
            ctx["counts"][label] += n
            ctx["files"][label].append(s.key)
        return []

    def finalize(self, ctx):
        rows = []
        for f in classify_labels(ctx["counts"], ctx["truth"], ctx["cutoff"]):
            shown = f.label if f.label == f.label.strip() else repr(f.label)
            tone = {"unknown": "error", "similar": "warning"}.get(f.kind)
            rows.append(ResultRow("", (shown, f.kind, f.suggestion, f.score, ctx["counts"][f.label],
                                       len(ctx["files"][f.label])), payload=f.label, tone=tone))
        return rows

    def summarize(self):
        return f"{len([r for r in self.rows if r])} suspicious label(s)" if self.rows else "✅ All labels look consistent"

    def on_row_activated(self, row):
        if row is not None:
            keys = self.ctx["files"].get(row.payload, [])
            self.app.apply_file_filter(keys, f"Files with label {row.payload!r}")

    def fix_in_mapping(self):
        """Ambiguities only FIND; Label Mapping is the single place that renames / merges labels."""
        rows = self.target_rows() or [r for r in self.rows if r]
        prefill = {r.payload: r.cells[2] for r in rows if r.cells[2]}
        focus = rows[0].payload if rows else None
        if not prefill and not focus:
            QMessageBox.information(self, self.TITLE, "No suggestions to hand over — tick rows first.")
            return
        self.app.open_label_map(prefill=prefill, focus=focus)
        self.start_scan()          # labels may have changed (written or mapped)
