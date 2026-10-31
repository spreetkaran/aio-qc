# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🎞 Sequence Check — consecutive frames checked against their neighbours (any dataset; see core.sequences).

Finds what a single image never reveals: an object present in the frames before and after but not annotated here,
a label that flips for a frame or two, a box that suddenly balloons or jumps, a text / attribute that changes for one
frame, and objects that exist in only one frame of a steady sequence. Missing boxes can be added (interpolated from
the neighbours) with one click — backed up, undoable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QMessageBox, QWidget

from AIO_QC.core.dataset import DETECTION_TYPES
from AIO_QC.core.sequences import CODES, FrameInfo, attrs_of, build_sequences, check_sequence, split_name
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import hexc
from .base_scan import BaseScanDialog, ResultRow, dspin, ispin


class SequenceDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Sequence Check", "🎞"
    HELP = ("Frames from one video / burst are compared with their neighbours. Sequences are found from the file "
            "names (folder + the name without its last number, ordered by that number; a big jump in the number "
            "starts a new sequence). Each object is followed from frame to frame, and one-frame breaks are reported "
            "with the reason. Double-click a row to see it — a missing object is drawn as a dashed box.")
    COLUMNS = ["Check", "Label", "Frame", "Sequence", "Why"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.mode = QComboBox()
        self.mode.addItem("Number in the file name (frame_000123, …)", "auto")
        self.mode.addItem("Each folder is one sequence (name order)", "folder")
        self.mode.setToolTip("How frames are put in order. Names without any number always use name order.")
        self.gap = dspin(1.5, 50, 3.0, 1, step=0.5)
        self.gap.setToolTip("A new sequence starts where the frame number jumps by more than this many times the "
                            "usual step (frames extracted every 5th frame: step 5; a jump > 15 starts a new one)")
        self.min_len = ispin(3, 1000, 4)
        self.min_len.setToolTip("Shorter runs are not checked (too little context)")
        self.max_gap = ispin(1, 3, 1)
        self.max_gap.setToolTip("How many consecutive frames an object may be missing and still be reported "
                                "(1 = present before and after a single frame)")
        self.iou = dspin(0.05, 0.9, 0.3, 2, step=0.05)
        self.iou.setToolTip("Overlap needed to link a box to the same object in the next frame; small fast objects "
                            "are also linked by a close centre and a similar size")
        self.size = dspin(1.2, 10, 1.8, 1, step=0.1)
        self.size.setToolTip("Report a box that is this many times larger / smaller than in BOTH neighbouring frames")
        self.scene = QCheckBox("Also cut where the picture changes completely (reads small thumbnails — slower)")
        self.scene.setToolTip("For folders that mix several recordings with continuous numbering")
        self.label = QComboBox()
        self.label.addItem("(all labels)", None)
        for lab, _n in self.session.label_totals().most_common():
            self.label.addItem(lab, lab)
        checks = QWidget()
        grid = QGridLayout(checks)
        grid.setContentsMargins(0, 0, 0, 0)
        self.checks = {}
        for i, (code, title) in enumerate(CODES.items()):
            cb = QCheckBox(title)
            cb.setChecked(True)
            grid.addWidget(cb, i // 3, i % 3)
            self.checks[code] = cb
        form.addRow("Order by:", self.mode)
        row = QWidget()
        g2 = QGridLayout(row)
        g2.setContentsMargins(0, 0, 0, 0)
        for c, (text, w) in enumerate((("new sequence at step ×", self.gap), ("min frames", self.min_len),
                                       ("missing up to", self.max_gap), ("link IoU", self.iou),
                                       ("size spike ×", self.size))):
            g2.addWidget(QLabel(text), 0, 2 * c)
            g2.addWidget(w, 0, 2 * c + 1)
        form.addRow("Sequences:", row)
        form.addRow("", self.scene)
        form.addRow("Label:", self.label)
        form.addRow("Report:", checks)
        self.add_action("➕ Add missing boxes", self._add_missing,
                        "Add the interpolated box for the ticked (or selected) ‘Missing in this frame’ rows — "
                        "backed up, undoable")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Remove the boxes added by the last ‘Add missing boxes’")
        self.undo_btn.setEnabled(False)
        self._last_backup = None

    # ------------------------------------------------------------------ scan
    def make_context(self):
        return {"adapter": self.session.adapter, "frames": [], "scene": self.scene.isChecked(),
                "label": self.label.currentData(),            # parameters read here, on the GUI thread
                "mode": self.mode.currentData(), "gap": self.gap.value(), "min_len": self.min_len.value(),
                "max_gap": self.max_gap.value(), "iou": self.iou.value(), "size": self.size.value(),
                "enabled": {c for c, cb in self.checks.items() if cb.isChecked()}}

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        if ann.normalized:
            W = H = 1.0
        else:
            W, H = (ann.width, ann.height) if ann.width and ann.height else (ctx["adapter"].image_size(s, ann) or (0, 0))
        if not (W and H):
            return []
        boxes = []
        want = ctx["label"]
        for i, sh in enumerate(ann.shapes):
            pts = sh.get("points") or []
            if not pts or sh.get("_error") or (want is not None and sh.get("label") != want):
                continue
            try:
                xs, ys = [float(p[0]) / W for p in pts], [float(p[1]) / H for p in pts]
            except (TypeError, ValueError, IndexError):
                continue
            if (sh.get("shape_type") or "polygon") == "circle" and len(pts) == 2:
                r = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
                box = (xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r)
            else:
                box = (min(xs), min(ys), max(xs), max(ys))
            if box[2] - box[0] <= 0 and box[3] - box[1] <= 0:
                pad = 0.004
                box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)       # points: a tiny box
            boxes.append((i, str(sh.get("label")), box, attrs_of(sh)))
        p = Path(s.key)
        pattern, n = split_name(p.stem)
        group = f"{s.split}|{p.parent}|{pattern}"
        thumb = None
        if ctx["scene"]:
            thumb = _thumb(ann.image_path or s.image_path)
        ctx["frames"].append(FrameInfo(s.key, group, n, boxes, thumb))
        return []

    def finalize(self, ctx):
        seqs = build_sequences(ctx["frames"], ctx["mode"], ctx["gap"], ctx["min_len"], 0.18 if ctx["scene"] else None)
        enabled = ctx["enabled"]
        rows, per = [], Counter()
        ctx["n_seq"], ctx["n_frames"] = len(seqs), sum(len(q) for q in seqs)
        ctx["n_files"] = len(ctx["frames"])
        for qi, seq in enumerate(seqs):
            name = _seq_name(seq)
            for f in check_sequence(seq, name, ctx["iou"], ctx["max_gap"], ctx["size"]):
                if f.code not in enabled:
                    continue
                per[f.code] += 1
                rows.append(ResultRow(f.key, (CODES[f.code], f.label, f"{f.frame + 1} / {len(seq)}", name, f.message),
                                      f.shape_index, payload=f, tone="warning" if f.severity == "warn" else "muted"))
        ctx["per"] = per
        ctx["frames"] = None                          # free the per-frame boxes
        return rows

    def summarize(self):
        c = self.ctx
        if not c or "n_seq" not in c:
            return super().summarize()
        per = ", ".join(f"{CODES[k].lower()} {v:,}" for k, v in c["per"].most_common())
        outside = c["n_files"] - c["n_frames"]
        return (f"{c['n_seq']:,} sequence(s) with {c['n_frames']:,} frames"
                + (f" ({outside:,} file(s) in no sequence of ≥ {c['min_len']} frames)" if outside else "")
                + f" — {len(self.rows):,} finding(s)" + (f": {per}" if per else ""))

    # ------------------------------------------------------------------ open / fix
    def on_row_activated(self, row):
        if row is None:
            return
        f = row.payload
        overlay = None
        if f is not None and f.code == "missing" and f.box is not None:
            sample = self.session.by_key.get(row.key)
            if sample is None:
                return
            ann = self.session.adapter.load(sample, light=True)
            if ann.normalized:
                size = self.session.adapter.image_size(sample, ann) or (1, 1)
            else:
                size = (ann.width or 1, ann.height or 1)
            x1, y1, x2, y2 = f.box
            box = (x1 * size[0], y1 * size[1], x2 * size[0], y2 * size[1])
            overlay = [{"box": box, "text": f"missing? {f.label}", "color": hexc("warn")}]
            self.app.open_sample(row.key, None, overlay, box)
            return
        self.app.open_sample(row.key, row.shape_index)

    def _add_missing(self):
        rows = [r for r in self.target_rows() if r.payload is not None and r.payload.code == "missing"]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) ‘Missing in this frame’ rows first.")
            return
        items = {}
        for r in rows:
            items.setdefault(r.key, []).append((r.payload.label, r.payload.box))
        if not self.confirm(f"Add {len(rows):,} interpolated box(es) to {len(items):,} file(s)?\n\nEach box is placed "
                            "halfway between the object's boxes in the neighbouring frames — check them afterwards "
                            "(the files are flagged). The originals are backed up (Undo)."):
            return
        st = self.session.add_boxes(self, items, what="sequence")
        if st["files"]:
            self._last_backup = st["backup"]
            self.undo_btn.setEnabled(True)
            self.session.set_review(st["changed"], REVIEW_FLAG, "Sequence Check: interpolated box added — check it")
        done = {(r.key, r.payload.box) for r in rows}
        self.remove_rows(lambda r: r.payload is not None and (r.key, r.payload.box) in done)
        lines = [f"Added {st['done']:,} box(es) in {st['files']:,} file(s); the files are flagged for a look."]
        if st["skipped"]:
            lines.append(f"{st['skipped']:,} skipped: the file already has that object there.")
        if st["refused"]:
            lines.append(f"{st['refused']:,} skipped: the label is not a class of this dataset.")
        self.report("Done", lines, st["errors"])

    def _undo(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last ‘Add missing "
                                         "boxes’?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.report("Undone", [f"Restored {len(keys):,} file(s)."] +
                    ([f"{len(skipped):,} changed since and left alone."] if skipped else []), [])


def _seq_name(seq) -> str:
    a, b = Path(seq[0].key), Path(seq[-1].key)
    return f"{a.parent.name}/{a.stem} … {b.stem}"


def _thumb(path):
    if path is None:
        return None
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            return None
        return cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        return None
