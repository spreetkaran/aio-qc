# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Classification Check (QC ▸ Labels, Ctrl+Alt+D) — everything that can be wrong with image-level classes, each with
the reason and a fix:

  · no class — and why: no flags at all, every flag false, a value that is not true / false, or an image that is
    not inside a class folder
  · several classes on an image of a one-class-per-image dataset
  · flag values written as 1, "true", "yes" … (read as set; Fix values writes true / false)
  · class names that differ only in letter case, spaces, _ / - or by a typo (Cat / cat, traffic_light / traffic
    light) — Merge into … puts them under one name
  · classes with fewer images than you need, and classes missing from a split (in train but not in val)
  · classes outside your class list (Label Mapping ▸ class list)

The dataset's mode (one class per image, or multi-label) is set here too. Every fix is one step of Undo (Ctrl+Z).
"""
from __future__ import annotations

import difflib
import os
from collections import Counter, defaultdict

from PyQt6.QtWidgets import QComboBox, QInputDialog, QMessageBox, QSpinBox

from AIO_QC.core.checks import normalize_label
from AIO_QC.core.dataset import CLASSIFICATION_TYPES
from .base_scan import BaseScanDialog, ResultRow

NO_CLASS, SEVERAL, VALUES, VARIANT, RARE, SPLIT, UNKNOWN = ("No class", "Several classes", "Flag values",
                                                           "Name variant", "Few images", "Missing from a split",
                                                           "Not in the class list")
TWICE = "Same image, other class"


def reason_no_class(ann, folders: bool) -> str:
    if folders:
        return "not inside a class folder"
    kinds = [k for k, *_r in ann.flag_issues]
    if "no_flags" in kinds:
        return "the file has no flags"
    raw = ann.raw if isinstance(ann.raw, dict) else {}
    flags = raw.get("flags") or {}
    if not flags:
        return "the flags are empty"
    if any(k == "value" for k in kinds):
        return "no flag is true (some values are not true / false: " + \
            ", ".join(f"{f}={v!r}" for k, f, v in ann.flag_issues if k == "value")[:120] + ")"
    return "every flag is false"


def variants(counts: Counter, cutoff: float = 0.85) -> dict:
    """{rarer name: more common name} for names that differ in case / spaces / _ - or by a small typo."""
    out = {}
    ranked = [n for n, _c in counts.most_common()]
    for i, n in enumerate(ranked):
        nn = normalize_label(n)
        for other in ranked[:i]:
            if other in out:
                continue
            no = normalize_label(other)
            if nn == no or (len(nn) > 3 and difflib.SequenceMatcher(None, nn, no).ratio() >= cutoff):
                out[n] = other
                break
    return out


def _same_image_other_class(by_size: dict) -> list:
    """The same picture (identical bytes) filed under different classes — a classifier learns both answers for it.
    Only files whose size collides are read (identical files have the same size)."""
    import hashlib
    rows = []
    for _size, items in by_size.items():
        if len(items) < 2 or len({t for _k, t, _p, _s in items}) < 2:
            continue
        by_hash = defaultdict(list)
        for it in items:
            try:
                with open(it[2], "rb") as f:
                    by_hash[hashlib.sha1(f.read()).hexdigest()].append(it)
            except OSError:
                continue
        for group in by_hash.values():
            if len({t for _k, t, _p, _s in group}) < 2:
                continue
            for k, t, p, sp in group:
                others = [f"{os.path.basename(p2)} ({', '.join(t2)})" for k2, t2, p2, _s2 in group if t2 != t]
                rows.append(ResultRow(k, (TWICE, ", ".join(t), sp, "identical to " + "; ".join(others[:3])),
                                      tone="error", payload={"kind": TWICE}))
    return rows


class ClassificationCheckDialog(BaseScanDialog):
    TITLE, ICON = "Classification Check", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("Everything that can be wrong with image-level classes: images without a class (and why), several "
            "classes on an image of a one-class-per-image dataset, flag values written as 1 or \"true\", class names "
            "that differ only in case, spacing or by a typo, classes with too few images or missing from a split, "
            "classes outside your class list, and the same picture (identical file) filed under different classes. "
            "Fix values, Set class… and Merge into… each are one step of Undo.")
    COLUMNS = ["Problem", "Classes", "Split", "Detail"]
    FILTER_COLUMN = 0
    AUTO_START = True

    def build_params(self, form):
        s = self.session
        self.mode = QComboBox()
        self.mode.addItem("One class per image", "single")
        self.mode.addItem("Several classes allowed (multi-label)", "multi")
        self.mode.setCurrentIndex(max(0, self.mode.findData(s.cls_mode)))
        self.mode.setEnabled(s.type.value == "cls_labelme")
        self.mode.setToolTip("Class folders hold one class per image. LabelMe flags can be either: multi-label "
                             "datasets do not report images with several classes, and C / 1 … 9 add and remove")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Dataset", self.mode)
        self.min_imgs = QSpinBox()
        self.min_imgs.setRange(0, 100000)
        self.min_imgs.setValue(int(s.cls_settings().get("min_images", 20)))
        self.min_imgs.setSuffix(" images")
        self.min_imgs.setToolTip("A class with fewer images than this is reported (a classifier needs examples)")
        self.min_imgs.valueChanged.connect(lambda v: (s.save_cls_settings(min_images=int(v)), self.start_scan()))
        form.addRow("Report classes with fewer than", self.min_imgs)
        self.add_action("Fix values", self.fix_values, "Rewrite flag values like 1 or \"true\" as true / false (the "
                                                       "class stays what it is read as)")
        self.add_action("Set class…", self.set_class, "Give the ticked (or selected) images a class")
        self.add_action("Merge into…", self.merge, "Put a name variant under the common name (every image)")

    def _mode_changed(self, _i):
        self.session.save_cls_settings(mode=self.mode.currentData())
        self.start_scan()

    def make_context(self):
        s = self.session
        return {"adapter": s.base_adapter, "folders": s.type.value == "cls_folders", "multi": s.cls_mode == "multi",
                "split_counts": defaultdict(Counter), "counts": Counter(), "keys_of": defaultdict(list),
                "truth": list(s.true_classes or []), "min": self.min_imgs.value(), "splits": list(s.splits or []),
                "by_size": defaultdict(list)}

    def scan_sample(self, ctx, s):
        try:
            ann = ctx["adapter"].load(s, light=True)
        except Exception as exc:
            return [ResultRow(s.key, ("Unreadable", "", s.split, str(exc)[:200]), tone="error")]
        tags = list(ann.tags or [])
        img = ann.image_path or s.image_path
        if img is not None and tags:                     # identical files have the same size: hashed later, only
            try:                                         # where sizes collide
                ctx["by_size"][os.path.getsize(img)].append((s.key, tuple(sorted(tags)), str(img), s.split))
            except OSError:
                pass
        for t in tags:
            ctx["counts"][t] += 1
            ctx["split_counts"][s.split][t] += 1
            ctx["keys_of"][t].append(s.key)
        rows = []
        if not tags:
            rows.append(ResultRow(s.key, (NO_CLASS, "", s.split, reason_no_class(ann, ctx["folders"])),
                                  tone="error", payload={"kind": NO_CLASS}))
        elif len(tags) > 1 and not ctx["multi"]:
            rows.append(ResultRow(s.key, (SEVERAL, ", ".join(tags), s.split,
                                          f"{len(tags)} classes on a one-class-per-image dataset"),
                                  tone="error", payload={"kind": SEVERAL}))
        bad = [(f, v) for k, f, v in ann.flag_issues if k == "value"]
        if bad and tags:
            rows.append(ResultRow(s.key, (VALUES, ", ".join(tags), s.split,
                                          "written as " + ", ".join(f"{f}={v!r}" for f, v in bad)[:160]),
                                  tone="warning", payload={"kind": VALUES}))
        return rows

    def finalize(self, ctx):
        rows = []
        counts = ctx["counts"]
        for rare, common in variants(counts).items():
            for k in ctx["keys_of"][rare]:
                rows.append(ResultRow(k, (VARIANT, rare, "", f"“{rare}” ({counts[rare]:,}) looks like “{common}” "
                                                            f"({counts[common]:,})"),
                                      tone="warning", payload={"kind": VARIANT, "from": rare, "to": common}))
        for c, n in sorted(counts.items()):
            if ctx["min"] and n < ctx["min"]:
                rows.append(ResultRow(ctx["keys_of"][c][0], (RARE, c, "", f"{n:,} image(s) — fewer than "
                                                                         f"{ctx['min']:,}"),
                                      tone="info", payload={"kind": RARE, "class": c}))
        splits = [x for x in ctx["splits"] if x]
        if len(splits) > 1:
            for c in sorted(counts):
                missing = [x for x in splits if not ctx["split_counts"][x][c]]
                if missing:
                    rows.append(ResultRow(ctx["keys_of"][c][0], (SPLIT, c, ", ".join(missing),
                                                                 f"no image of “{c}” in {', '.join(missing)}"),
                                          tone="warning", payload={"kind": SPLIT, "class": c}))
        rows += _same_image_other_class(ctx["by_size"])
        if ctx["truth"]:
            truth = set(ctx["truth"])
            for c in sorted(set(counts) - truth):
                for k in ctx["keys_of"][c]:
                    rows.append(ResultRow(k, (UNKNOWN, c, "", "not in the class list"), tone="warning",
                                          payload={"kind": UNKNOWN, "from": c}))
        return rows

    def summarize(self):
        if not self.rows:
            return super().summarize()
        c = Counter(r.cells[0] for r in self.rows if r)
        return " · ".join(f"{k}: {n:,}" for k, n in c.most_common())

    def on_row_activated(self, row):
        if row is not None and row.key in self.session.by_key:
            self.app.open_sample(row.key, None)

    # ------------------------------------------------------------------ fixes
    def _keys(self, kinds=None):
        rows = [r for r in self.target_rows() if r and r.payload and (kinds is None or r.payload["kind"] in kinds)]
        return list(dict.fromkeys(r.key for r in rows if r.key in self.session.by_key))

    def fix_values(self):
        keys = self._keys({VALUES}) or [r.key for r in self.rows if r and r.payload and r.payload["kind"] == VALUES]
        if not keys:
            QMessageBox.information(self, self.TITLE, "No flag values to fix.")
            return
        done, errors = self.session.set_tags(keys, add=(), remove=(), what="flag values → true / false", force=True)
        self.report("Fix values", [f"{len(done):,} file(s) rewritten (Ctrl+Z undoes it)."], errors)
        self.start_scan()

    def set_class(self):
        keys = self._keys()
        if not keys:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first.")
            return
        from .class_picker import ClassPicker
        s = self.session
        cur = Counter(t for k in keys for t in s.tags_of(k))
        dlg = ClassPicker(self, s.cls_classes(), cur, len(keys), multi=s.cls_mode == "multi",
                          counts=s.label_totals())
        if not dlg.exec() or dlg.result is None:
            return
        if dlg.result[0] == "set":
            done, errors = s.set_tags(keys, tags=dlg.result[1], what=f"class → {dlg.result[1][0]}")
        else:
            done, errors = s.set_tags(keys, add=dlg.result[1], remove=dlg.result[2], what="classes changed")
        if errors:
            self.report("Set class", [f"{len(done):,} image(s) changed."], errors)
        self.start_scan()

    def merge(self):
        s = self.session
        rows = [r for r in self.target_rows() if r and r.payload and r.payload["kind"] in (VARIANT, UNKNOWN)]
        if not rows:
            rows = [r for r in self.rows if r and r.payload and r.payload["kind"] in (VARIANT, UNKNOWN)]
        names = list(dict.fromkeys(r.payload["from"] for r in rows))
        if not names:
            QMessageBox.information(self, self.TITLE, "No name variants to merge.")
            return
        done_all, err_all = [], []
        for name in names:
            sugg = next((r.payload.get("to") for r in rows if r.payload["from"] == name and r.payload.get("to")), "")
            classes = [c for c in s.cls_classes() if c != name]
            if sugg in classes:
                classes.remove(sugg)
                classes.insert(0, sugg)
            target, ok = QInputDialog.getItem(self, self.TITLE, f"Put every image of “{name}” under:", classes, 0,
                                              True)
            if not ok or not target.strip():
                continue
            keys = [k for k in s.by_key if name in s.tags_of(k)]
            d, e = s.set_tags(keys, add=[target.strip()], remove=[name], what=f"{name} → {target.strip()}") \
                if s.type.value != "cls_folders" else s.set_tags(keys, tags=[target.strip()],
                                                                 what=f"{name} → {target.strip()}")
            done_all += d
            err_all += e
        if done_all or err_all:
            self.report("Merge", [f"{len(done_all):,} image(s) moved under the common name (Ctrl+Z undoes it)."],
                        err_all)
        self.start_scan()
