# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The classifier versions of the QC checks that were written for boxes. On a classification dataset (class folders,
LabelMe flags) the same toolbar buttons open these, each asking the question that tool asks — of whole images and
their classes:

  Label Ambiguities   class names that are one class written two ways (case, spaces, typos, plurals), names outside
                      the class list, classes used by a single image — Merge into… / Rename…
  Text Labels         the class names as text: spaces at the ends, invisible characters, characters a folder name
                      cannot hold, separators and letter case unlike the other names — Rename to the suggestion
  Duplicates          the same picture twice (exact / near copy), and above all a copy filed under ANOTHER class
  Outliers            images unusual for their class: size, shape, brightness, contrast, colour, file size
  Sequence Check      frames of one video in order: a class that flips for a frame or two and flips back
  Dim / Area          images outside the width / height / area / aspect ranges you set
  Training Size       what the classifier's transform does to each image at imgsz: enlarged (blurry), centre crop
                      cutting much of the picture, squashed — and per class the imgsz that keeps them sharp
  OCR Check           the text in the whole image read by OCR and compared with its class (digits, letters, signs)
  Validation          class folders: images outside any class folder, files that are not images, unreadable or
                      empty files, wrong extensions, empty class folders, folders nested in class folders
  Pairs               class folders: annotation files left next to the images (or without one), one file name in
                      several class folders of a split

Every fix goes through Image QC (moves / flag rewrites, File History, one step of Undo).
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QComboBox, QInputDialog, QMessageBox

from AIO_QC.core.checks import normalize_label
from AIO_QC.core.dataset import CLASSIFICATION_TYPES, DatasetType
from .base_scan import BaseScanDialog, ResultRow, dspin, ispin


# ====================================================================== shared fixes
class _ClsFixes:
    """Set class / merge / rename for classification rows (every change one step of Undo)."""

    def _row_keys(self, rows=None):
        rows = self.target_rows() if rows is None else rows
        return list(dict.fromkeys(r.key for r in rows if r and r.key in self.session.by_key))

    def set_class(self):
        keys = self._row_keys()
        if not keys:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first.")
            return
        from .class_picker import ClassPicker
        s = self.session
        cur = Counter(t for k in keys for t in s.tags_of(k))
        dlg = ClassPicker(self, s.cls_classes(), cur, len(keys), multi=s.cls_mode == "multi", counts=s.label_totals())
        if not dlg.exec() or dlg.result is None:
            return
        if dlg.result[0] == "set":
            done, errors = s.set_tags(keys, tags=dlg.result[1], what=f"class → {dlg.result[1][0]}")
        else:
            done, errors = s.set_tags(keys, add=dlg.result[1], remove=dlg.result[2], what="classes changed")
        self.report(self.TITLE, [f"{len(done):,} image(s) changed (Ctrl+Z undoes it)."], errors)
        self.start_scan()

    def rename_class(self, old: str, new: str) -> tuple[list, list]:
        from .cls_classes_dialog import MERGE, RENAME, apply_class_changes
        s = self.session
        act = MERGE if new in s.cls_classes() else RENAME
        return apply_class_changes(s, {old: (act, new)})

    def _names_of_rows(self, rows):
        return list(dict.fromkeys(r.payload["class"] for r in rows if r and isinstance(r.payload, dict)
                                  and r.payload.get("class")))

    def merge_into(self):
        rows = [r for r in self.target_rows() if r and isinstance(r.payload, dict) and r.payload.get("class")]
        names = self._names_of_rows(rows)
        if not names:
            QMessageBox.information(self, self.TITLE, "Tick (or select) the classes to merge first.")
            return
        s = self.session
        done_all, err_all = [], []
        for name in names:
            sugg = next((r.payload.get("to") for r in rows if r.payload["class"] == name and r.payload.get("to")), "")
            classes = [c for c in s.cls_classes() if c != name]
            if sugg in classes:
                classes.remove(sugg)
            classes.insert(0, sugg) if sugg else None
            target, ok = QInputDialog.getItem(self, self.TITLE, f"Put every image of “{name}” under:", classes, 0, True)
            if not ok or not target.strip() or target.strip() == name:
                continue
            d, e = self.rename_class(name, target.strip())
            done_all += d
            err_all += e
        if done_all or err_all:
            self.report(self.TITLE, [f"{len(done_all):,} image(s) moved under the new name (Ctrl+Z undoes it)."],
                        err_all)
        self.start_scan()

    def rename_to_suggestion(self):
        rows = [r for r in self.target_rows() if r and isinstance(r.payload, dict) and r.payload.get("to")]
        plan = {}
        for r in rows:
            plan.setdefault(r.payload["class"], r.payload["to"])
        if not plan:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows with a suggestion first.")
            return
        if not self.confirm("Rename\n" + "\n".join(f"  “{a}” → “{b}”" for a, b in list(plan.items())[:12]) +
                            ("\n  …" if len(plan) > 12 else "") + "\n\n(each one step of Undo)"):
            return
        done_all, err_all = [], []
        for a, b in plan.items():
            d, e = self.rename_class(a, b)
            done_all += d
            err_all += e
        self.report(self.TITLE, [f"{len(plan)} class(es) renamed · {len(done_all):,} image(s) changed."], err_all)
        self.start_scan()

    def rename_one(self):
        rows = [r for r in self.target_rows() if r and isinstance(r.payload, dict) and r.payload.get("class")]
        names = self._names_of_rows(rows)
        if len(names) != 1:
            QMessageBox.information(self, self.TITLE, "Select one class to rename.")
            return
        old = names[0]
        sugg = next((r.payload.get("to") for r in rows if r.payload.get("to")), old)
        new, ok = QInputDialog.getText(self, self.TITLE, f"New name for “{old}”:", text=sugg)
        if not ok or not new.strip() or new.strip() == old:
            return
        d, e = self.rename_class(old, new.strip())
        self.report(self.TITLE, [f"“{old}” → “{new.strip()}”: {len(d):,} image(s)."], e)
        self.start_scan()

    def open_in_gallery(self, cls: str | None = None):
        rows = self.target_rows()
        if cls is None:
            names = self._names_of_rows(rows) or [t for r in rows[:1] if r for t in self.session.tags_of(r.key)]
            cls = names[0] if names else None
        self.app.open_tool("class_gallery")
        g = self.app.open_dialogs.get("class_gallery")
        if g is not None and cls:
            g.show_class(cls)

    def on_row_activated(self, row):
        if row is not None and row.key in self.session.by_key:
            self.app.open_sample(row.key, None)


def _reveal(folder: Path):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def _class_of(session, key) -> str:
    t = session.tags_of(key)
    return ", ".join(t) if t else "(no class)"


def _image_of(ctx, s):
    p = s.image_path
    if p is None or not Path(p).exists():
        try:
            p = ctx["adapter"].load(s, light=True).image_path
        except Exception:                               # noqa: BLE001
            p = None
    return Path(p) if p else None


# ====================================================================== 1. label ambiguities
class ClsAmbiguityDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Label Ambiguities (classes)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("Class names that are one class written two ways — letter case, spaces, _ / -, a typo, a plural — names "
            "outside your class list, and classes only one or two images use (often a slip). Merge into… puts every "
            "image of a name under another; Rename… renames a class. Each is one step of Undo.")
    COLUMNS = ["Problem", "Class", "Images", "Detail"]
    FILTER_COLUMN = 0
    AUTO_START = True

    def build_params(self, form):
        self.sim = dspin(0.6, 1.0, 0.85, 2, 0.01)
        self.sim.setToolTip("How alike two names must be to count as one class written two ways (1 = only case / "
                            "spacing differences)")
        self.sim.valueChanged.connect(lambda _v: self.start_scan())
        form.addRow("Similar names from", self.sim)
        self.rare = ispin(0, 1000, 2)
        self.rare.setToolTip("Classes with at most this many images are listed (0 = off)")
        self.rare.valueChanged.connect(lambda _v: self.start_scan())
        form.addRow("Classes used by at most", self.rare)
        self.add_action("Merge into…", self.merge_into, "Put every image of the ticked class under another class")
        self.add_action("Rename…", self.rename_one, "Rename the selected class (every image)")
        self.add_action("Class Gallery", self.open_in_gallery, "Look at the selected class's images")

    def make_context(self):
        return {"counts": Counter(), "keys_of": defaultdict(list), "sim": self.sim.value(),
                "truth": list(self.session.true_classes or []), "rare": self.rare.value()}

    def scan_sample(self, ctx, s):
        for t in self.session.tags_of(s.key):
            ctx["counts"][t] += 1
            ctx["keys_of"][t].append(s.key)
        return []

    def finalize(self, ctx):
        from .classification_check_dialog import variants
        counts, rows, seen = ctx["counts"], [], set()

        def row(kind, name, detail, to=None, tone="warning"):
            seen.add((kind, name))
            rows.append(ResultRow(ctx["keys_of"][name][0], (kind, name, counts[name], detail), tone=tone,
                                  payload={"class": name, "to": to}))
        def singular(n):
            k = normalize_label(n)
            return k[:-2] if k.endswith("es") and len(k) > 4 else k[:-1] if k.endswith("s") and len(k) > 3 else k
        for rare, common in variants(counts, ctx["sim"]).items():
            same = normalize_label(rare) == normalize_label(common)
            plural = not same and singular(rare) == singular(common)
            if plural:
                row("Singular / plural", rare, f"“{rare}” ({counts[rare]:,}) and “{common}” ({counts[common]:,}) are "
                    "probably one class", common)
                continue
            row("Written two ways" if same else "Similar name", rare,
                f"“{rare}” ({counts[rare]:,}) looks like “{common}” ({counts[common]:,})", common)
        stem = {}
        for n in counts:
            stem.setdefault(singular(n), []).append(n)
        for group in stem.values():
            if len(group) > 1:
                group = sorted(group, key=lambda n: -counts[n])
                for n in group[1:]:
                    if not any((k, n) in seen for k in ("Written two ways", "Similar name", "Singular / plural")):
                        row("Singular / plural", n, f"“{n}” and “{group[0]}” ({counts[group[0]]:,}) are probably one "
                            "class", group[0])
        if ctx["truth"]:
            truth = set(ctx["truth"])
            for n in sorted(set(counts) - truth):
                row("Not in the class list", n, "not one of the classes in your class list")
        if ctx["rare"]:
            for n, c in sorted(counts.items()):
                if c <= ctx["rare"] and len(counts) > 1:
                    row("Rarely used", n, f"only {c} image(s) — a slip, or a class that needs more examples",
                        tone="info")
        return rows

    def summarize(self):
        live = [r for r in self.rows if r]
        if not live:
            return "Every class name is written one way."
        c = Counter(r.cells[0] for r in live)
        return " · ".join(f"{k}: {n}" for k, n in c.most_common())


# ====================================================================== 2. class-name text
BAD_FOLDER_CHARS = set('<>:"/\\|?*')
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", " ", " ", " "}


def _sep_of(n: str) -> str | None:
    seps = [c for c in (" ", "_", "-") if c in n.strip()]
    return seps[0] if len(seps) == 1 else ("mixed" if seps else None)


def _case_of(n: str) -> str | None:
    letters = [c for c in n if c.isalpha()]
    if len(letters) < 2:
        return None
    if all(c.islower() for c in letters):
        return "lower"
    if all(c.isupper() for c in letters):
        return "UPPER"
    words = [w for w in re.split(r"[ _\-]+", n.strip()) if w and w[0].isalpha()]
    if words and all(w[0].isupper() and (w[1:].islower() or not w[1:]) for w in words):
        return "Title"
    return "mixed"


def _styled(n: str, sep: str | None, case: str | None) -> str:
    words = [w for w in re.split(r"[ _\-]+", "".join(c for c in n if c not in INVISIBLE).strip()) if w]
    if case == "lower":
        words = [w.lower() for w in words]
    elif case == "UPPER":
        words = [w.upper() for w in words]
    elif case == "Title":
        words = [w[:1].upper() + w[1:].lower() for w in words]
    out = (sep if sep in (" ", "_", "-") else (_sep_of(n) if _sep_of(n) in (" ", "_", "-") else "_")).join(words)
    return "".join("_" if c in BAD_FOLDER_CHARS else c for c in out).rstrip(". ")


def name_problems(names) -> dict:
    """{name: [(problem, detail, suggestion or None)]} for a dataset's class names."""
    seps = Counter(s for s in (_sep_of(n) for n in names) if s in (" ", "_", "-"))
    cases = Counter(c for c in (_case_of(n) for n in names) if c and c != "mixed")
    sep = seps.most_common(1)[0][0] if seps and seps.most_common(1)[0][1] >= max(2, 0.6 * sum(seps.values())) \
        else None
    case = cases.most_common(1)[0][0] if cases and cases.most_common(1)[0][1] >= max(2, 0.6 * sum(cases.values())) \
        else None
    label = {" ": "spaces", "_": "_", "-": "-"}
    out = {}
    for n in names:
        p = []
        fix = _styled(n, sep, case)
        if n != n.strip():
            p.append(("Spaces at the ends", f"“{n}” starts or ends with a space", n.strip()))
        if "  " in n:
            p.append(("Double spaces", "two spaces in a row", re.sub(r" {2,}", " ", n.strip())))
        inv = sorted({c for c in n if c in INVISIBLE or unicodedata.category(c) in ("Cc", "Cf")})
        if inv:
            p.append(("Invisible characters", "contains " + ", ".join(f"U+{ord(c):04X}" for c in inv),
                      "".join(c for c in n if c not in inv)))
        bad = sorted({c for c in n if c in BAD_FOLDER_CHARS})
        if bad or n.upper().split(".")[0] in RESERVED or n.endswith((".", " ")):
            p.append(("Not a valid folder name", ("contains " + " ".join(bad)) if bad else
                      "a reserved or trailing-dot name — Windows cannot make that folder", fix))
        s_ = _sep_of(n)
        if s_ == "mixed":
            p.append(("Mixed separators", "spaces, _ and - in one name", fix))
        elif sep and s_ and s_ != sep:
            p.append(("Separator unlike the others", f"uses {label[s_]} where the other names use {label[sep]}",
                      fix))
        c_ = _case_of(n)
        if case and c_ and c_ != case:
            p.append(("Letter case unlike the others", f"{c_} where the other names are {case}", fix))
        if len(n) > 40:
            p.append(("Very long name", f"{len(n)} characters — hard to read in charts and folder lists", None))
        if p:
            out[n] = [(a, b, c if c and c != n else None) for a, b, c in p]
    return out


class ClsTextDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Class Name Text", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("A classifier's labels are its class names — so their text is checked: spaces at the ends, double spaces, "
            "invisible characters (copied from a web page), characters a folder name cannot hold, mixed separators, "
            "and a separator or letter case unlike the other names. Rename to suggestion fixes the ticked names "
            "(class folders are renamed by moving the images; one step of Undo each).")
    COLUMNS = ["Problem", "Class", "Images", "Detail", "Suggestion"]
    FILTER_COLUMN = 0
    AUTO_START = True
    REVIEW_ACTIONS = False

    def build_params(self, form):
        self.add_action("Rename to suggestion", self.rename_to_suggestion, "Rename the ticked classes as suggested")
        self.add_action("Rename…", self.rename_one, "Rename the selected class")

    def make_context(self):
        return {"counts": Counter(), "first": {}}

    def scan_sample(self, ctx, s):
        for t in self.session.tags_of(s.key):
            ctx["counts"][t] += 1
            ctx["first"].setdefault(t, s.key)
        return []

    def finalize(self, ctx):
        names = sorted(set(ctx["counts"]) | set(self.session.cls_classes()))
        rows = []
        for n, probs in sorted(name_problems(names).items()):
            key = ctx["first"].get(n) or next(iter(self.session.by_key), "")
            for kind, detail, fix in probs:
                rows.append(ResultRow(key, (kind, n, ctx["counts"].get(n, 0), detail, fix or ""),
                                      tone="error" if kind == "Not a valid folder name" else "warning",
                                      payload={"class": n, "to": fix}))
        return rows

    def summarize(self):
        live = [r for r in self.rows if r]
        return (f"{len({r.cells[1] for r in live})} class name(s) to tidy" if live
                else "Every class name is clean text.")


# ====================================================================== 3. duplicates
def _dup_base():
    from .qc_basic import ImageDuplicatesDialog
    return ImageDuplicatesDialog


class ClsDuplicatesDialog(_ClsFixes, _dup_base()):
    TITLE, ICON = "Duplicate Images (classes)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("The same picture more than once — identical files and near copies (re-encoded, resized, slightly "
            "brighter). A copy filed under ANOTHER class teaches the classifier two answers for one picture: give "
            "the copies the kept copy's class, or trash them. Copies across splits leak the answer into val / test.")
    COLUMNS = ["Group", "Kind", "Distance", "Splits in group", "Leakage", "Class vs kept copy"]

    def build_params(self, form):
        super().build_params(form)
        self.add_action("Give copies the kept class", self.use_kept_class,
                        "Every ticked copy filed under another class gets the class of its group's kept copy")
        self.add_action("Set class…", self.set_class, "Give the ticked images a class")

    def finalize(self, ctx):
        rows = super().finalize(ctx)
        s = self.session
        kept = {}
        for r in rows:
            if r.payload[1] == 0:
                kept[r.payload[0]] = s.tags_of(r.key)
        out = []
        for r in rows:
            mine, k = s.tags_of(r.key), kept.get(r.payload[0]) or []
            if r.payload[1] == 0:
                vs = "kept · " + (", ".join(mine) or "no class")
            elif sorted(mine) == sorted(k):
                vs = "same class"
            else:
                vs = f"⚠ OTHER class: {', '.join(mine) or 'none'} (kept: {', '.join(k) or 'none'})"
            cells = r.cells[:5] + (vs,)
            tone = "error" if vs.startswith("⚠") or r.cells[4] else None
            out.append(ResultRow(r.key, cells, payload=r.payload, tone=tone))
        return out

    def summarize(self):
        live = [r for r in self.rows if r]
        other = sum(1 for r in live if str(r.cells[5]).startswith("⚠"))
        return super().summarize() + f" · {other} copy(ies) under another class"

    def use_kept_class(self):
        s = self.session
        kept = {r.payload[0]: s.tags_of(r.key) for r in self.rows if r and r.payload[1] == 0}
        plan = defaultdict(list)
        for r in self.target_rows() or [r for r in self.rows if r and str(r.cells[5]).startswith("⚠")]:
            k = kept.get(r.payload[0])
            if r and r.payload[1] > 0 and k and sorted(s.tags_of(r.key)) != sorted(k):
                plan[tuple(k)].append(r.key)
        if not plan:
            QMessageBox.information(self, self.TITLE, "No copies under another class among the ticked rows.")
            return
        done_all, err_all = [], []
        for tags, keys in plan.items():
            d, e = s.set_tags(keys, tags=list(tags), what=f"class → {', '.join(tags)} (as its copy)")
            done_all += d
            err_all += e
        self.report(self.TITLE, [f"{len(done_all):,} copy(ies) now have their kept copy's class (Ctrl+Z undoes "
                                 "it)."], err_all)
        self.start_scan()


# ====================================================================== 4. outliers
FEATURES = (("size", "Image size", "{:.0f} px²"), ("aspect", "Shape (width / height)", "{:.2f}"),
            ("bright", "Brightness", "{:.0f}"), ("contrast", "Contrast", "{:.0f}"),
            ("colour", "Colourfulness", "{:.0f}"), ("kb", "File size", "{:.0f} KB"))


def image_features(path) -> dict | None:
    """Cheap per-image numbers a class should be consistent in."""
    import cv2
    from AIO_QC.core.io import image_size
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if not buf.size:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None or img.size == 0:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    wh = image_size(str(path)) or (img.shape[1] * 4, img.shape[0] * 4)
    w, h = (wh[0], wh[1]) if wh and wh[0] else (img.shape[1], img.shape[0])
    f = img.astype(np.float32)
    gray = f.mean(2)
    rg = f[:, :, 2] - f[:, :, 1]
    yb = 0.5 * (f[:, :, 2] + f[:, :, 1]) - f[:, :, 0]
    colour = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {"size": float(w * h), "aspect": w / max(1, h), "bright": float(gray.mean()),
            "contrast": float(gray.std()), "colour": colour, "kb": buf.size / 1024.0, "wh": (int(w), int(h))}


class ClsOutliersDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Outliers (classes)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("Images that are unusual FOR THEIR CLASS: much bigger / smaller, another shape, far darker or brighter, "
            "flat or harsh, colour among grey images (or the reverse), a tiny or huge file. Often a wrong class, a "
            "broken image or a different source. Robust score: distance from the class's median in MADs (3.5 is "
            "unusual). Class Gallery ▸ Least typical looks at the picture itself.")
    COLUMNS = ["Class", "Unusual in", "Value", "Class typical", "Score"]
    FILTER_COLUMN = 1
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.thr = dspin(2.0, 20.0, 3.5, 1, 0.5)
        self.thr.setToolTip("How far from the class's typical image (robust z-score) counts as unusual")
        form.addRow("Score at least", self.thr)
        self.min_n = ispin(3, 10000, 8)
        self.min_n.setToolTip("Classes with fewer images are skipped (too few to know what is typical)")
        form.addRow("Classes with at least", self.min_n)
        self.add_action("Set class…", self.set_class, "Give the ticked images another class")
        self.add_action("Class Gallery", self.open_in_gallery, "Look at the selected image's class")

    def make_context(self):
        return {"adapter": self.session.adapter, "feats": [], "thr": self.thr.value(), "min": self.min_n.value()}

    def scan_sample(self, ctx, s):
        tags = self.session.tags_of(s.key)
        p = _image_of(ctx, s)
        if not tags or p is None:
            return []
        f = image_features(p)
        if f is not None:
            ctx["feats"].append((s.key, tags[0], f))
        return []

    def finalize(self, ctx):
        by = defaultdict(list)
        for k, c, f in ctx["feats"]:
            by[c].append((k, f))
        rows = []
        for c, items in by.items():
            if len(items) < ctx["min"]:
                continue
            for feat, title, fmt in FEATURES:
                v = np.array([f[feat] for _k, f in items], float)
                x = np.log(np.maximum(v, 1e-6)) if feat in ("size", "aspect", "kb") else v
                med = float(np.median(x))
                mad = float(np.median(np.abs(x - med))) or (float(np.std(x)) / 1.4826 if np.std(x) else 0.0)
                if not mad:
                    continue
                z = 0.6745 * (x - med) / mad
                typical = float(np.exp(med)) if feat in ("size", "aspect", "kb") else med
                for (k, f), zi in zip(items, z):
                    if abs(zi) >= ctx["thr"]:
                        val = f[feat]
                        shown = f"{f['wh'][0]}×{f['wh'][1]}" if feat == "size" else fmt.format(val)
                        typ = (f"≈{int(np.sqrt(typical * 1.0))}² px" if feat == "size" else fmt.format(typical))
                        word = ("higher" if zi > 0 else "lower")
                        rows.append(ResultRow(k, (c, title, shown, typ, round(float(zi), 1)),
                                              tone="warning" if abs(zi) < 2 * ctx["thr"] else "error",
                                              payload={"class": c, "feature": feat, "z": float(zi), "dir": word}))
        rows.sort(key=lambda r: -abs(r.payload["z"]))
        ctx["feats"] = []
        return rows

    def summarize(self):
        live = [r for r in self.rows if r]
        return (f"{len({r.key for r in live}):,} unusual image(s) in {len({r.cells[0] for r in live})} class(es)"
                if live else "No image is unusual for its class.")


# ====================================================================== 5. sequence
FRAME_RE = re.compile(r"^(.*?)(\d+)$")


class ClsSequenceDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Sequence Check (classes)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("Frames of one video (file names that differ only in a frame number, in any class folder) put in order: "
            "a class that flips for a frame or two and flips back (cat, cat, DOG, cat, cat) is almost always a slip — "
            "Use the neighbours' class fixes the ticked frames. Frames without a class inside a run are listed too.")
    COLUMNS = ["Problem", "Sequence", "Frame", "Class", "Neighbours", "Detail"]
    FILTER_COLUMN = 0

    def build_params(self, form):
        self.flip = ispin(1, 20, 2)
        self.flip.setToolTip("A run of another class this long or shorter, with the same class on both sides, is "
                             "reported")
        form.addRow("Flips of at most (frames)", self.flip)
        self.min_seq = ispin(3, 100000, 5)
        form.addRow("Sequences of at least (frames)", self.min_seq)
        self.add_action("Use the neighbours' class", self.use_neighbours,
                        "The ticked frames get the class of the frames around them")
        self.add_action("Set class…", self.set_class, "Give the ticked frames a class")

    def make_context(self):
        return {"frames": defaultdict(list), "L": self.flip.value(), "min": self.min_seq.value()}

    def scan_sample(self, ctx, s):
        stem = os.path.splitext(s.name)[0]
        m = FRAME_RE.match(stem)
        if m:
            ctx["frames"][(s.split, m.group(1).rstrip("_-. ") or "(numbers)")].append(
                (int(m.group(2)), s.key, tuple(self.session.tags_of(s.key))))
        return []

    def finalize(self, ctx):
        rows, L = [], ctx["L"]
        for (split, seq), frames in ctx["frames"].items():
            if len(frames) < ctx["min"]:
                continue
            frames.sort()
            runs = []                                      # [class tuple, start, end] over frame positions
            for i, (_n, _k, t) in enumerate(frames):
                if runs and runs[-1][0] == t:
                    runs[-1][2] = i
                else:
                    runs.append([t, i, i])
            name = f"{seq}" + (f" [{split}]" if split else "")
            for j in range(1, len(runs) - 1):
                t, a, b = runs[j]
                before, after = runs[j - 1], runs[j + 1]
                if before[0] != after[0] or b - a + 1 > L or before[2] - before[1] + 1 <= L \
                        or after[2] - after[1] + 1 <= L:
                    continue
                around = ", ".join(before[0]) or "no class"
                kind = "Frame without a class" if not t else "Brief flip"
                for i in range(a, b + 1):
                    n, k, tt = frames[i]
                    rows.append(ResultRow(k, (kind, name, n, ", ".join(tt) or "(none)", around,
                                              f"{b - a + 1} frame(s) of “{', '.join(tt) or 'none'}” between "
                                              f"{before[2] - before[1] + 1} and {after[2] - after[1] + 1} of "
                                              f"“{around}”"),
                                          tone="error", payload={"to": list(before[0])}))
        ctx["frames"] = {}
        return rows

    def summarize(self):
        live = [r for r in self.rows if r]
        return (f"{len(live):,} frame(s) whose class flips briefly in {len({r.cells[1] for r in live})} sequence(s)"
                if live else "No brief class flips in the frame sequences.")

    def use_neighbours(self):
        rows = [r for r in (self.target_rows() or [r for r in self.rows if r]) if r and r.payload.get("to")]
        plan = defaultdict(list)
        for r in rows:
            plan[tuple(r.payload["to"])].append(r.key)
        if not plan:
            QMessageBox.information(self, self.TITLE, "Nothing to fix.")
            return
        done_all, err_all = [], []
        for tags, keys in plan.items():
            d, e = self.session.set_tags(keys, tags=list(tags), what=f"class → {', '.join(tags)} (as its neighbours)")
            done_all += d
            err_all += e
        self.report(self.TITLE, [f"{len(done_all):,} frame(s) now have their neighbours' class (Ctrl+Z undoes it)."],
                    err_all)
        self.start_scan()


# ====================================================================== 6. dim / area
class ClsDimAreaDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Dim / Area (images)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("A classifier sees whole images: images outside the width / height / area / shape ranges you set — too "
            "small to show the class, huge panoramas, thumbnails mixed into photos. 0 = no limit.")
    COLUMNS = ["Problem", "Class", "Width", "Height", "Megapixels", "Aspect"]
    FILTER_COLUMN = 0

    def build_params(self, form):
        self.min_w, self.max_w = ispin(0, 100000, 64), ispin(0, 100000, 0)
        self.min_h, self.max_h = ispin(0, 100000, 64), ispin(0, 100000, 0)
        self.min_mp, self.max_mp = dspin(0, 1000, 0, 3, 0.01), dspin(0, 1000, 0, 2, 1)
        self.max_ar = dspin(0, 100, 3.0, 2, 0.1)
        for lab, a, b in (("Width (px)", self.min_w, self.max_w), ("Height (px)", self.min_h, self.max_h),
                          ("Area (megapixels)", self.min_mp, self.max_mp)):
            from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(QLabel("from"))
            hl.addWidget(a)
            hl.addWidget(QLabel("to"))
            hl.addWidget(b)
            hl.addStretch(1)
            form.addRow(lab, w)
        self.max_ar.setToolTip("Longer side / shorter side above this is reported (0 = off)")
        form.addRow("Longer / shorter side at most", self.max_ar)
        self.cls_box = QComboBox()
        self.cls_box.addItem("All classes", "")
        for c in self.session.cls_classes():
            self.cls_box.addItem(c, c)
        form.addRow("Class", self.cls_box)
        self.add_action("Set class…", self.set_class, "Give the ticked images a class")

    def make_context(self):
        return {"adapter": self.session.adapter, "w": (self.min_w.value(), self.max_w.value()),
                "h": (self.min_h.value(), self.max_h.value()), "mp": (self.min_mp.value(), self.max_mp.value()),
                "ar": self.max_ar.value(), "cls": self.cls_box.currentData()}

    def scan_sample(self, ctx, s):
        from AIO_QC.core.io import image_size
        tags = self.session.tags_of(s.key)
        if ctx["cls"] and ctx["cls"] not in tags:
            return []
        p = _image_of(ctx, s)
        wh = image_size(str(p)) if p is not None else None
        if not wh or not wh[0]:
            return [ResultRow(s.key, ("Size unknown", ", ".join(tags), "", "", "", ""), tone="error")]
        w, h = wh[0], wh[1]
        mp, ar = w * h / 1e6, max(w, h) / max(1, min(w, h))
        probs = []
        for name, v, (lo, hi), unit in (("Width", w, ctx["w"], " px"), ("Height", h, ctx["h"], " px"),
                                        ("Area", mp, ctx["mp"], " MP")):
            if lo and v < lo:
                probs.append(f"{name} below {lo:g}{unit}")
            if hi and v > hi:
                probs.append(f"{name} above {hi:g}{unit}")
        if ctx["ar"] and ar > ctx["ar"]:
            probs.append(f"Shape: {ar:.1f} : 1")
        return [ResultRow(s.key, (p_, ", ".join(tags), w, h, round(mp, 3), round(w / max(1, h), 2)),
                          tone="warning") for p_ in probs]

    def summarize(self):
        live = [r for r in self.rows if r]
        return f"{len({r.key for r in live}):,} image(s) outside the ranges" if live else \
            "Every image is inside the ranges."


# ====================================================================== 7. training size
class ClsTrainSizeDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Training Size (classifier)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("What a classifier's transform does to each image at the training size (imgsz). Centre crop (the usual "
            "classification transform): the shorter side becomes imgsz and the middle square is kept — long images "
            "lose their ends, small images are enlarged (blurry). Stretch: the whole image squeezed to a square. Per "
            "class: how many images are enlarged or heavily cropped, and the imgsz that keeps 95 % of them sharp.")
    COLUMNS = ["Problem", "Class", "Image size", "At training size", "Detail"]
    FILTER_COLUMN = 0

    def build_params(self, form):
        self.imgsz = ispin(16, 4096, 224)
        self.imgsz.setSingleStep(32)
        form.addRow("Training size (imgsz)", self.imgsz)
        self.mode = QComboBox()
        self.mode.addItem("Centre crop (shorter side → imgsz)", "crop")
        self.mode.addItem("Stretch whole image to imgsz × imgsz", "stretch")
        form.addRow("Transform", self.mode)
        self.up = dspin(1.0, 20.0, 2.0, 1, 0.5)
        self.up.setToolTip("Enlarged more than this many times: the class's details are interpolated, not seen")
        form.addRow("Report enlarged more than", self.up)
        self.cut = ispin(5, 95, 30)
        self.cut.setSuffix(" %")
        form.addRow("Report crops cutting more than", self.cut)
        self.squash = dspin(1.1, 10.0, 1.6, 1, 0.1)
        form.addRow("Report stretch more than", self.squash)
        self.per_class = QCheckBox("Per-class summary rows")
        self.per_class.setChecked(True)
        form.addRow("", self.per_class)
        self.add_action("Set class…", self.set_class, "Give the ticked images a class")

    def make_context(self):
        return {"adapter": self.session.adapter, "imgsz": self.imgsz.value(), "mode": self.mode.currentData(),
                "up": self.up.value(), "cut": self.cut.value() / 100, "squash": self.squash.value(),
                "sizes": defaultdict(list), "first": {}, "summary": self.per_class.isChecked()}

    def scan_sample(self, ctx, s):
        from AIO_QC.core.io import image_size
        tags = self.session.tags_of(s.key)
        c = tags[0] if tags else "(no class)"
        p = _image_of(ctx, s)
        wh = image_size(str(p)) if p is not None else None
        if not wh or not wh[0]:
            return []
        w, h = wh[0], wh[1]
        ctx["sizes"][c].append(min(w, h))
        ctx["first"].setdefault(c, s.key)
        S, rows = ctx["imgsz"], []
        if ctx["mode"] == "crop":
            k = S / max(1, min(w, h))
            cut = 1 - min(w, h) / max(w, h)
            shown = f"{round(w * k)}×{round(h * k)} → {S}×{S}"
            if k > ctx["up"]:
                rows.append(("Enlarged", f"×{k:.1f}: a {min(w, h)} px side stretched to {S} px — blurry"))
            if cut > ctx["cut"]:
                rows.append(("Centre crop cuts much", f"{cut:.0%} of the picture is cut off ({'left / right' if w > h else 'top / bottom'})"))
        else:
            k = S / max(1, max(w, h))
            sq = max(w, h) / max(1, min(w, h))
            shown = f"{S}×{S} (from {w}×{h})"
            if S / max(1, min(w, h)) > ctx["up"]:
                rows.append(("Enlarged", f"×{S / min(w, h):.1f} on the shorter side — blurry"))
            if sq > ctx["squash"]:
                rows.append(("Squashed", f"stretched {sq:.1f} : 1 into a square — shapes distorted"))
        return [ResultRow(s.key, (a, ", ".join(tags), f"{w}×{h}", shown, b), tone="warning",
                          payload={"class": c}) for a, b in rows]

    def finalize(self, ctx):
        if not ctx["summary"]:
            return []
        rows, S = [], ctx["imgsz"]
        for c, mins in sorted(ctx["sizes"].items()):
            a = np.array(mins, float)
            p5 = int(np.percentile(a, 5))
            up = float((S / a > ctx["up"]).mean())
            rows.append(ResultRow(ctx["first"][c], ("Class summary", c, f"shorter side: median {int(np.median(a))} px",
                                                    f"{up:.0%} enlarged > ×{ctx['up']:g}",
                                                    f"imgsz ≤ {p5} keeps 95 % of “{c}” from being enlarged"),
                                  tone="info", emphasis=True, payload={"class": c}))
        return rows

    def summarize(self):
        live = [r for r in self.rows if r and r.cells[0] != "Class summary"]
        c = Counter(r.cells[0] for r in live)
        return " · ".join(f"{k}: {n:,}" for k, n in c.most_common()) or "Every image keeps its detail at this size."


# ====================================================================== 8. OCR on whole images
CLASS_NAME = "(the class name)"


def _ocr_base():
    from .ocr_dialog import OcrCheckDialog
    return OcrCheckDialog


class ClsOcrDialog(_ocr_base()):
    TITLE, ICON = "OCR Check (classes)", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("For classifiers whose classes are text — digits, letters, signs, plates: the whole image is read by an "
            "OCR engine from your own Python environment and compared with its class name (or checked with your "
            "validators). An image whose text says another class is probably filed wrong. Reads are cached.")

    def _fill_classes(self):
        super()._fill_classes()
        saved = self.cfg.get("classes", {})
        for lab, (it, _vb, _lines, tf) in self._cls_widgets.items():
            tf.insertItem(0, CLASS_NAME)
            tf.setCurrentText(saved.get(lab, {}).get("text_field") or CLASS_NAME)
            tf.setToolTip("Compare the text read in the image with the class name (or leave empty: validators only)")
            if not saved and len(lab.strip()) <= 3:          # short names are usually the text itself (7, A, km)
                from PyQt6.QtCore import Qt
                it.setCheckState(Qt.CheckState.Checked)

    def make_context(self):
        ctx = super().make_context()
        ctx["thr"] = dict(ctx["thr"], geometry=False)       # the whole image: no box to fit
        return ctx

    def _row(self, f):
        row = super()._row(f)
        cells = list(row.cells)
        cells[4] = str(cells[4]).replace("stored “", "class “")
        return ResultRow(row.key, tuple(cells), shape_index=None, payload=row.payload, tone=row.tone)

    def scan_sample(self, ctx, s):
        from AIO_QC.core.ocr_check import Read, evaluate
        if ctx["load_error"]:
            return []
        tags = [t for t in self.session.tags_of(s.key) if t in ctx["classes"]]
        if not tags:
            return []
        p = _image_of({"adapter": ctx["adapter"]}, s)
        if p is None:
            return []
        box = (0.0, 0.0, 1.0, 1.0)
        bkey = "whole"
        cache = ctx["cache"]
        data = (cache.get_preds(ctx["sig"], p) if cache is not None else None) or {"boxes": {}}
        if bkey not in data["boxes"]:
            if not ctx["loaded"]:
                c = ctx["cfg"]
                try:
                    ctx["load_info"] = ctx["svc"].ensure_loaded(c["python"], c["engine"], c["codes"], c["gpu"],
                                                                c["mode"], c["strategy"], ctx["thr"]["min_conf"])
                    ctx["loaded"] = True
                except Exception as exc:
                    ctx["load_error"] = str(exc)
                    raise
            res = ctx["svc"].read([{"path": str(p), "boxes": [list(box)], "norm": True}], 0.0, ctx["thr"]["min_h"],
                                  cancel=ctx["cancel"])[0]
            if "error" in res:
                raise RuntimeError(res["error"])
            data["w"], data["h"] = res["w"], res["h"]
            data["boxes"][bkey] = res["boxes"][0]
            if cache is not None:
                cache.put_preds(ctx["sig"], p, data)
            ctx["ran"] += 1
        else:
            ctx["cached"] += 1
        W, H = data.get("w", 0), data.get("h", 0)
        rows = []
        for t in tags:
            cc = ctx["classes"][t]
            gt = t if cc.get("text_field") == CLASS_NAME else None
            r = Read(s.key, 0, t, s.split, (0.0, 0.0, float(W), float(H)), data["boxes"][bkey], gt, (W, H))
            ctx["reads"].append(r)
            f = evaluate(r, cc, ctx["thr"], ctx["validators"])
            if f is not None:
                rows.append(self._row(f))                 # a classifier: the whole image, no shape
        return rows


# ====================================================================== 9. validation (class folders)
MAGIC = ((b"\xff\xd8\xff", (".jpg", ".jpeg", ".jpe", ".jfif")), (b"\x89PNG\r\n\x1a\n", (".png",)),
         (b"GIF8", (".gif",)), (b"BM", (".bmp", ".dib")), (b"II*\x00", (".tif", ".tiff")),
         (b"MM\x00*", (".tif", ".tiff")))


def sniff(head: bytes) -> str | None:
    """The image type its first bytes say ('.png', …), or None when unknown."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    for magic, exts in MAGIC:
        if head.startswith(magic):
            return exts[0]
    return None


def _ext_ok(ext: str, kind: str) -> bool:
    return any(ext in exts for magic, exts in MAGIC if exts[0] == kind) or ext == kind


def _walk_dataset(root: Path):
    """(dir, subdirs, files) under the dataset, skipping AIO QC's own and hidden folders."""
    from AIO_QC.core.issues_folder import is_issues_dir
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".") and d not in ("__pycache__",)
                 and not is_issues_dir(os.path.join(dp, d))]         # images set aside: not part of the dataset
        yield Path(dp), dn, fn


class ClsValidationDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Validation (class folders)", ""
    SUPPORTED = {DatasetType.CLS_FOLDERS}
    HELP = ("A class-folder dataset is valid when every image sits in exactly one class folder and every file there "
            "is a readable image: images outside any class folder (no class), files that are not images, empty or "
            "unreadable files, an extension that does not match the content (a PNG named .jpg), empty class "
            "folders and folders nested inside a class folder.")
    COLUMNS = ["Problem", "Class", "File", "Detail"]
    FILE_COLUMN = False
    FILTER_COLUMN = 0
    AUTO_START = True

    def build_params(self, form):
        self.deep = QCheckBox("Decode every image (slower: finds truncated / corrupt files)")
        form.addRow("", self.deep)
        self.add_action("Set class…", self.set_class, "Give the ticked images a class (moves them into its folder)")

    def make_context(self):
        return {"adapter": self.session.adapter, "deep": self.deep.isChecked()}

    def scan_sample(self, ctx, s):
        tags = self.session.tags_of(s.key)
        p = _image_of(ctx, s)
        rel = os.path.relpath(str(p or s.key), str(self.session.root))
        rows = []
        if not tags:
            rows.append(ResultRow(s.key, ("Not in a class folder", "", rel, "the image has no class — move it into "
                                                                            "its class folder (Set class…)"),
                                  tone="error"))
        if p is None or not p.is_file():
            return rows + [ResultRow(s.key, ("Missing file", ", ".join(tags), rel, "the image file is gone"),
                                     tone="error")]
        try:
            size = p.stat().st_size
            with open(p, "rb") as f:
                head = f.read(16)
        except OSError as exc:
            return rows + [ResultRow(s.key, ("Unreadable", ", ".join(tags), rel, str(exc)[:160]), tone="error")]
        if size == 0:
            return rows + [ResultRow(s.key, ("Empty file", ", ".join(tags), rel, "0 bytes"), tone="error")]
        kind = sniff(head)
        ext = p.suffix.lower()
        if kind and not _ext_ok(ext, kind):
            rows.append(ResultRow(s.key, ("Wrong extension", ", ".join(tags), rel,
                                          f"named {ext} but the content is {kind[1:].upper()}"), tone="warning"))
        if ctx["deep"] or kind is None:
            import cv2
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if img is None:
                rows.append(ResultRow(s.key, ("Unreadable", ", ".join(tags), rel, "the image cannot be decoded"),
                                      tone="error"))
        return rows

    def finalize(self, ctx):
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        s = self.session
        rows = []
        exts = {e.lower() for e in IMAGE_EXTENSIONS}
        known = {str(smp.image_path) for smp in s.samples if smp.image_path is not None}
        classes = set(s.cls_classes())
        for split, base in getattr(s.base_adapter, "bases", []) or [("", Path(s.root))]:
            base = Path(base)
            if not base.is_dir():
                continue
            split_names = {"train", "val", "valid", "validation", "test"}
            from AIO_QC.core.issues_folder import is_issues_dir
            for d in sorted(x for x in base.iterdir() if x.is_dir() and not x.name.startswith(".")
                            and not is_issues_dir(x)):
                if not split and d.name.lower() in split_names:
                    continue
                if not any(Path(f).suffix.lower() in exts for _dp, _dn, fn in _walk_dataset(d) for f in fn):
                    rows.append(ResultRow(str(d), ("Empty class folder", d.name, os.path.relpath(d, s.root),
                                                   "no images" + (f" in {split}" if split else "")
                                                   + (" — a class with no examples" if d.name in classes else "")),
                                          tone="info"))
            for dp, dn, fn in _walk_dataset(base):
                rel_parts = Path(os.path.relpath(dp, base)).parts
                cls = rel_parts[0] if rel_parts and rel_parts[0] != "." else ""
                if cls and len(rel_parts) == 2 and any(Path(x).suffix.lower() in exts for x in fn):
                    rows.append(ResultRow(str(dp), ("Folder inside a class folder", cls,
                                                    os.path.relpath(dp, s.root),
                                                    f"{sum(1 for x in fn if Path(x).suffix.lower() in exts)} image(s) "
                                                    "one level deeper — still read as this class"), tone="info"))
                for f in fn:
                    full = Path(dp) / f
                    if f.startswith("."):
                        continue
                    ext = full.suffix.lower()
                    if ext in exts or str(full) in known:
                        continue
                    if ext in (".json", ".txt", ".xml", ".csv", ".yaml", ".yml"):
                        continue                                  # annotation / list files: the Pairs check
                    rows.append(ResultRow(str(full), ("Not an image", cls, os.path.relpath(full, s.root),
                                                      f"{ext or 'no extension'} file in the dataset"),
                                          tone="warning"))
        return rows

    def summarize(self):
        live = [r for r in self.rows if r]
        c = Counter(r.cells[0] for r in live)
        return " · ".join(f"{k}: {n:,}" for k, n in c.most_common()) or "Every image is in a class folder and readable."

    def on_row_activated(self, row):
        if row is None:
            return
        if row.key in self.session.by_key:
            self.app.open_sample(row.key, None)
        else:
            p = Path(row.key)
            _reveal(p if p.is_dir() else p.parent)


# ====================================================================== 10. pairs (class folders)
class ClsPairsDialog(_ClsFixes, BaseScanDialog):
    TITLE, ICON = "Pairs (class folders)", ""
    SUPPORTED = {DatasetType.CLS_FOLDERS}
    HELP = ("In a class-folder dataset the folder IS the label, so nothing should be paired with an image — but "
            "datasets converted from detection often keep annotation files next to the images (they may say "
            "something else), or annotation files whose image is gone. And one file name in several class folders "
            "of a split usually means the same image was filed twice.")
    COLUMNS = ["Problem", "Class", "File", "Detail"]
    FILE_COLUMN = False
    FILTER_COLUMN = 0
    AUTO_START = True

    def build_params(self, form):
        self.add_action("Set class…", self.set_class, "Give the ticked images a class")

    def make_context(self):
        return {"names": defaultdict(list)}

    def scan_sample(self, ctx, s):
        if s.image_path is not None:
            ctx["names"][(s.split, Path(s.name).stem.lower())].append(s.key)
        return []

    def finalize(self, ctx):
        import hashlib
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        s = self.session
        rows = []
        for (split, stem), keys in ctx["names"].items():
            tags = {k: tuple(s.tags_of(k)) for k in keys}
            if len(keys) < 2 or len(set(tags.values())) < 2:
                continue
            digests = {}
            for k in keys:
                try:
                    digests[k] = hashlib.sha1(Path(s.by_key[k].image_path).read_bytes()).hexdigest()
                except (OSError, KeyError, TypeError):
                    digests[k] = None
            same = len({d for d in digests.values() if d}) == 1
            for k in keys:
                others = ", ".join(f"“{', '.join(t)}”" for kk, t in tags.items() if kk != k)
                rows.append(ResultRow(k, ("One name in several classes", ", ".join(tags[k]),
                                          os.path.relpath(k, s.root),
                                          ("identical file" if same else "different files") + f" also in {others}"
                                          + (f" ({split})" if split else "")),
                                      tone="error" if same else "warning"))
        exts = {e.lower() for e in IMAGE_EXTENSIONS}
        for dp, _dn, fn in _walk_dataset(Path(s.root)):
            stems = {Path(f).stem.lower(): f for f in fn if Path(f).suffix.lower() in exts}
            rel = Path(os.path.relpath(dp, s.root)).parts
            cls = next((p for p in reversed(rel) if p in set(s.cls_classes())), "")
            for f in fn:
                ext = Path(f).suffix.lower()
                if ext not in (".json", ".txt", ".xml"):
                    continue
                if f.lower() in ("classes.txt", "labels.txt", "readme.txt") or f.lower().endswith(("_list.txt",)):
                    continue
                full = Path(dp) / f
                img = stems.get(Path(f).stem.lower())
                what = self._describe(full)
                if img:
                    rows.append(ResultRow(str(Path(dp) / img) if str(Path(dp) / img) in s.by_key else str(full),
                                          ("Annotation file next to an image", cls, os.path.relpath(full, s.root),
                                           f"{what} — the class comes from the folder; this file is ignored"),
                                          tone="info"))
                else:
                    rows.append(ResultRow(str(full), ("Annotation without an image", cls,
                                                      os.path.relpath(full, s.root), what), tone="warning"))
        return rows

    @staticmethod
    def _describe(p: Path) -> str:
        try:
            if p.suffix.lower() == ".json":
                from AIO_QC.core.io import load_json_light
                d = load_json_light(str(p)) or {}
                n = len(d.get("shapes") or [])
                fl = [k for k, v in (d.get("flags") or {}).items() if v is True]
                return f"LabelMe file: {n} shape(s)" + (f", flags {', '.join(fl)}" if fl else "")
            if p.suffix.lower() == ".txt":
                n = sum(1 for line in p.read_text(errors="ignore").splitlines() if line.strip())
                return f"text / YOLO file: {n} line(s)"
            return f"{p.suffix[1:].upper()} file"
        except Exception:                                 # noqa: BLE001
            return "annotation file"

    def summarize(self):
        live = [r for r in self.rows if r]
        c = Counter(r.cells[0] for r in live)
        return " · ".join(f"{k}: {n:,}" for k, n in c.most_common()) or "Nothing is paired with the images — as it " \
                                                                         "should be."

    def on_row_activated(self, row):
        if row is None:
            return
        if row.key in self.session.by_key:
            self.app.open_sample(row.key, None)
        else:
            _reveal(Path(row.key).parent)


def _model_check():
    from .cls_model_check_dialog import ClsModelCheckDialog
    return ClsModelCheckDialog


# tool key -> the classifier version (used when the detection tool does not support the open dataset)
CLS_VARIANTS = {
    "ambiguity": ClsAmbiguityDialog, "text_labels": ClsTextDialog, "duplicates": ClsDuplicatesDialog,
    "outliers": ClsOutliersDialog, "sequence": ClsSequenceDialog, "dim_area": ClsDimAreaDialog,
    "train_size": ClsTrainSizeDialog, "ocr_check": ClsOcrDialog, "validation": ClsValidationDialog,
    "pairs": ClsPairsDialog, "model_check": _model_check(),
}
