# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image–Label Pairs (QC drawer ▸ Dataset): every image matched with its label file, and everything that is not —
JSONs without their image, images without a JSON (LabelMe); labels without their image, images without a label,
labels in the wrong folder or with another upper / lower case (YOLO, images/ and labels/ mirrored) — plus files
that are not LabelMe JSONs / YOLO labels at all. Each finding names the likely partner (same name elsewhere in the
dataset) and, where one is safe, proposes a fix: point imagePath at the image found, move the label next to its
image, create an empty label (YOLO background) or an empty LabelMe JSON. Fixes are previewed, logged and undoable;
"Move to trash" is recoverable (.aio_qc_trash).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core import pairs as P
from AIO_QC.core.dataset import FILE_TYPES, DatasetType
from .base_scan import BaseScanDialog, ResultRow

AUTO_FIXES = ("set_image_path", "move_label", "rename_label", "empty_label", "empty_json")


class PairsDialog(BaseScanDialog):
    SUPPORTED = FILE_TYPES | {DatasetType.CLS_LABELME}
    TITLE, ICON = "Image–Label Pairs", ""
    HELP = ("Every image matched with its label file, and everything that is not. LabelMe: JSONs whose image is "
            "missing (imagePath and the same name next to it), images no JSON uses, two JSONs on one image, "
            "imagePath naming another file, several images with the JSON's name. YOLO (images/ and labels/ mirror "
            "each other): labels without their image, images without a label (background — or a missing label), "
            "labels whose name differs only in upper / lower case (they pair on Windows, not on Linux where "
            "training usually runs), labels left in images/, several images on one label. Plus files that are not "
            "LabelMe JSONs / YOLO labels at all (what they look like is said). For each, the likely partner is "
            "searched across the dataset (another folder or split) and a safe fix proposed: point imagePath at the "
            "image, move the label next to its image, create an empty label (background) or an empty LabelMe JSON. "
            "Double-click a row to open it (files outside the dataset open their folder).")
    COLUMNS = ["Problem", "Path", "Proposed fix", "Likely partner", "Detail"]
    FILTER_COLUMN = 0
    FILE_COLUMN = False                          # "Path" says it (relative to the dataset), for non-sample files too
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.add_action("🩹 Apply proposed fixes", self.apply_fixes,
                        "For the ticked (or selected) rows with a proposed fix: preview, then apply — logged and "
                        "undoable (↶ Undo fixes)")
        self.undo_btn = self.add_action("↶ Undo fixes", self.undo_fixes, "Put back the files moved / created by the "
                                                                          "last Apply")
        self.undo_btn.setEnabled(False)
        self.add_action("🗑 Move to trash", self.trash_rows,
                        "Move the ticked (or selected) files to .aio_qc_trash (recoverable): files without a partner "
                        "and files that are not label files")
        self._undo = []

    def post_init(self):
        self.progress.setFormat("")                  # one walk of the dataset: no per-file count to show

    # ---------------------------------------------------------------- scan
    def samples_to_scan(self):
        return []                                    # one walk of the dataset, in finalize()

    def make_context(self):
        ad = getattr(self.session, "base_adapter", None) or self.session.adapter
        cm = getattr(ad, "class_map", None)
        return {"root": self.session.root, "yolo": self.session.is_yolo, "bases": list(getattr(ad, "bases", []) or []),
                "n": len(cm.names) if cm is not None and self.session.is_yolo else 0,
                "kpt": bool(getattr(cm, "kpt_shape", None)), "keys": set(self.session.by_key)}

    def finalize(self, ctx):
        stop = (lambda: self.worker is not None and getattr(self.worker, "_stop", False))
        if ctx["yolo"]:
            rep = P.yolo_pairs(ctx["bases"], ctx["root"], ctx["n"], ctx["kpt"], cancel=stop)
        else:
            rep = P.labelme_pairs(ctx["root"], cancel=stop)
        ctx["report"] = rep
        rows = []
        for f in rep.findings:
            key = f.key if f.key in ctx["keys"] else f.path
            rows.append(ResultRow(key, (f.text, self._rel(f.path), P.FIX_TEXT.get(f.fix, "—") if f.fix else "—",
                                        self._rel(f.partner) if f.partner else "", f.detail),
                                  payload=f, tone="error" if f.severity == P.ERROR else "warning"))
        return rows

    def _rel(self, p) -> str:
        try:
            return Path(os.path.relpath(p, self.session.root)).as_posix()
        except ValueError:
            return str(p)

    def summarize(self):
        last = self.table.columnCount() - 1          # Detail: takes the rest of the width; the full text on hover
        for r in range(self.table.rowCount()):
            it = self.table.item(r, last)
            if it is not None and not it.toolTip():
                it.setToolTip(it.text())
        rep = (self.ctx or {}).get("report")
        if rep is None:
            return super().summarize()
        live = [r for r in self.rows if r]
        c = Counter(r.payload.kind for r in live)
        if rep.kind == "yolo":
            head = (f"{rep.matched:,} of {rep.images:,} image(s) paired with a label · {rep.labels:,} label file(s)")
        else:
            head = (f"{rep.matched:,} of {rep.labels:,} JSON(s) paired with an image · {rep.images:,} image(s)")
        if not c:
            return head + " — every file has its partner"
        return head + " — " + " · ".join(f"{P.TEXT[k]}: {n:,}" for k, n in c.most_common())

    def on_row_activated(self, row):
        if row is None:
            return
        if row.key in self.session.by_key:
            self.app.open_sample(row.key, None)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(row.payload.path).parent)))

    # ---------------------------------------------------------------- fixes
    def apply_fixes(self):
        rows = [r for r in self.target_rows() if r.payload is not None and r.payload.fix in AUTO_FIXES]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows whose proposed fix is not “—” or "
                                                      "“move to trash”.")
            return
        c = Counter(P.FIX_TEXT[r.payload.fix] for r in rows)
        if not self.confirm("Apply these fixes?\n\n" + "\n".join(f"• {n:,} × {t}" for t, n in c.most_common()) +
                            "\n\nFiles are moved / created (never overwritten); ↶ Undo fixes puts them back."):
            return
        undo, errors, reindex, added = [], [], set(), []
        imagepath = {}
        for r in rows:
            f = r.payload
            try:
                if f.fix == "set_image_path":
                    imagepath[r.key] = f.fix_arg
                elif f.fix in ("move_label", "rename_label"):
                    src, dst = (f.fix_arg if isinstance(f.fix_arg, tuple) else (f.path, f.fix_arg))
                    self._move(src, dst, undo)
                    reindex.add(r.key)
                elif f.fix == "empty_label":
                    self._create(f.fix_arg, "", undo)
                    reindex.add(r.key)
                elif f.fix == "empty_json":
                    added.append(self._empty_json(Path(f.path), undo))
            except (OSError, ValueError) as exc:
                errors.append(f"{Path(f.path).name}: {exc}")
        if imagepath:
            from AIO_QC.core.fixes import apply_fix

            def make(rel):
                return lambda ann: apply_fix(ann, "set_image_path", None, rel)
            changed, errs = self.session.modify_many(self, [(k, make(v)) for k, v in imagepath.items()
                                                            if k in self.session.by_key], "Fixing imagePath…", raw=True)
            errors += errs
        if reindex:
            self.session.reindex_keys([k for k in reindex if k in self.session.by_key])
        if added:
            ad = getattr(self.session, "base_adapter", None) or self.session.adapter
            new = []
            for p in added:
                s, _structural = ad.sample_for_path(p)
                if s is not None:
                    new.append(s)
            if new:
                self.session.add_samples(new)
        self._undo = undo
        self._undo_keys = set(reindex) | set(imagepath)
        self.undo_btn.setEnabled(bool(undo))
        self.session.log(list(reindex) + [str(p) for p in added], "pairs", f"{len(rows)} pairing fix(es)",
                         tool=self.TITLE)
        self.report("Fixes applied", [f"Applied {len(rows) - len(errors)} of {len(rows)} fix(es)."], errors)
        self.start_scan()

    def _move(self, src, dst, undo):
        src, dst = Path(src), Path(dst)
        same_name = os.path.normcase(str(src)) == os.path.normcase(str(dst))      # an upper / lower case rename
        if dst.exists() and not same_name:
            raise ValueError(f"'{dst.name}' already exists there — not overwritten")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if same_name and src != dst:                 # Windows: rename via a temporary name
            tmp = src.with_name(src.name + ".aioqc-tmp")
            os.replace(src, tmp)
            os.replace(tmp, dst)
        else:
            os.replace(src, dst)
        undo.append(("move", str(dst), str(src)))

    @staticmethod
    def _create(path, text, undo):
        p = Path(path)
        if p.exists():
            raise ValueError(f"'{p.name}' already exists — not overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        undo.append(("create", str(p), None))

    def _empty_json(self, img: Path, undo) -> str:
        from AIO_QC.core.io import image_size
        size = image_size(img) or (None, None)
        data = {"version": "5.4.1", "flags": {}, "shapes": [], "imagePath": img.name, "imageData": None,
                "imageHeight": size[1], "imageWidth": size[0]}
        target = img.with_suffix(".json")
        from AIO_QC.core import provenance as PV
        PV.stamp_labelme(data, o=PV.make_origin("rule", "Image–Label Pairs", detail="empty label file for an image"))
        self._create(target, json.dumps(data, indent=2), undo)
        return str(target)

    def undo_fixes(self):
        if not self._undo:
            return
        errors, gone = [], []
        for op, a, b in reversed(self._undo):
            try:
                if op == "move" and Path(a).exists():
                    Path(b).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(a, b)
                elif op == "create" and Path(a).exists():
                    os.remove(a)
                    gone.append(a)
            except OSError as exc:
                errors.append(f"{Path(a).name}: {exc}")
        self._undo = []
        self.undo_btn.setEnabled(False)
        drop = [k for k in gone if k in self.session.by_key]
        if drop:
            self.session.forget_samples(drop)
        self.session.reindex_keys([k for k in getattr(self, "_undo_keys", ()) if k in self.session.by_key])
        self.report("Undone", ["The files moved / created by the last Apply are back as they were."], errors)
        self.start_scan()

    def trash_rows(self):
        """The file each row is about: a JSON or a whole YOLO sample (its image) leaves the dataset; a stray label
        or an image no JSON uses is moved as a file."""
        rows = self.target_rows()
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first.")
            return
        by_key = self.session.by_key
        samples, with_images, paths, touched = [], [], [], set()
        for r in rows:
            f = r.payload
            if r.key in by_key and os.path.normcase(f.path) == os.path.normcase(r.key):
                (with_images if self.session.is_yolo else samples).append(r.key)
            else:
                paths.append(f.path)
                if r.key in by_key:
                    touched.add(r.key)                # e.g. its label left: the image becomes a background image
        samples, with_images, paths = (list(dict.fromkeys(x)) for x in (samples, with_images, paths))
        n = len(samples) + len(with_images) + len(paths)
        if not self.confirm(f"Move {n:,} file(s) to .aio_qc_trash?\n(Recoverable: they are moved, not deleted.)"):
            return
        _a, e1 = self.session.trash_samples(samples)
        _b, e2 = self.session.trash_samples(with_images, include_images=True)
        done, e3 = self.session.trash_paths(paths)
        if touched:
            self.session.reindex_keys([k for k in touched if k in by_key])
        self.report("Done", [f"Moved {len(samples) + len(with_images) + len(done):,} file(s) to trash."], e1 + e2 + e3)
        self.start_scan()
