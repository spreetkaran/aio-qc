# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🔤 Text Labels — check the text stored in your annotations (LabelMe `description` or another shape field) without
any OCR model and without assuming any kind of text: formats, known texts and confused characters learned from the
open dataset, the validators you choose (optional), and what the other frames of the same object say — each finding
with a correction that fits those rules, applied in one click when it is the only sensible one (backed up, undoable).

Validators, text field and normalisation are shared with OCR Check.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QToolButton, QWidget,
)

from AIO_QC.core import text_labels as TL
from AIO_QC.core import validators as VL
from AIO_QC.core.dataset import DatasetType
from AIO_QC.core.ocr_check import DEFAULT_CLASS
from AIO_QC.core.sequences import FrameInfo, build_sequences, split_name, track
from AIO_QC.core.shape_ops import Backup, fingerprint, resolve
from AIO_QC.core.state import load_ocr_config, save_ocr_config
from AIO_QC.utils import newButton
from .base_scan import BaseScanDialog, ResultRow



def classes_with_text(session, max_files: int = 400, share: float = 0.6) -> list[str]:
    """Classes whose shapes carry text in this dataset (a sample of files): ≥ 60 % of their shapes have some."""
    import random
    samples = list(session.samples)
    if len(samples) > max_files:
        samples = random.Random(3).sample(samples, max_files)
    have, total = Counter(), Counter()
    for s in samples:
        try:
            ann = session.adapter.load(s, light=True)
        except Exception:
            continue
        for sh in ann.shapes:
            lab = str(sh.get("label", ""))
            total[lab] += 1
            if TL.read_text(sh)[1]:
                have[lab] += 1
    return sorted(l for l, n in total.items() if n >= 3 and have[l] / n >= share)


class TextLabelsDialog(BaseScanDialog):
    TITLE = "Text Labels"
    ICON = "🔤"
    HELP = ("Checks the text STORED in your annotations — no OCR model needed, nothing specific to one kind of text. "
            "The rules come from the validators you choose (optional) and from the dataset itself: the formats most "
            "texts of a class share, the texts that occur again and again, and the characters it confuses. Video "
            "frames of the same object are compared. Every finding comes with a correction that fits those rules — "
            "applied in one click when it is the only sensible one, otherwise listed for you to pick.")
    COLUMNS = ["Issue", "Label", "Stored text", "Suggestion", "Sure", "Why"]
    FILTER_COLUMN = 0
    SUPPORTED = {DatasetType.LABELME}

    # ------------------------------------------------------------------ parameters
    def build_params(self, form):
        root = self.app.session.root
        self.cfg = load_ocr_config(root)
        self.validators = VL.discover(root)
        saved = self.cfg.get("classes") or {}
        labels = sorted(self.app.session.label_totals(raw=False))
        chosen = [l for l in labels if l in saved] or classes_with_text(self.app.session)
        first = saved.get(chosen[0], {}) if chosen else {}

        self.cls_btn = QToolButton()
        self.cls_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.cls_btn.setMenu(QMenu(self.cls_btn))
        self.cls_btn.setProperty("chosen", chosen)
        self.cls_btn.menu().aboutToShow.connect(lambda: self._fill_menu(self.cls_btn, [(l, l, "") for l in labels]))
        self._label_btn(self.cls_btn, "choose the classes that hold text")

        self.val_btn = QToolButton()
        self.val_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.val_btn.setMenu(QMenu(self.val_btn))
        self.val_btn.setProperty("chosen", list(first.get("validators") or []))
        self.val_btn.menu().aboutToShow.connect(self._fill_validators)
        self._label_btn(self.val_btn, "none — empty / frames / format checks only", self._vtitle)
        self.val_btn.setToolTip("Any one of these must accept the text (same validators as OCR Check)")

        self.field = QComboBox()
        self.field.setEditable(True)
        self.field.addItem("auto", "")
        for f in TL.TEXT_FIELDS:
            self.field.addItem(f, f)
        tf = first.get("text_field", "")
        self.field.setCurrentIndex(max(0, self.field.findData(tf)) if self.field.findData(tf) >= 0 else 0)
        if tf and self.field.findData(tf) < 0:
            self.field.setEditText(tf)
        self.field.setToolTip("The shape field that holds the text. auto = the first of "
                              + ", ".join(TL.TEXT_FIELDS) + " that has one")
        nrm = {**DEFAULT_CLASS["normalize"], **(self.cfg.get("normalize") or {})}
        self.upper = QCheckBox("UPPER CASE")
        self.upper.setChecked(bool(nrm.get("upper", True)))
        self.remove = QLineEdit(nrm.get("remove", " -._"))
        self.remove.setMaximumWidth(90)
        self.remove.setToolTip("Characters removed before checking (the stored text is not changed unless you apply "
                               "a fix)")
        self.frames = QCheckBox("Compare the frames of the same object (video / burst sequences)")
        self.frames.setChecked(self.app.settings.value("textlabels/frames", True, type=bool))
        self.learn = QCheckBox("Learn from this dataset: common formats, known texts, confused characters")
        self.learn.setToolTip("Per class: the run structures most texts share (e.g. 4 letters · 7 digits) when ≥ 80 % "
                              "fit a few of them, texts that occur several times, and character swaps the frames of "
                              "one object disagree on. Free text gets no format rule.")
        self.learn.setChecked(self.app.settings.value("textlabels/learn", True, type=bool))
        self.fmt = QCheckBox("Flag texts not stored in their normalised form")
        self.fmt.setChecked(self.app.settings.value("textlabels/format", False, type=bool))
        edit = newButton("Validators…", slot=self._open_validators, tip="New / edit validators, learn a pattern "
                                                                         "from your stored texts")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.val_btn, 1)
        h.addWidget(edit)
        nrow = QWidget()
        n = QHBoxLayout(nrow)
        n.setContentsMargins(0, 0, 0, 0)
        n.addWidget(self.upper)
        n.addWidget(QLabel("remove"))
        n.addWidget(self.remove)
        n.addStretch(1)
        form.addRow("Text classes:", self.cls_btn)
        form.addRow("Text field:", self.field)
        form.addRow("Validators:", row)
        form.addRow("Normalise:", nrow)
        form.addRow(self.learn)
        form.addRow(self.frames)
        form.addRow(self.fmt)

        self.add_action("✓ Apply suggestion", self.apply_rows,
                        "Write the suggested text into the ticked (or selected) shapes. Sure suggestions are applied "
                        "directly; for the others you pick from the candidates. Backed up — ↶ Undo puts them back.")
        self.add_action("✎ Edit text…", self.edit_row, "Type the right text for the selected shape")
        self.undo_btn = self.add_action("↶ Undo", self.undo, "Put back the files changed by the last Apply / Edit")
        self.undo_btn.setEnabled(False)
        self._last_backup = None

    def _vtitle(self, name):
        v = self.validators.get(name)
        return v.title if v else f"{name} (missing)"

    @staticmethod
    def _label_btn(b, empty, title=lambda x: x):
        chosen = b.property("chosen") or []
        b.setText(" · ".join(title(x) for x in chosen) if chosen else empty)

    def _fill_menu(self, b, items):
        m = b.menu()
        m.clear()
        chosen = set(b.property("chosen") or [])
        for key, text, tip in items:
            a = m.addAction(text)
            a.setCheckable(True)
            a.setChecked(key in chosen)
            a.setToolTip(tip)
            a.toggled.connect(lambda on, k=key, bb=b: self._toggle(bb, k, on))

    def _toggle(self, b, key, on):
        chosen = [x for x in (b.property("chosen") or []) if x != key] + ([key] if on else [])
        b.setProperty("chosen", chosen)
        if b is self.val_btn:
            self._label_btn(b, "none — empty / frames / format checks only", self._vtitle)
        else:
            self._label_btn(b, "choose the classes that hold text")

    def _fill_validators(self):
        m = self.val_btn.menu()
        m.clear()
        chosen = set(self.val_btn.property("chosen") or [])
        for origin in ("dataset", "user", "built-in"):
            vs = sorted((v for v in self.validators.values() if v.origin == origin), key=lambda v: v.title.lower())
            if not vs:
                continue
            m.addSection({"dataset": "This dataset", "user": "Mine", "built-in": "Built-in"}[origin])
            for v in vs:
                a = m.addAction(v.title)
                a.setCheckable(True)
                a.setChecked(v.name in chosen)
                a.setToolTip(v.description)
                a.setEnabled(not v.error)
                a.toggled.connect(lambda on, k=v.name: self._toggle(self.val_btn, k, on))

    def _open_validators(self):
        from .validator_editor import ValidatorEditor
        classes = list(self.cls_btn.property("chosen") or [])
        dlg = ValidatorEditor(self, self.session.root, reads_provider=self._texts_for, class_names=classes,
                              normalize_opts=self._normalize())
        dlg.exec()
        self.validators = VL.discover(self.session.root)
        self._label_btn(self.val_btn, "none — empty / frames / format checks only", self._vtitle)

    def _texts_for(self, label):
        """Stored texts of a class (normalised) — for "Test on this dataset" / "Learn from examples"."""
        fld = self.field.currentData() if self.field.currentText() == "auto" else self.field.currentText().strip()
        out = []
        keys = set(self.session.keys_with_label(label))
        for s in self.session.samples:
            if s.key not in keys:
                continue
            try:
                ann = self.session.adapter.load(s, light=True)
            except Exception:
                continue
            for sh in ann.shapes:
                if sh.get("label") == label:
                    _f, raw = TL.read_text(sh, fld or "")
                    if raw:
                        out.append(VL.normalize(raw, self._normalize()))
            if len(out) >= 5000:
                break
        return out

    def _normalize(self) -> dict:
        return {"upper": self.upper.isChecked(), "remove": self.remove.text(), "strip": True}

    def make_context(self):
        classes = list(self.cls_btn.property("chosen") or [])
        if not classes:
            raise ValueError("Choose the classes that hold text first (Text classes).")
        names = list(self.val_btn.property("chosen") or [])
        vals = [self.validators[n] for n in names if n in self.validators and not self.validators[n].error]
        fld = "" if self.field.currentText().strip() in ("", "auto") else self.field.currentText().strip()
        nopts = self._normalize()
        # shared with OCR Check: the same classes get the same validators / field / normalisation
        saved = dict(self.cfg.get("classes") or {})
        for c in classes:
            saved[c] = {**DEFAULT_CLASS, **saved.get(c, {}), "validators": names, "text_field": fld,
                        "normalize": nopts}
        self.cfg["classes"], self.cfg["normalize"] = saved, nopts
        try:
            save_ocr_config(self.session.root, self.cfg)
        except OSError:
            pass
        st = self.app.settings
        st.setValue("textlabels/frames", self.frames.isChecked())
        st.setValue("textlabels/format", self.fmt.isChecked())
        st.setValue("textlabels/learn", self.learn.isChecked())
        return {"adapter": self.session.adapter, "classes": set(classes), "cls": {"text_field": fld, "normalize": nopts},
                "validators": vals, "frames": self.frames.isChecked(), "learn": self.learn.isChecked(),
                "format": self.fmt.isChecked(), "texts": {}, "fr": [], "n_shapes": 0}

    # ------------------------------------------------------------------ scan: collect, then judge with what was learned
    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        W, H = (ann.width or 0), (ann.height or 0)
        boxes = []
        for i, sh in enumerate(ann.shapes):
            if sh.get("label") not in ctx["classes"] or sh.get("_error"):
                continue
            ctx["n_shapes"] += 1
            fld, raw = TL.read_text(sh, ctx["cls"]["text_field"])
            norm = VL.normalize(raw, ctx["cls"]["normalize"]) if raw and raw.strip() else ""
            ctx["texts"][(s.key, i)] = (str(sh.get("label")), fld or ctx["cls"]["text_field"] or "description", raw,
                                        norm, fingerprint(sh))
            pts = sh.get("points") or []
            if ctx["frames"] and W and H and len(pts) >= 2:
                try:
                    xs, ys = [float(p[0]) / W for p in pts], [float(p[1]) / H for p in pts]
                except (TypeError, ValueError, IndexError):
                    continue
                boxes.append((i, str(sh.get("label")), (min(xs), min(ys), max(xs), max(ys)), {}))
        if boxes:
            p = Path(s.key)
            pattern, n = split_name(p.stem)
            ctx["fr"].append(FrameInfo(s.key, f"{s.split}|{p.parent}|{pattern}", n, boxes))
        return []

    def rescan_sample(self, ctx, s):
        return []                                  # findings depend on the whole dataset: Rescan to refresh

    def finalize(self, ctx):
        texts = ctx["texts"]
        by_cls = defaultdict(list)
        for (_k, _i), (lab, _f, _r, norm, _fp) in texts.items():
            by_cls[lab].append(norm)
        # 1. what each class's texts look like here
        models = {lab: TL.Model(v) for lab, v in by_cls.items()} if ctx["learn"] else {}
        suggs = {lab: TL.Suggester(ctx["validators"], models.get(lab)) for lab in by_cls}
        # 2. the frames of one object
        cons = {}
        if ctx["frames"] and ctx["fr"]:
            tracks = []
            for seq in build_sequences(ctx["fr"], "auto", 3.0, 2):
                for tr in track(seq, 0.3, 1):
                    if len(tr) >= 2:
                        tracks.append([(seq[t].key, b[0], texts.get((seq[t].key, b[0]), ("",) * 4)[3])
                                       for t, b in sorted(tr.items())])
            cons = TL.consensus(tracks, lambda t, k, i: suggs[texts[(k, i)][0]].ok(t))
            # 3. characters this dataset confuses (frames that disagree by one or two substitutions)
            if models:
                pairs = defaultdict(list)
                for (k, i), (major, _v, _n) in cons.items():
                    lab, _f, _r, norm, _fp = texts[(k, i)]
                    if norm:
                        pairs[lab].append((norm, major))
                for lab, pr in pairs.items():
                    if lab in models:
                        models[lab].learn_pairs(pr)
        ctx["fr"] = None
        # 4. every text, judged with all of it
        found = {}
        for (k, i), (lab, fld, raw, _norm, fp) in texts.items():
            f = TL.check_text(k, i, lab, fld, raw, ctx["cls"], ctx["validators"], suggs[lab], ctx["format"], fp)
            if (k, i) in cons:
                f = f or TL.TextFinding(k, i, lab, fld, raw, texts[(k, i)][3], "track", "", fp=fp)
                f = TL.merge_track(f, cons[(k, i)])
            if f is not None:
                found[(k, i)] = f
        ctx["kinds"] = Counter(f.kind for f in found.values())
        ctx["learned"] = {lab: m for lab, m in models.items()}
        order = list(TL.KINDS)
        return [self._row(f) for f in sorted(found.values(), key=lambda f: (order.index(f.kind), f.key, f.idx))]

    def _row(self, f: TL.TextFinding) -> ResultRow:
        title, tone = TL.KINDS[f.kind]
        sugg = f.suggestion or ""
        if f.suggestion and not f.sure and len(f.candidates) > 1:
            sugg = f"{f.suggestion}  (+{len(f.candidates) - 1} more)"
        why = f.detail + (f" → {f.why}" if f.why else "")
        return ResultRow(f.key, (title, f.label, f.raw if f.raw is not None else "", sugg,
                                 "yes" if f.sure else ("pick" if f.suggestion else ""), why),
                         f.idx, payload=f, tone=tone)

    def summarize(self):
        c = self.ctx or {}
        if "kinds" not in c:
            return super().summarize()
        found = [r.payload for r in self.rows if r is not None]
        sure = sum(1 for f in found if f.suggestion and f.sure)
        parts = ", ".join(f"{TL.KINDS[k][0]} {v:,}" for k, v in c["kinds"].most_common())
        learned = []
        for lab, m in sorted((c.get("learned") or {}).items()):
            bits = [f"format {m.format_text()}" if m.formats else "free text (no common format)",
                    f"{len(m.known):,} known text(s)"] + ([f"{m.n_pairs} confused character pair(s)"] if m.n_pairs
                                                         else [])
            learned.append(f"<b>{lab}</b>: " + " · ".join(bits))
        return (f"{c['n_shapes']:,} text shape(s) checked — {len(found):,} finding(s)" + (f": {parts}" if parts else "")
                + (f" · {sure:,} with a sure fix" if sure else "")
                + ("<br>Learned — " + " ; ".join(learned) if learned else ""))

    # ------------------------------------------------------------------ fixes
    def apply_rows(self):
        rows = [r for r in self.target_rows() if r.payload is not None and r.payload.suggestion]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) findings that have a suggestion.")
            return
        picks = []
        unsure = [r for r in rows if not r.payload.sure]
        if len(unsure) == 1 and len(rows) == 1:
            f = unsure[0].payload
            text, ok = QInputDialog.getItem(self, self.TITLE, f"“{f.raw}” — which is right?\n({f.why})",
                                            f.candidates, 0, True)
            if not ok or not text.strip():
                return
            picks = [(unsure[0], text.strip())]
        else:
            picks = [(r, r.payload.suggestion) for r in rows if r.payload.sure]
            if unsure:
                if not picks:
                    QMessageBox.information(self, self.TITLE, "These findings have several possible corrections — "
                                            "select them one at a time to pick the right one.")
                    return
                if not self.confirm(f"{len(unsure):,} of the ticked findings have several possible corrections and "
                                    f"are skipped (select them one at a time to pick).\n\nApply the "
                                    f"{len(picks):,} sure one(s)?"):
                    return
            elif len(picks) > 1 and not self.confirm(f"Write the suggested text into {len(picks):,} shape(s)?\n\n"
                                                     "The originals are backed up (Undo)."):
                return
        self._write(picks)

    def edit_row(self):
        rows = [r for r in self.target_rows() if r.payload is not None]
        if len(rows) != 1:
            QMessageBox.information(self, self.TITLE, "Select one finding.")
            return
        f = rows[0].payload
        items = list(dict.fromkeys(([f.raw] if f.raw else []) + f.candidates))
        text, ok = QInputDialog.getItem(self, self.TITLE, f"Text for this {f.label}:", items or [""], 0, True)
        if ok and text.strip() and text.strip() != (f.raw or ""):
            self._write([(rows[0], text.strip())])

    def _write(self, picks):
        backup = Backup(self.session.root, "textlabels", "Text Labels")
        by_key = defaultdict(list)
        for r, text in picks:
            f = r.payload
            by_key[r.key].append((f.idx, f.fp, f.field, f.raw, text))
        stats = {"done": 0, "changed_since": 0}

        def make(key, todo):
            def fn(ann):
                found, _miss = resolve(ann.shapes, [(i, fp) for i, fp, *_ in todo])
                live = {fingerprint(ann.shapes[j]): j for j in found}
                hit = []
                for _i, fp, fld, old, new in todo:
                    j = live.get(fp)
                    cur = ann.shapes[j].get(fld) if j is not None else None
                    if j is None or ((cur if isinstance(cur, str) else "") or "") != (old or ""):
                        stats["changed_since"] += 1           # moved / retyped since the scan: never guessed
                        continue
                    hit.append((j, fld, new))
                if not hit:
                    return False
                sample = self.session.by_key[key]
                if sample.ann_path is not None:
                    backup.save(sample.ann_path)
                for j, fld, new in hit:
                    ann.shapes[j][fld] = new
                    stats["done"] += 1
                return True
            return fn
        with self.session.as_tool(self.TITLE):
            changed, errors = self.session.modify_many(self, [(k, make(k, v)) for k, v in by_key.items()],
                                                       "Writing texts...")
        for k in changed:
            p = self.session.by_key[k].ann_path
            if p is not None:
                backup.written(p)
        backup.close()
        if changed:
            self._last_backup = backup
            self.undo_btn.setEnabled(True)
        done = {(r.key, r.payload.idx) for r, _t in picks}
        self.remove_rows(lambda row: row.payload is not None and (row.key, row.payload.idx) in done)
        self._compact()
        self.summary.setText(self.summarize())
        msg = f"Wrote {stats['done']:,} text(s) in {len(changed):,} file(s)"
        if stats["changed_since"]:
            msg += f" · {stats['changed_since']:,} skipped (changed since the scan)"
        self.app.statusBar().showMessage(msg + " — ↶ Undo puts them back", 8000)
        if errors:
            self.app.log_errors([f"[{self.TITLE}] {e}" for e in errors])

    def undo(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last Apply / Edit?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.report("Undone", [f"Restored {len(keys):,} file(s). Rescan to see their findings again."] +
                    ([f"{len(skipped):,} file(s) changed afterwards and were left alone."] if skipped else []), [])

