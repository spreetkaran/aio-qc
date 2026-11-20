# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Train Model ▸ Augmentation and learning rate — the Ultralytics hyper-parameters people actually tune, with presets.

HypPanel(settings, optimizer, lr0, close_mosaic)   the dialog's own optimizer / learning-rate / close-mosaic widgets
                                                   are placed here, next to the rest of the schedule
  values()          -> {key: value} only what differs from the environment's defaults (what goes to Ultralytics)
  set_env(env)      the probed environment: its own defaults; settings its Ultralytics lacks are switched off
  set_task(task)    "det" / "seg": copy-paste is for segmentation only; "cls" (a YOLO classifier) / "clst" (any other
                    architecture, AIO QC's PyTorch trainer): the classifier settings — auto-augment policy, random
                    erasing, and the Fine-tuning group (head-only epochs, layer-wise decay, EMA, label smoothing,
                    class balance / hard examples); "clst" has its own defaults
  values(full)      full=True: every usable setting (the PyTorch trainer gets them all)
  set_names(names)  class names: text-like ones (plates, digits) switch mirroring off ("No flips for text")
  apply_aug(key) · apply_lr(key)    presets (core.training.AUG_PRESETS / LR_PRESETS)
  changed           signal: anything changed (the dialog re-checks Start)
Remembered between sessions (train/hyp, train/hyp_aug, train/hyp_lr, train/hyp_opt …); "Text in the images" follows
each dataset's class names (it can be unticked).
"""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QSpinBox, QVBoxLayout, QWidget)

from AIO_QC import utils
from AIO_QC.core import training as T
from AIO_QC.core.diag import ignored as _ignored

CUSTOM = "custom"


def _same(a, b) -> bool:
    if a is None or b is None:
        return (a or 0) == (b or 0)                   # freeze: None and 0 both mean "nothing frozen"
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


class _RecipeCombo(QComboBox):
    """The list is read again each time it opens (recipes are saved in another window)."""

    def __init__(self, refill):
        super().__init__()
        self._refill = refill

    def showPopup(self):
        self._refill()
        super().showPopup()


class HypPanel(QWidget):
    changed = pyqtSignal()
    edit_recipe = pyqtSignal(str)                      # open Image Augmentation on this recipe ("" for a new one)

    def __init__(self, settings, optimizer: QComboBox, lr0: QDoubleSpinBox, close_mosaic: QSpinBox, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.optimizer, self.lr0, self.close_mosaic = optimizer, lr0, close_mosaic
        self.env = None
        self.task = "det"
        self.defaults = T.hyp_defaults(None)
        self.ft_box = None
        self.cm_default = 10                           # close_mosaic (the dialog's own spin box)
        self.known = None
        self._flip_saved = None                        # mirror / upside-down values while "text" shows them as 0
        self.w: dict = {}
        self.rows: dict = {}
        self._quiet = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ---------------------------------------------------------------- learning rate & schedule
        lr_box = QGroupBox("Learning rate && schedule")
        lf = QFormLayout(lr_box)
        self.lr_preset = self._preset_combo(T.LR_PRESETS, self.apply_lr)
        lf.addRow("Preset", self.lr_preset)
        lf.addRow("Optimizer", self.optimizer)
        self.lr0.setToolTip("The learning rate at the start (after warm-up). SGD ≈ 0.01, Adam / AdamW ≈ 0.001; "
                            "fine-tuning: lower")
        lf.addRow("Start learning rate", self.lr0)
        self.lr_form = lf
        for key in [h[0] for h in T.HYP if h[2] == "lr"]:
            self._add(lf, key)
        self.auto_note = QLabel("Optimizer auto: Ultralytics picks the optimizer and its learning rate from the run "
                                "length — start learning rate and momentum apply once an optimizer is chosen.")
        self.auto_note.setWordWrap(True)
        utils.style(self.auto_note, "QLabel{color:@muted;}")
        lf.addRow("", self.auto_note)
        self.tune_hint = QLabel("")
        self.tune_hint.setWordWrap(True)
        self.tune_hint.setTextFormat(Qt.TextFormat.RichText)
        self.tune_hint.linkActivated.connect(lambda _l: self.apply_lr("finetune"))
        self.tune_hint.setVisible(False)
        lf.addRow("", self.tune_hint)
        lay.addWidget(lr_box)

        # ---------------------------------------------------------------- augmentation
        aug_box = QGroupBox("Augmentation")
        af = QFormLayout(aug_box)
        self.aug_preset = self._preset_combo(T.AUG_PRESETS, self.apply_aug)
        af.addRow("Preset", self.aug_preset)
        self.text_safe = QCheckBox("Text in the images: no flips")
        self.text_safe.setToolTip("Plates, signs, digits: mirrored or upside-down text is not real text — the model "
                                  "would learn characters that never occur")
        self.text_safe.toggled.connect(self._text_toggled)
        af.addRow("", self.text_safe)
        aug_keys = [h[0] for h in T.HYP if h[2] == "aug"]
        for key in aug_keys[:4]:                      # flips, mosaic, mixup …
            self._add(af, key)
        self.close_mosaic.setToolTip("Mosaic is switched off for the last N epochs (the model finishes on whole, "
                                     "natural images). 0: mosaic to the end")
        self.close_mosaic.setSuffix(" epochs")
        af.addRow("Mosaic off for the last", self.close_mosaic)
        for key in aug_keys[4:]:
            self._add(af, key)
        lay.addWidget(aug_box)

        # ---------------------------------------------------------------- fine-tuning a classifier
        self.ft_box = QGroupBox("Fine-tuning && improving (classifiers)")
        ff = QFormLayout(self.ft_box)
        for key in [h[0] for h in T.HYP if h[2] == "ft"]:
            self._add(ff, key)
        self.ft_note = QLabel("")
        self.ft_note.setWordWrap(True)
        utils.style(self.ft_note, "QLabel{color:@muted;}")
        ff.addRow(self.ft_note)
        ff.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.ft_box.setVisible(False)
        lay.addWidget(self.ft_box)

        # ---------------------------------------------------------------- a recipe from Image Augmentation
        rec_box = QGroupBox("Extra effects (Image Augmentation recipe)")
        rf = QFormLayout(rec_box)
        self.recipe_box = _RecipeCombo(self._fill_recipes)
        self.recipe_box.setMinimumWidth(90)
        self.recipe_box.setToolTip("Weather, lighting, blur, noise, compression, occlusion … from a recipe built in "
                                   "Image Augmentation — applied to every training image in memory, differently "
                                   "each epoch. Nothing is written: no disk space, however large the dataset")
        self.recipe_edit = utils.newButton("Edit…", icon="tool-augment", slot=lambda: self.edit_recipe.emit(
            self.recipe_box.currentData() or ""), tip="Open Image Augmentation: see this recipe on your images, "
                                                      "change it, make a new one")
        rrow = QHBoxLayout()
        rrow.addWidget(self.recipe_box, 1)
        rrow.addWidget(self.recipe_edit)
        rf.addRow("Recipe", rrow)
        self.recipe_share = QSpinBox()
        self.recipe_share.setRange(5, 100)
        self.recipe_share.setSingleStep(5)
        self.recipe_share.setSuffix(" % of the images")
        self.recipe_share.setMinimumWidth(90)
        self.recipe_share.setToolTip("The rest goes through with Ultralytics' own augmentation only")
        rf.addRow("Apply to", self.recipe_share)
        self.recipe_note = QLabel("")
        self.recipe_note.setWordWrap(True)
        utils.style(self.recipe_note, "QLabel{color:@muted;}")
        rf.addRow(self.recipe_note)
        rf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        lay.addWidget(rec_box)
        self._fill_recipes()
        self.recipe_box.currentIndexChanged.connect(lambda _i: self._after())
        self.recipe_share.valueChanged.connect(lambda _v: self._after())

        foot = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        utils.style(self.summary, "QLabel{color:@muted;}")
        foot.addWidget(self.summary, 1)
        reset = utils.newButton("Reset all", icon="undo", slot=self.reset,
                                tip="Every setting here back to its default")
        foot.addWidget(reset, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(foot)

        for w in (self.lr_preset, self.aug_preset, self.optimizer, self.lr0, self.close_mosaic):
            w.setMinimumWidth(90)
        for f in (lf, af):
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.optimizer.currentTextChanged.connect(lambda _t: self._edited("lr"))
        self.lr0.valueChanged.connect(lambda _v: self._edited("lr"))
        self.close_mosaic.valueChanged.connect(lambda _v: self._edited("aug"))
        self._restore()

    # ------------------------------------------------------------------ building
    def _preset_combo(self, presets, apply):
        cb = QComboBox()
        for key, p in presets.items():
            cb.addItem(p["title"], key)
            cb.setItemData(cb.count() - 1, p["tip"], Qt.ItemDataRole.ToolTipRole)
            cb.setItemData(cb.count() - 1, p.get("tasks"), Qt.ItemDataRole.UserRole + 1)
        cb.addItem("Custom", CUSTOM)
        cb.setItemData(cb.count() - 1, "Changed by hand", Qt.ItemDataRole.ToolTipRole)
        cb.activated.connect(lambda i, c=cb, f=apply: f(c.itemData(i)) if c.itemData(i) != CUSTOM else None)
        return cb

    def _add(self, form: QFormLayout, key: str):
        _k, label, group, kind, lo, hi, step, dec, suffix, default, tip, _tasks = T.HYP_ROW[key]
        if kind == "choice":
            w = QComboBox()
            w.addItem("Linear", False)
            w.addItem("Cosine", True)
            w.currentIndexChanged.connect(lambda _i, g=group: self._edited(g))
        elif kind == "pick":
            w = QComboBox()
            for val, text in T.PICKS[key]:
                w.addItem(text, val)
            w.currentIndexChanged.connect(lambda _i, g=group: self._edited(g))
        elif kind == "int":
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(step))
            w.setSuffix(suffix)
            if key == "freeze":
                w.setSpecialValueText("none")
            w.valueChanged.connect(lambda _v, g=group: self._edited(g))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(int(dec))
            w.setRange(float(lo), float(hi))
            w.setSingleStep(float(step))
            w.setSuffix(suffix)
            w.valueChanged.connect(lambda _v, g=group: self._edited(g))
        w.setToolTip(tip)
        w.setMinimumWidth(90)                          # the column stays as narrow as the rest of the dialog
        lab = QLabel(label)
        lab.setToolTip(tip)
        form.addRow(lab, w)
        self.w[key] = w
        self.rows[key] = lab
        self._set(key, default)

    # ------------------------------------------------------------------ values
    def _get(self, key):
        w = self.w[key]
        if isinstance(w, QComboBox) and T.HYP_ROW[key][3] == "pick":
            return w.currentData()
        if isinstance(w, QComboBox):
            return bool(w.currentData())
        if isinstance(w, QSpinBox):
            return w.value() or None if key == "freeze" else w.value()
        return round(w.value(), 6)

    def _set(self, key, v):
        w = self.w[key]
        w.blockSignals(True)
        try:
            if isinstance(w, QComboBox) and T.HYP_ROW[key][3] == "pick":
                i = w.findData(v)
                w.setCurrentIndex(i if i >= 0 else 0)
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(1 if v else 0)
            elif isinstance(w, QSpinBox):
                w.setValue(int(v or 0) if not isinstance(v, (list, tuple)) else 0)
            else:
                w.setValue(float(v or 0.0))
        except (TypeError, ValueError):
            pass
        finally:
            w.blockSignals(False)

    def _usable(self, key) -> bool:
        tasks = T.HYP_ROW[key][11]
        if tasks and self.task not in tasks:
            return False
        if self.known is not None and key not in self.known and self.task != "clst":
            return False
        if key in ("fliplr", "flipud") and self.text_safe.isChecked():
            return False
        return True

    def values(self, full: bool = False) -> dict:
        """What goes to the trainer: the settings that differ from the defaults (Ultralytics) — or, full=True, every
        usable one (the PyTorch classifier trainer)."""
        out = {}
        for key in T.HYP_KEYS:
            if key not in self.w or not self._usable(key):
                continue
            v = self._get(key)
            if full or not _same(v, self.defaults.get(key)):
                out[key] = v
        if self.text_safe.isChecked():
            for key in ("fliplr", "flipud"):
                if self.known is None or key in self.known:
                    if not _same(0.0, self.defaults.get(key)):
                        out[key] = 0.0
        return out

    # ------------------------------------------------------------------ recipe (Image Augmentation)
    def _fill_recipes(self, select=None):
        from AIO_QC.core import augment as A
        from AIO_QC.core import augment_ops as G
        cb = self.recipe_box
        want = select if select is not None else (cb.currentData() if cb.count() else
                                                  (self.settings.value("train/recipe", "") or ""))
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("None — Ultralytics' own augmentation only", "")
        saved = A.recipe_names()
        for n in saved:
            cb.addItem(n, n)
        for n in G.presets():
            if n not in saved:
                cb.addItem(f"{n}  (preset)", n)
        i = cb.findData(want or "")
        cb.setCurrentIndex(i if i >= 0 else 0)
        cb.blockSignals(False)

    def refresh_recipes(self):
        """A recipe was saved / deleted in Image Augmentation."""
        self._fill_recipes()
        self._after()

    def set_recipe(self, name: str | None, share: int | None = None):
        """Train with this recipe (from Image Augmentation ▸ Use in Train Model)."""
        self._fill_recipes(select=name or "")
        if share is not None:
            self.recipe_share.setValue(int(share))
        self._after()

    def augment(self) -> dict | None:
        """{"recipe": …, "share": 0–1} for the training runner, or None."""
        name = self.recipe_box.currentData()
        if not name:
            return None
        from AIO_QC.core import augment as A
        r = A.load_recipe(name)
        if r is None or not any(v["on"] for v in r["ops"].values()):
            return None
        return {"recipe": r, "share": self.recipe_share.value() / 100.0}

    def _recipe_text(self) -> str:
        name = self.recipe_box.currentData()
        if not name:
            return "Build one in Image Augmentation (Models tab) and see it on your own images first."
        from AIO_QC.core import augment as A
        from AIO_QC.core import augment_ops as G
        r = A.load_recipe(name)
        if r is None:
            return "⚠ This recipe is no longer there."
        on = [k for k, v in r["ops"].items() if v["on"]]
        if not on:
            return "⚠ Nothing is switched on in this recipe."
        geo = [k for k in on if G.OPS[k]["kind"] in ("matrix", "dense")]
        libs = (self.env or {}).get("aug_libs")
        missing = sorted({G.OPS[k]["needs"] for k in on if G.OPS[k]["needs"] and libs is not None
                          and not libs.get(G.OPS[k]["needs"])})
        how = {"one": "one effect per image", "each": "every effect by its chance"}.get(
            r["mode"], f"{r['min']}–{r['max']} effects per image")
        txt = f"{len(on)} effects, {how}, in memory — no files."
        if geo:
            txt += f" Its {len(geo)} geometric effects move boxes and masks with the image (on top of the " \
                   "settings above: keep both mild)."
        if self.text_safe.isChecked() and any(k in on for k in ("fliplr", "flipud", "rot90")):
            txt += " ⚠ It mirrors / turns images, and the classes look like text — untick those in the recipe."
        if missing:
            n = sum(1 for k in on if G.OPS[k]["needs"] in missing)
            txt += f" ⚠ {n} of them need {' and '.join(missing)} in the training environment (pip install " \
                   f"{' '.join(missing)} there) — until then they are left out."
        return txt

    # ------------------------------------------------------------------ presets
    def apply_aug(self, key):
        p = T.AUG_PRESETS.get(key)
        if p is None:
            return
        vals = T.preset_values(p, self.task)
        self._unzero_flips()
        self._quiet = True
        try:
            for k in [h[0] for h in T.HYP if h[2] == "aug"]:
                self._set(k, vals.get(k, self.defaults.get(k)))
            self.close_mosaic.setValue(int(vals.get("close_mosaic", self.cm_default)))
        finally:
            self._quiet = False
        self._select(self.aug_preset, key)
        self._after()

    def apply_lr(self, key):
        p = T.LR_PRESETS.get(key)
        if p is None:
            return
        vals = T.preset_values(p, self.task)
        self._quiet = True
        try:
            i = self.optimizer.findText(str(vals.get("optimizer", "auto")))
            self.optimizer.setCurrentIndex(max(0, i))
            self.lr0.setValue(float(vals.get("lr0", 0.0)))
            for k in [h[0] for h in T.HYP if h[2] in ("lr", "ft")]:
                self._set(k, vals.get(k, self.defaults.get(k)))
        finally:
            self._quiet = False
        self._select(self.lr_preset, key)
        self._after()

    def reset(self):
        self.recipe_box.setCurrentIndex(0)
        self.apply_lr("default")
        self.apply_aug("default")

    @staticmethod
    def _select(cb, key):
        i = cb.findData(key)
        cb.blockSignals(True)
        cb.setCurrentIndex(i if i >= 0 else cb.findData(CUSTOM))
        cb.blockSignals(False)

    def _edited(self, group):
        if self._quiet:
            return
        self._select(self.lr_preset if group in ("lr", "ft") else self.aug_preset, CUSTOM)
        self._after()

    def _text_toggled(self, on):
        self._after()

    # ------------------------------------------------------------------ context
    def _unzero_flips(self):
        """The real mirror / upside-down values back in their boxes (before a preset or new defaults change them;
        the next refresh shows 0 again while "text" is ticked)."""
        if self._flip_saved is not None:
            for k, v in self._flip_saved.items():
                self._set(k, v)
            self._flip_saved = None

    def set_env(self, env):
        self.env = env
        self._unzero_flips()
        self._quiet = True
        try:
            tick = self.text_safe.isChecked()
            self.text_safe.blockSignals(True)
            self.text_safe.setChecked(False)           # values() with the user's own flips
            before = self.values()
            self.text_safe.setChecked(tick)
            self.text_safe.blockSignals(False)
        finally:
            self._quiet = False
        self.defaults = T.hyp_defaults(env, self.task)
        self.known = T.supported_keys(env)
        cm = ((env or {}).get("cfg") or {}).get("close_mosaic")
        if isinstance(cm, int) and not isinstance(cm, bool):
            if self.close_mosaic.value() == self.cm_default:
                self.close_mosaic.blockSignals(True)
                self.close_mosaic.setValue(cm)
                self.close_mosaic.blockSignals(False)
            self.cm_default = cm
        self._quiet = True
        try:                                           # keep what was chosen; untouched rows show the new defaults
            for key in T.HYP_KEYS:
                if key in self.w:
                    self._set(key, before.get(key, self.defaults.get(key)))
        finally:
            self._quiet = False
        self._after()

    def set_task(self, task):
        """The settings follow the task. Switching to or from the PyTorch classifier trainer ("clst") swaps the
        defaults: a row still at the old default shows the new one; a row the user changed keeps its value."""
        task = task or "det"
        if task != self.task and "clst" in (task, self.task):
            old, new = self.defaults, T.hyp_defaults(self.env, task)
            self._quiet = True
            try:
                for key in self.w:
                    if _same(self._get(key), old.get(key)):
                        self._set(key, new.get(key))
            finally:
                self._quiet = False
            self.defaults = new
        self.task = task
        cls = task in T.CLS
        if self.ft_box is not None:
            self.ft_box.setVisible(cls)
        for cb in (self.lr_preset, self.aug_preset):          # presets made for one task only
            m = cb.model()
            for i in range(cb.count()):
                tasks = cb.itemData(i, Qt.ItemDataRole.UserRole + 1)
                it = m.item(i) if hasattr(m, "item") else None
                if it is not None:
                    it.setEnabled(not tasks or task in tasks)
        self._after()

    def set_names(self, names):
        """Each dataset decides: class names that look like text (plates, characters) tick "Text in the images"."""
        guess = T.text_like(names)
        self.text_safe.blockSignals(True)
        self.text_safe.setChecked(guess)
        self.text_safe.blockSignals(False)
        self.text_safe.setToolTip(("Ticked because the class names look like text (plates, characters) — untick if "
                                   "the text never matters. " if guess else "")
                                  + "Mirrored or upside-down text is not real text: the model would learn characters "
                                    "that never occur")
        self._after()

    def set_finetune_hint(self, on: bool):
        show = on and self.lr_preset.currentData() != "finetune"
        self.tune_hint.setText(f"Starting from a trained model? <a href='#' style='color:{utils.hexc('accent')}'>"
                               "Use the fine-tuning schedule</a> (small learning rate, backbone frozen)." if show
                               else "")
        self.tune_hint.setVisible(show)

    # ------------------------------------------------------------------ state
    def _after(self):
        if self.text_safe.isChecked() and self._flip_saved is None and "fliplr" in self.w:
            self._flip_saved = {k: self._get(k) for k in ("fliplr", "flipud")}
            for k in ("fliplr", "flipud"):
                self._set(k, 0.0)                      # what will happen: no flips
        elif not self.text_safe.isChecked() and self._flip_saved is not None:
            for k, v in self._flip_saved.items():
                self._set(k, v)
            self._flip_saved = None
        auto = self.optimizer.currentText() == "auto"
        self.lr0.setEnabled(not auto)
        self.auto_note.setVisible(auto)
        self.auto_note.setText(
            "Optimizer auto: AdamW, its learning rate from the batch size (0.0005 at batch 64 from pretrained "
            "weights)." if self.task == "clst" else
            "Optimizer auto: Ultralytics picks the optimizer and its learning rate from the run length — start "
            "learning rate and momentum apply once an optimizer is chosen.")
        cls = self.task in T.CLS
        self.close_mosaic.setEnabled(not cls)
        if self.ft_box is not None and cls:
            self.ft_note.setText(
                "YOLO classifiers (Ultralytics) use the dropout here; the rest is for the other architectures."
                if self.task == "cls" else
                "Starting from your trained model (an earlier run's best.pt): its classes are matched by name — "
                "they keep what they learned, new classes start fresh.")
        for key, w in self.w.items():
            ok = self._usable(key)
            if key == "momentum":
                ok = ok and not auto
            w.setEnabled(ok)
            self.rows[key].setEnabled(ok)
            tip = T.HYP_ROW[key][10]
            if self.known is not None and key not in self.known and self.task != "clst":
                tip = f"Not in this environment's Ultralytics {(self.env or {}).get('ultralytics') or ''} " \
                      "(a newer one has it)"
            elif T.HYP_ROW[key][11] and self.task not in T.HYP_ROW[key][11]:
                tip += " — not for this task" + (" (a YOLO classifier — the other architectures have it)"
                                                 if self.task == "cls" and "clst" in T.HYP_ROW[key][11] else "")
            elif key in ("fliplr", "flipud") and self.text_safe.isChecked():
                tip = "Off: text in the images"
            w.setToolTip(tip)
        if self.lr_preset.currentData() == "finetune":
            self.tune_hint.setVisible(False)
        v = self.values()
        extra = []
        if self.optimizer.currentText() != "auto":
            extra.append(f"optimizer={self.optimizer.currentText()}")
            if self.lr0.value() > 0:
                extra.append(f"lr0={self.lr0.value():g}")
        if self.close_mosaic.value() != self.cm_default and self.task not in T.CLS:
            extra.append(f"close_mosaic={self.close_mosaic.value()}")
        bits = extra + [f"{k}={json.dumps(x) if isinstance(x, bool) else x}" for k, x in v.items()]
        if self.recipe_box.currentData():
            bits.append(f"recipe “{self.recipe_box.currentData()}”")
        self.recipe_share.setEnabled(bool(self.recipe_box.currentData()))
        self.recipe_note.setText(self._recipe_text())
        self.summary.setText(("Changed from the defaults: " + ", ".join(bits)) if bits else "All defaults.")
        self._save()
        self.changed.emit()

    def _save(self):
        try:
            vals = {k: self._get(k) for k in self.w}
            vals.update(self._flip_saved or {})        # "text" is per dataset: remember the user's own values
            self.settings.setValue("train/hyp", json.dumps(vals))
            self.settings.setValue("train/hyp_aug", self.aug_preset.currentData())
            self.settings.setValue("train/hyp_lr", self.lr_preset.currentData())
            self.settings.setValue("train/recipe", self.recipe_box.currentData() or "")
            self.settings.setValue("augment/share", int(self.recipe_share.value()))
            self.settings.setValue("train/hyp_opt", self.optimizer.currentText())
            self.settings.setValue("train/hyp_lr0", float(self.lr0.value()))
            self.settings.setValue("train/hyp_close_mosaic", int(self.close_mosaic.value()))
        except Exception as _exc:
            _ignored(_exc)

    def _restore(self):
        self._quiet = True
        try:
            try:
                saved = json.loads(self.settings.value("train/hyp", "") or "{}")
            except (TypeError, ValueError):
                saved = {}
            for k, v in saved.items():
                if k in self.w:
                    self._set(k, v)
            try:
                self.recipe_share.setValue(int(self.settings.value("augment/share", 100) or 100))
            except (TypeError, ValueError):
                pass
            opt = self.settings.value("train/hyp_opt", "auto") or "auto"
            self.optimizer.setCurrentIndex(max(0, self.optimizer.findText(str(opt))))
            try:
                self.lr0.setValue(float(self.settings.value("train/hyp_lr0", 0.0) or 0.0))
                self.close_mosaic.setValue(int(self.settings.value("train/hyp_close_mosaic", 10) or 0))
            except (TypeError, ValueError):
                pass
        finally:
            self._quiet = False
        self._select(self.aug_preset, self.settings.value("train/hyp_aug", "default") or "default")
        self._select(self.lr_preset, self.settings.value("train/hyp_lr", "default") or "default")
        self._after()
