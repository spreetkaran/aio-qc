# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Models ▸ Improve Model — one guided loop: 1 the model (a finished training run, or any .pt) · 2 check it on this
dataset (Model Check with just that model: where is it weak, and why) · 3 label more (flag the images where model
and labels disagree most; pick new images from a pool where the model is unsure, with empty label files so they can
be labelled in place) · 4 retrain (fine-tune from it). Every check adds a round to the history, so each round shows
whether the model got better. Kept per dataset (.aio_qc/improve.json, core.improve).
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from AIO_QC.core import improve as IM
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import newButton, newIcon, style as apply_style
from AIO_QC.core.diag import ignored as _ignored
from .base_scan import readable_table


def _pct(v):
    return "—" if v is None else f"{v:.2f}"


class ImproveDialog(QDialog):
    TITLE = "Improve Model"

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-improve"))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.st = IM.load(self.root)
        self.hard: list = []
        self._mc = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        head = QLabel("<b>Train → check → label more → retrain.</b> Each check is one round in the history below, "
                      "so you see whether the model got better.")
        head.setWordWrap(True)
        lay.addWidget(head)

        # 1 · model
        g1 = QGroupBox("1 · The model")
        l1 = QHBoxLayout(g1)
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(380)
        self.model_box.currentIndexChanged.connect(self._model_changed)
        l1.addWidget(self.model_box, 1)
        l1.addWidget(newButton("Other .pt…", icon="open", slot=self._browse, tip="Any YOLO weights file"))
        l1.addWidget(newButton("", icon="reset-default", slot=self.refresh_models, tip="Look for new runs"))
        self.train_btn = newButton("Train a first model…", icon="tool-train", slot=lambda: self.app.open_training())
        l1.addWidget(self.train_btn)
        lay.addWidget(g1)

        # 2 · check
        g2 = QGroupBox("2 · Check it on this dataset")
        l2 = QVBoxLayout(g2)
        r2 = QHBoxLayout()
        self.check_btn = newButton("Run Model Check", icon="tool-model-check", slot=self.check,
                                   tip="Model Check with just this model: every image of the dataset (predictions "
                                       "are cached, so a re-check is quick)")
        r2.addWidget(self.check_btn)
        self.check_note = QLabel("")
        self.check_note.setWordWrap(True)
        r2.addWidget(self.check_note, 1)
        l2.addLayout(r2)
        self.classes = QTableWidget(0, 6)
        self.classes.setHorizontalHeaderLabels(["Class", "Labels", "Precision", "Recall", "mAP50-95", "What to do"])
        self.classes.verticalHeader().setVisible(False)
        self.classes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.classes.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.classes.setWordWrap(True)
        readable_table(self.classes)
        self.classes.cellDoubleClicked.connect(self._class_activated)
        l2.addWidget(self.classes, 1)
        lay.addWidget(g2, 2)

        # 3 · label more
        g3 = QGroupBox("3 · Label more")
        l3 = QGridLayout(g3)
        self.n_hard = QSpinBox()
        self.n_hard.setRange(5, 100000)
        self.n_hard.setValue(int(self.app.settings.value("improve/n_hard", 100)))
        self.n_hard.valueChanged.connect(lambda v: self.app.settings.setValue("improve/n_hard", v))
        self.flag_btn = newButton("Flag the hardest images", icon="review-flag", slot=self.flag_hardest,
                                  tip="The files where the model and the labels disagree most — wrong or missing "
                                      "labels are often the quickest win")
        self.review_btn = newButton("Review them", icon="tool-review-grid", slot=self.review_flagged)
        l3.addWidget(QLabel("Fix:"), 0, 0)
        l3.addWidget(self.n_hard, 0, 1)
        l3.addWidget(self.flag_btn, 0, 2)
        l3.addWidget(self.review_btn, 0, 3)
        self.pick_btn = newButton("Pick new images from a pool…", icon="review-next", slot=self.pick,
                                  tip="Preparation ▸ Pick images to label, set up for this model: the most varied "
                                      "images it is unsure about, with empty label files so you can label them here")
        l3.addWidget(QLabel("Add:"), 1, 0)
        l3.addWidget(self.pick_btn, 1, 1, 1, 3)
        self.label_note = QLabel("")
        self.label_note.setWordWrap(True)
        apply_style(self.label_note, "color:@muted;")
        l3.addWidget(self.label_note, 2, 0, 1, 4)
        l3.setColumnStretch(4, 1)
        lay.addWidget(g3)

        # 4 · retrain
        g4 = QGroupBox("4 · Retrain")
        l4 = QHBoxLayout(g4)
        self.retrain_btn = newButton("Fine-tune from this model…", icon="tool-train", slot=self.retrain,
                                     tip="Train Model with this model as the start and fine-tuning settings")
        self.convert_btn = newButton("Convert to YOLO first…", icon="tool-convert",
                                     slot=lambda: self.app.open_tool("convert_yolo"),
                                     tip="A LabelMe dataset trains from its YOLO copy: convert again after labelling")
        self.extend_btn = newButton("Extend it: new classes / more images…", icon="tool-train", slot=self.extend,
                                    tip="Train Model ▸ Extend: the model keeps its classes (and ids), learns the new "
                                        "ones — old data mixed in, pre-labels, two stages, a before / after check")
        l4.addWidget(self.retrain_btn)
        l4.addWidget(self.extend_btn)
        l4.addWidget(self.convert_btn)
        self.retrain_note = QLabel("")
        self.retrain_note.setWordWrap(True)
        l4.addWidget(self.retrain_note, 1)
        lay.addWidget(g4)

        # history
        g5 = QGroupBox("Rounds")
        l5 = QVBoxLayout(g5)
        self.trend = QLabel("")
        l5.addWidget(self.trend)
        self.history = QTableWidget(0, 6)
        self.history.setHorizontalHeaderLabels(["When", "Model", "Precision", "Recall", "mAP50-95", "Weak classes"])
        self.history.verticalHeader().setVisible(False)
        self.history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        readable_table(self.history)
        l5.addWidget(self.history)
        lay.addWidget(g5, 1)

        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(newButton("Close", slot=self.close))
        lay.addLayout(foot)

        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        from .training_monitor import monitor
        monitor().ended.connect(self._run_ended)
        self.refresh_models()
        self._show_check()
        self._show_history()

    # ------------------------------------------------------------------ data
    @property
    def session(self):
        return self.app.session

    @property
    def root(self) -> Path:
        return Path(self.app.session.root)

    def _save(self):
        try:
            IM.save(self.root, self.st)
        except OSError as exc:
            _ignored(exc)

    def _projects(self) -> list[Path]:
        out = []
        t = (self.app.settings.value("train/project", "") or "").strip()
        if t:
            out.append(Path(t).expanduser())
        td = getattr(self.app, "train_dialog", None)
        if td is not None:
            try:
                out.append(td._project())
            except Exception as exc:
                _ignored(exc)
        out += [self.root / "runs", self.root.parent / "runs"]
        try:
            y = self.app._dataset_yaml_for_training()
            if y is not None:
                out.append(Path(y).parent / "runs")
        except Exception as exc:
            _ignored(exc)
        seen, uniq = set(), []
        for p in out:
            k = str(p.resolve()) if p.exists() else str(p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        return uniq

    def refresh_models(self, select: str | None = None):
        from AIO_QC.core import training as T
        want = select or self.st.get("model") or self.model_box.currentData()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        seen = set()
        for proj in self._projects():
            for r in T.list_runs(proj):
                if not r.get("best_pt") or r.get("mode") == "val" or r["best_pt"] in seen:
                    continue
                seen.add(r["best_pt"])
                score = f" · {r.get('metric', 'mAP50-95')} {r['best']:.3f}" if r.get("best") is not None else ""
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(r["started"] or 0)))
                self.model_box.addItem(f"{r['name']}{score} · {when} · {r['state']}", r["best_pt"])
                self.model_box.setItemData(self.model_box.count() - 1, str(r["dir"]), Qt.ItemDataRole.UserRole + 1)
        if want and want not in seen and Path(want).is_file():
            self.model_box.addItem(Path(want).name + " · " + str(Path(want).parent), want)
        if self.model_box.count() == 0:
            self.model_box.addItem("No trained model yet — train one, or choose a .pt", "")
        i = self.model_box.findData(want) if want else -1
        self.model_box.setCurrentIndex(max(0, i))
        self.model_box.blockSignals(False)
        self._model_changed()

    def model_path(self) -> str:
        return self.model_box.currentData() or ""

    def run_dir(self) -> Path | None:
        d = self.model_box.currentData(Qt.ItemDataRole.UserRole + 1)
        return Path(d) if d else None

    def _model_changed(self, *_):
        p = self.model_path()
        has = bool(p)
        for b in (self.check_btn, self.pick_btn, self.retrain_btn):
            b.setEnabled(has)
        self.train_btn.setVisible(not has)
        if has and p != self.st.get("model"):
            self.st["model"] = p
            self.st["run"] = str(self.run_dir() or "")
            self._save()
        cls = self.session.is_classification
        lm = not self.session.is_yolo and not cls
        self.convert_btn.setVisible(lm)
        self.pick_btn.setVisible(True)                 # classifiers: picks sorted into folders by the model's guess
        self.retrain_note.setText(
            "Fixed classes are used as they are: Train reads the class folders (LabelMe flags: Convert & Split ▸ "
            "Class folders first)." if cls else
            "After labelling, convert to YOLO again so the new labels are in the training copy." if lm else
            "New and fixed labels are used as they are.")
        self._cls_headers(cls)
        self._show_check()

    def _cls_headers(self, cls: bool):
        """A classifier is judged by its accuracy, per class by precision / recall over images."""
        self.classes.setHorizontalHeaderLabels(["Class", "Images" if cls else "Labels", "Precision", "Recall",
                                                "" if cls else "mAP50-95", "What to do"])
        self.classes.setColumnHidden(4, cls)
        self.history.setHorizontalHeaderLabels(["When", "Model", "Precision", "Recall",
                                                "Top-1" if cls else "mAP50-95", "Weak classes"])

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Model weights", str(self.root), "YOLO weights (*.pt *.onnx)")
        if f:
            self.refresh_models(select=f)

    def _run_ended(self, run, state, _status):
        if state == "finished" and self.isVisible():
            best = Path(run) / "weights" / "best.pt"
            self.refresh_models(select=str(best) if best.is_file() else None)

    # ------------------------------------------------------------------ 2 · check
    def check(self):
        from AIO_QC.core import model_registry as R
        p = self.model_path()
        if not p:
            return
        cls = self.session.is_classification
        names, task, imgsz = {}, ("classify" if cls else "detect"), None
        rd = self.run_dir()
        if rd is not None:
            from .train_dialog import TrainDialog
            spec = TrainDialog._spec(rd)
            names = {str(i): n for i, n in enumerate(spec.get("names") or [])}
            task = "classify" if (cls or spec.get("task") == "cls") else \
                "segment" if spec.get("task") == "seg" else "detect"
            imgsz = (spec.get("args") or {}).get("imgsz")
        entry = R.register(p, root=self.root, name=(f"{rd.name} (best)" if rd else Path(p).stem), names=names,
                           task=task, train_imgsz=imgsz, labels=set(self.session.label_totals()))
        self.app.open_tool("model_check")
        d = self.app.open_dialogs.get("model_check")
        if d is None:
            return
        d.focus_models([entry["id"]])
        self._mc = d
        try:
            d.scanFinished.disconnect(self._checked)
        except TypeError:
            pass
        d.scanFinished.connect(self._checked)
        self.check_note.setText("Checking in Model Check …")
        d.start_scan()

    def _checked(self, ok: bool):
        d = self._mc
        if d is None:
            return
        try:
            d.scanFinished.disconnect(self._checked)
        except TypeError:
            pass
        d.focus_models(None)
        if self.session.is_classification:                 # a classifier: accuracy per class, the mistakes
            m = (d.ctx or {}).get("metrics")
            if not ok or not m or not m.get("n"):
                self.check_note.setText(f"Not checked — {getattr(d, 'scan_error', '') or d.status.text()}")
                return
            spots = IM.weak_spots_cls(m)
            vals = [r for r in d.ctx.get("recs", []) if r.get("split") == "val"]
            spots["split"] = "val" if vals else "all"
            self.hard = IM.hardest_cls(d.ctx.get("recs", []))
            self.st["hard"] = [k for k, _n in self.hard[:5000]]
            IM.record_check(self.st, str(self.run_dir() or ""), self.model_path(), spots, m.get("n", 0))
            self._save()
            self._show_check()
            self._show_history()
            self.raise_()
            return
        ev = (d.ctx or {}).get("eval")
        if not ok or ev is None:
            self.check_note.setText(f"Not checked — {d.scan_error or d.status.text()}")
            return
        from AIO_QC.core.evaluation import View
        view = View(split="val") if "val" in ev.splits else View()
        conf, iou = float(d.ctx.get("conf", 0.25)), float(d.ctx.get("iou", 0.5))
        spots = IM.weak_spots(ev, conf, iou, view)
        spots["split"] = view.split or "all"
        self.hard = IM.hardest_images(ev, conf, iou, 100000)
        self.st["hard"] = [k for k, _n in self.hard[:5000]]
        IM.record_check(self.st, str(self.run_dir() or ""), self.model_path(), spots, len(ev.keys))
        self._save()
        self._show_check()
        self._show_history()
        self.raise_()

    def _show_check(self):
        lc = self.st.get("last_check") or {}
        spots = lc.get("spots") if lc.get("model") == self.model_path() else None
        self.classes.setRowCount(0)
        if not spots:
            self.check_note.setText("Not checked yet." if self.model_path() else "")
            self.flag_btn.setEnabled(False)
            return
        where = "the val split" if spots.get("split") == "val" else "every image (no val split)"
        self.check_note.setText(IM.summary_text(spots) + f" Measured on {where}.")
        for c in spots["classes"]:
            r = self.classes.rowCount()
            self.classes.insertRow(r)
            cells = [c["label"], str(c["gt"]), _pct(c["precision"]), _pct(c["recall"]), _pct(c["ap5095"]),
                     c["advice"]]
            for k, t in enumerate(cells):
                it = QTableWidgetItem(t)
                if k in (1, 2, 3, 4):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.classes.setItem(r, k, it)
            if c["weak"] or c["few"]:
                from AIO_QC.utils import theme_color
                for k in range(6):
                    self.classes.item(r, k).setForeground(theme_color("warn"))
        self.classes.resizeColumnsToContents()
        self.classes.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.classes.resizeRowsToContents()
        n = len(self.st.get("hard") or [])
        self.flag_btn.setEnabled(n > 0)
        self.label_note.setText(f"{n:,} image(s) where the model and the labels disagree." if n else "")

    def _class_activated(self, r, _c):
        it = self.classes.item(r, 0)
        if self.session.is_classification and it is not None:     # a classifier: that class, its mistakes first
            gal = self.app.open_class_gallery()
            if gal is not None:
                from AIO_QC.core import cls_review as CR
                gal.show_label(it.text(), sort=CR.SUSPICIOUS)
            return
        d = self.app.open_dialogs.get("model_check")
        if d is not None and it is not None:
            d.show()
            d.raise_()
            if hasattr(d, "tabs") and hasattr(d, "_analysis"):
                d.tabs.setCurrentWidget(d._analysis)

    # ------------------------------------------------------------------ 3 · label more
    def flag_hardest(self):
        keys = [k for k in (self.st.get("hard") or [])[: self.n_hard.value()] if k in self.session.by_key]
        if not keys:
            self.label_note.setText("Nothing to flag — run the check first.")
            return
        self.session.set_review(keys, REVIEW_FLAG, "Improve Model: the model and the labels disagree here")
        self.label_note.setText(f"⚑ {len(keys):,} file(s) flagged — Review them, fix what is wrong (or mark it OK), "
                                "then retrain.")
        self.st.setdefault("done", {})["label"] = time.time()
        self._save()

    def review_flagged(self):
        self.app.open_tool("review_grid")

    def pick(self):
        from .data_prep import PickPage
        self.app.open_data_prep()
        dlg = self.app.open_dialogs.get("data_prep")
        if dlg is None:
            return
        pg = dlg.show_page(PickPage)
        pg.model_path = self.model_path()
        pg.unsure.setChecked(True)
        cls = self.session.is_classification
        pg.stubs.setChecked(not cls)
        pg.sort_cls.setChecked(cls)
        cur = pg.out.text().strip()
        mine = cur and (str(Path(cur).resolve()).startswith(str(self.root.resolve()))
                        or Path(cur).name.startswith(f"{self.root.name}_to_label"))
        if cls and mine and str(Path(cur).resolve()).startswith(str(self.root.resolve())):
            mine = False                             # a classifier's picks never land inside its class folders
        if not mine:                                 # a folder remembered from another dataset: not for these picks
            stamp = time.strftime("%Y%m%d")
            out = (self.root / f"to_label_{stamp}") if not (self.session.is_yolo or cls) else \
                (self.root.parent / f"{self.root.name}_to_label_{stamp}")
            pg.out.setText(str(out))
        if cls:
            self.label_note.setText("Preparation ▸ Pick: choose the pool folder, Preview, then Copy. The picks arrive "
                                    "sorted into folders by your classifier's guess — open that folder, confirm or "
                                    "move them in the Class Gallery, then add them to the dataset.")
            return
        self.label_note.setText("Preparation ▸ Pick: choose the pool folder, Preview, then Copy. The picks get empty "
                                "label files" + (" and join this dataset, flagged “to label”." if not
                                                 self.session.is_yolo else "."))

    # ------------------------------------------------------------------ 4 · retrain
    def retrain(self):
        rd = self.run_dir()
        p = self.model_path()
        if not p:
            return
        self.app.open_training(run=rd) if rd is not None else self.app.open_training()
        td = getattr(self.app, "train_dialog", None)
        if td is None:
            return
        if rd is not None:
            td.current = rd
            td.fine_tune()
        else:
            td.model_box.insertItem(0, f"{Path(p).name}", p)
            td.model_box.setCurrentIndex(0)
            td.hyp.apply_lr("finetune")
        self.st.setdefault("done", {})["retrain"] = time.time()
        self._save()

    def extend(self):
        p = self.model_path()
        if not p:
            return
        self.app.open_training()
        td = getattr(self.app, "train_dialog", None)
        if td is None:
            return
        k = td.model_box.findData(p)
        if k < 0:
            td.model_box.insertItem(0, Path(p).name, p)
            k = 0
        td.model_box.setCurrentIndex(k)
        td.goal.setCurrentIndex(td.goal.findData("classes"))
        self.st.setdefault("done", {})["retrain"] = time.time()
        self._save()

    # ------------------------------------------------------------------ history
    def _show_history(self):
        h = self.st.get("history") or []
        self.history.setRowCount(0)
        for x in reversed(h):
            r = self.history.rowCount()
            self.history.insertRow(r)
            name = Path(x.get("run") or x.get("model") or "").name
            main = x.get("ap5095") if x.get("ap5095") is not None else x.get("top1")
            for k, t in enumerate([x.get("when", ""), name, _pct(x.get("precision")), _pct(x.get("recall")),
                                   _pct(main), ", ".join(x.get("weak") or []) or "—"]):
                self.history.setItem(r, k, QTableWidgetItem(t))
        self.history.resizeColumnsToContents()
        self.history.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.trend.setText(IM.trend(self.st) or ("One round so far." if h else "No rounds yet."))

    def closeEvent(self, e):
        try:
            from .training_monitor import monitor
            monitor().ended.disconnect(self._run_ended)
        except TypeError:
            pass
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)

