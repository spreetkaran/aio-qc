# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🔤 OCR Check — read the text inside text-class boxes with an OCR engine from your own Python environment and
check it: no text · low confidence · validator rules · truncated / loose boxes · several lines · stored-text
mismatch; across the dataset: identity leakage between splits and very frequent identities.

OCR runs once per box (cached by image version + engine settings); thresholds, validators and class settings are
re-evaluated instantly from the cached reads.
"""
from __future__ import annotations

from AIO_QC.core.dataset import DETECTION_TYPES

import csv
import threading
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu,
    QMessageBox, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.core import ocr_langs as OL
from AIO_QC.core import validators as VL
from AIO_QC.core.env_manager import discover_envs
from AIO_QC.core.geometry import to_polygon
from AIO_QC.core.ocr_check import DEFAULT_CLASS, DEFAULT_THRESHOLDS, KINDS, Read, evaluate, summarize
from AIO_QC.core.state import load_ocr_config, save_ocr_config
from AIO_QC.utils import hexc, newButton, style as apply_style, theme_color
from .base_scan import BaseScanDialog, ResultRow, dspin, readable_table
from .card_gallery import Card, CardView, Panel, Shape, box_points
from .inference_service import FuncWorker
from .validator_editor import ValidatorEditor

MODES = [("read", "Read text (fast)"), ("locate", "Read + locate text (box-fit checks)")]
TONE = {"error": "error", "warn": "warning", "muted": "muted"}


class OcrCheckDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "OCR Check", "🔤"
    HELP = ("Reads the text in your text-class boxes with an OCR engine from your own Python environment and checks "
            "it (validators, readability, box fit, stored transcriptions, identity leakage). OCR runs once per box and "
            "is cached; changing thresholds or validators re-checks instantly.")
    COLUMNS = ["Issue", "Class", "Read", "Conf", "Detail", "Shape"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = True

    # ================================================================== parameters
    def build_params(self, form):
        self.cfg = load_ocr_config(self.session.root)
        # ---- environment + engine
        env_row = QWidget()
        el = QHBoxLayout(env_row)
        el.setContentsMargins(0, 0, 0, 0)
        self.python = QComboBox()
        self.python.setEditable(True)
        self.python.setMinimumWidth(320)
        seen = set()
        for p in [self.cfg.get("python")] + [py for _n, py in discover_envs()]:
            if p and p not in seen:
                seen.add(p)
                self.python.addItem(p)
        if self.cfg.get("python"):
            self.python.setCurrentText(self.cfg["python"])
        self.python.lineEdit().setPlaceholderText("Python of the environment that has PaddleOCR / RapidOCR / EasyOCR …")
        el.addWidget(self.python, 1)
        el.addWidget(newButton("Browse…", icon="open", slot=self._browse_python))
        el.addWidget(newButton("Check", icon="review-ok", slot=self._probe,
                               tip="Find the OCR engines installed in this environment"))
        form.addRow("Environment:", env_row)
        self.env_info = QLabel("Choose the environment and press Check.")
        self.env_info.setWordWrap(True)
        apply_style(self.env_info, "color:@muted;")
        form.addRow("", self.env_info)

        eng_row = QWidget()
        rl = QHBoxLayout(eng_row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.engine = QComboBox()
        self.engine.setMinimumWidth(170)
        if self.cfg.get("engine"):
            self.engine.addItem(self.cfg["engine"], self.cfg["engine"])
        old = self.cfg.get("lang")                       # settings from before multi-language support
        self.langs = list(self.cfg.get("langs") or ([old] if old in OL.NAMES else ["en"]))
        self.lang_btn = QToolButton()
        self.lang_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lang_btn.setMenu(QMenu(self.lang_btn))
        self.lang_btn.menu().aboutToShow.connect(self._fill_lang_menu)
        self.lang_btn.setToolTip("Languages to read. The first ticked is the primary language. EasyOCR / Tesseract "
                                 "read several at once; PaddleOCR / RapidOCR use one model per language (see Strategy)")
        self.strategy = QComboBox()
        self.strategy.addItem("Best of all languages", "best")
        self.strategy.addItem("Primary first, others only when unsure (faster)", "fallback")
        self.strategy.setCurrentIndex(max(0, self.strategy.findData(self.cfg.get("strategy", "fallback"))))
        self.strategy.setToolTip("Several languages with a one-model-per-language engine: read every box with every "
                                 "model and keep the most confident read, or read with the primary language and try "
                                 "the others only below the confidence threshold")
        self.engine.currentIndexChanged.connect(lambda _i: self._update_langs())
        self.device = QComboBox()
        self.device.addItems(["CPU", "GPU"])
        self.device.setCurrentIndex(1 if self.cfg.get("gpu") else 0)
        self.mode = QComboBox()
        for k, t in MODES:
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.cfg.get("mode", "read"))))
        self.mode.setToolTip("Read + locate also finds WHERE the text is: needed for truncated / loose / several-lines "
                             "checks (slower)")
        for text, w in (("Engine", self.engine), ("Languages", self.lang_btn), ("Device", self.device),
                        ("Mode", self.mode)):
            rl.addWidget(QLabel(text))
            rl.addWidget(w)
        self.strategy_lbl = QLabel("Strategy")
        rl.addWidget(self.strategy_lbl)
        rl.addWidget(self.strategy)
        rl.addStretch(1)
        form.addRow("OCR:", eng_row)
        self._update_langs()

        # ---- text classes
        self.cls_table = QTableWidget(0, 4)
        self.cls_table.setHorizontalHeaderLabels(["Text class", "Validators (any one must pass)", "Lines per box",
                                                  "Stored text field"])
        self.cls_table.verticalHeader().setVisible(False)
        self.cls_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.cls_table.setMaximumHeight(170)
        self.cls_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        form.addRow("Text classes:", self.cls_table)
        norm = QWidget()
        nl = QHBoxLayout(norm)
        nl.setContentsMargins(0, 0, 0, 0)
        n = self.cfg.get("normalize", DEFAULT_CLASS["normalize"])
        self.upper = QCheckBox("Upper-case")
        self.upper.setChecked(n.get("upper", True))
        self.remove = QLineEdit(n.get("remove", " -._"))
        self.remove.setMaximumWidth(90)
        self.remove.setToolTip("Characters removed before checking (spaces, dashes, dots …)")
        nl.addWidget(self.upper)
        nl.addWidget(QLabel("Ignore characters:"))
        nl.addWidget(self.remove)
        nl.addWidget(newButton("Validators…", icon="tool-validation", slot=self._open_validators,
                               tip="Library of validators: built-in, pattern builder, learn from examples, code"))
        nl.addStretch(1)
        form.addRow("Before checking:", norm)
        self._fill_classes()

        self.add_action("🔧 Fix in editor", self._fix_in_editor, "Open the files of the ticked rows in labelme / labelImg")
        self.add_action("⤓ Export reads", self._export_reads, "CSV of every read: file, shape, class, text, confidence, "
                                                              "validator, issues")
        self.engine_versions: dict = {}

    # ------------------------------------------------------------------ languages
    def _fill_lang_menu(self):
        m = self.lang_btn.menu()
        m.clear()
        eng = self.engine.currentData() or "paddleocr"
        for key, name, *_codes in OL.LANGS:
            a = m.addAction(name)
            a.setCheckable(True)
            a.setChecked(key in self.langs)
            supported = OL.code(eng, key) is not None
            a.setEnabled(supported or key in self.langs)
            if not supported:
                a.setToolTip(f"{self.engine.currentText() or eng} has no model for {name}")
            if key == (self.langs[0] if self.langs else None):
                a.setText(f"{name}   (primary)")
            a.toggled.connect(lambda on, k=key: self._toggle_lang(k, on))

    def _toggle_lang(self, key, on):
        if on and key not in self.langs:
            self.langs.append(key)
        elif not on and key in self.langs and len(self.langs) > 1:
            self.langs.remove(key)
        self._update_langs()

    def _update_langs(self):
        eng = self.engine.currentData() or ""
        _codes, missing = OL.resolve(eng, self.langs) if eng else ([], [])
        txt = OL.describe(self.langs)
        self.lang_btn.setText((txt if len(txt) <= 34 else f"{len(self.langs)} languages") + (" ⚠" if missing else ""))
        self.lang_btn.setToolTip(f"{txt}" + (f"\n⚠ {self.engine.currentText()} cannot read: {', '.join(missing)}"
                                             if missing else ""))
        several = len(self.langs) > 1 and eng not in OL.NATIVE_MULTI
        self.strategy.setVisible(several)
        self.strategy_lbl.setVisible(several)

    def _fill_classes(self):
        self.validators = VL.discover(self.session.root)
        labels = sorted(self.session.label_totals(raw=False))
        saved = self.cfg.get("classes", {})
        if not saved:                                    # sensible first guess: names that say "text" / "line"
            saved = {l: {} for l in labels if any(w in l.lower() for w in ("text", "line", "ocr", "word", "char"))}
        self.cls_table.setRowCount(len(labels))
        self._cls_widgets = {}
        for r, lab in enumerate(labels):
            c = {**DEFAULT_CLASS, **saved.get(lab, {})}
            it = QTableWidgetItem(lab)
            it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(Qt.CheckState.Checked if lab in saved else Qt.CheckState.Unchecked)
            self.cls_table.setItem(r, 0, it)
            vb = QToolButton()
            vb.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            vb.setProperty("chosen", list(c["validators"]))
            vb.setMenu(QMenu(vb))
            vb.menu().aboutToShow.connect(lambda b=vb: self._fill_validator_menu(b))
            self._label_validators(vb)
            self.cls_table.setCellWidget(r, 1, vb)
            lines = QComboBox()
            lines.addItem("single line", "single")
            lines.addItem("several lines", "multi")
            lines.setCurrentIndex(0 if c["lines"] == "single" else 1)
            self.cls_table.setCellWidget(r, 2, lines)
            tf = QComboBox()
            tf.setEditable(True)
            tf.addItems(["", "description", "text"])
            tf.setCurrentText(c.get("text_field", ""))
            tf.setToolTip("If your annotations store the transcription in a shape field (LabelMe 'description', or a "
                          "custom key), OCR is compared with it")
            self.cls_table.setCellWidget(r, 3, tf)
            self._cls_widgets[lab] = (it, vb, lines, tf)
        self.cls_table.resizeColumnToContents(0)

    def _label_validators(self, b):
        chosen = b.property("chosen") or []
        titles = [self.validators[n].title if n in self.validators else f"{n} (missing)" for n in chosen]
        b.setText(" · ".join(titles) if titles else "none — readability / box checks only")
        b.setToolTip("Any one of these must accept the text")

    def _fill_validator_menu(self, b):
        m = b.menu()
        m.clear()
        chosen = set(b.property("chosen") or [])
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
                a.toggled.connect(lambda on, n=v.name, bb=b: self._toggle_validator(bb, n, on))
        m.addSeparator()
        m.addAction("New / edit validators…", self._open_validators)

    def _toggle_validator(self, b, name, on):
        chosen = [n for n in (b.property("chosen") or []) if n != name] + ([name] if on else [])
        b.setProperty("chosen", chosen)
        self._label_validators(b)
        self._reevaluate()

    def _collect(self) -> dict:
        classes = {}
        for lab, (it, vb, lines, tf) in self._cls_widgets.items():
            if it.checkState() == Qt.CheckState.Checked:
                classes[lab] = {"validators": list(vb.property("chosen") or []), "lines": lines.currentData(),
                                "text_field": tf.currentText().strip(),
                                "normalize": {"upper": self.upper.isChecked(), "remove": self.remove.text(),
                                              "strip": True}}
        return {"python": self.python.currentText().strip(), "engine": self.engine.currentData(),
                "langs": list(self.langs), "strategy": self.strategy.currentData(), "gpu": self.device.currentIndex() == 1,
                "mode": self.mode.currentData(), "classes": classes,
                "normalize": {"upper": self.upper.isChecked(), "remove": self.remove.text(), "strip": True},
                "thresholds": self._thr()}

    # ================================================================== environment
    def _browse_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Python executable of the OCR environment", "",
                                              "Python (python*);;All files (*)")
        if path:
            self.python.setCurrentText(path)
            self._probe()

    def _probe(self):
        py = self.python.currentText().strip()
        if not py or not Path(py).exists():
            self.env_info.setText("That Python executable does not exist.")
            return
        self.env_info.setText("Checking the environment …")
        w = FuncWorker(lambda: self.app.ocr.probe(py), self)
        w.done.connect(self._probed)
        w.failed.connect(lambda m: self.env_info.setText(f"✗ {m}"))
        w.start()
        self._probe_worker = w

    def _probed(self, info):
        eng = info.get("engines", {})
        cur = self.engine.currentData() or self.cfg.get("engine")
        self.engine.clear()
        for key in ("paddleocr", "rapidocr", "easyocr", "tesseract"):
            e = eng.get(key)
            if e:
                self.engine.addItem(f"{e['title']} {e['version']}" + (" ⚠" if e.get("error") else ""), key)
                self.engine_versions[key] = e["version"]
        i = self.engine.findData(cur)
        self.engine.setCurrentIndex(i if i >= 0 else 0)
        self.device.model().item(1).setEnabled(bool(info.get("gpu")))
        if not info.get("gpu"):
            self.device.setCurrentIndex(0)
        found = ", ".join(f"{e['title']} {e['version']}" for e in eng.values()) or "no OCR engine"
        errs = "; ".join(e["error"] for e in eng.values() if e.get("error"))
        self.env_info.setText(f"Python {info.get('python')} · {found} · GPU: {info.get('gpu') or 'none'}"
                              + (f"<br><span style='color:{hexc('warn')}'>{errs}</span>" if errs else "")
                              + ("" if eng else "<br>Install one, e.g. <code>pip install rapidocr_onnxruntime</code> "
                                                "(light, CPU) or <code>pip install paddleocr paddlepaddle</code>."))

    # ================================================================== live thresholds + tabs
    def post_init(self):
        t = {**DEFAULT_THRESHOLDS, **self.cfg.get("thresholds", {})}
        live = QWidget()
        lv = QHBoxLayout(live)
        lv.setContentsMargins(0, 0, 0, 0)
        self.min_conf = dspin(0.0, 1.0, t["min_conf"], 2, 0.05)
        self.trunc = dspin(0.0, 1.0, t["trunc_tol"], 2, 0.02)
        self.loose = dspin(0.0, 1.0, t["loose"], 2, 0.05)
        self.min_h = dspin(2, 200, t["min_h"], 0, 1)
        title = QLabel("⚡ Live")
        title.setStyleSheet("font-weight:700;")
        lv.addWidget(title)
        for text, w, tip in (("confidence ≥", self.min_conf, "Reads below this are 'low confidence'"),
                             ("truncation >", self.trunc, "Text past the box edge, as a fraction of the box height"),
                             ("loose <", self.loose, "Text area / box area below this = loose box"),
                             ("min height px", self.min_h, "Smaller boxes are skipped as unreadable (next scan)")):
            lab = QLabel(text)
            lab.setToolTip(tip)
            w.setToolTip(tip)
            lv.addWidget(lab)
            lv.addWidget(w)
        lv.addStretch(1)
        self.table_host.insertWidget(0, live)
        for w in (self.min_conf, self.trunc, self.loose):
            w.valueChanged.connect(lambda _v: self._reeval_timer.start())
        self._reeval_timer = QTimer(self, singleShot=True, interval=250, timeout=self._reevaluate)

        self.init_marks()
        self.cards = CardView(self, n_panels=1)
        self.ident_tbl = QTableWidget(0, 4)
        self.ident_tbl.setHorizontalHeaderLabels(["Text (identity)", "Kind", "Splits", "Files"])
        self.ident_tbl.verticalHeader().setVisible(False)
        self.ident_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ident_tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        readable_table(self.ident_tbl)
        self.ident_tbl.cellDoubleClicked.connect(self._open_identity)
        self.summary_lbl = QLabel("Run a scan to see the summary.")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.summary_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        ident_host = QWidget()
        il = QVBoxLayout(ident_host)
        il.setContentsMargins(0, 0, 0, 0)
        ih = QLabel("Leakage = the same text (plate, container ID, serial …) in several splits — usually the same "
                    "object in train and val. Look-alike characters (O/0, I/1, S/5 …) are treated as equal. "
                    "Double-click to list the files.")
        ih.setWordWrap(True)
        apply_style(ih, "color:@muted;")
        il.addWidget(ih)
        il.addWidget(self.ident_tbl, 1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(self.cards, "Cards")
        self.tabs.addTab(self.table, "Table")
        self.tabs.addTab(ident_host, "Identities")
        self.tabs.addTab(self.summary_lbl, "Summary")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_cards())
        self.table.horizontalHeader().sortIndicatorChanged.connect(lambda *_: self._refresh_cards())

    def _thr(self) -> dict:
        return {**DEFAULT_THRESHOLDS, "min_conf": self.min_conf.value(), "trunc_tol": self.trunc.value(),
                "loose": self.loose.value(), "min_h": self.min_h.value(),
                "geometry": (self.mode.currentData() == "locate")}

    # ================================================================== scan
    def make_context(self):
        cfg = self._collect()
        if not cfg["classes"]:
            raise ValueError("Tick at least one text class.")
        if not cfg["python"] or not Path(cfg["python"]).exists():
            raise ValueError("Choose the Python of the environment that has the OCR engine, then press Check.")
        if not cfg["engine"]:
            raise ValueError("Press Check to find the OCR engines in that environment, then pick one.")
        codes, missing = OL.resolve(cfg["engine"], cfg["langs"])
        if missing:
            raise ValueError(f"{self.engine.currentText()} cannot read: {', '.join(missing)}. Untick them or choose "
                             "another engine (EasyOCR / Tesseract cover the most languages).")
        cfg["codes"] = codes
        save_ocr_config(self.session.root, cfg)
        self.cfg = cfg
        self.validators = VL.discover(self.session.root)
        thr = self._thr()
        ver = self.engine_versions.get(cfg["engine"], "?")
        return {"adapter": self.session.adapter, "cfg": cfg, "classes": cfg["classes"], "thr": thr,
                "validators": self.validators, "reads": [], "cache": self.session.cache, "svc": self.app.ocr,
                "cancel": threading.Event(), "loaded": False, "load_error": None,
                "sig": (f"ocr|{cfg['engine']}|{ver}|{'+'.join(codes)}|{cfg['strategy'] if len(codes) > 1 else ''}|"
                        f"{cfg['gpu']}|{cfg['mode']}|{thr['pad']}|{thr['min_h']}"
                        + (f"|{thr['min_conf']}" if len(codes) > 1 and cfg['strategy'] == 'fallback' else "")),
                "ran": 0, "cached": 0}

    def stop_scan(self):
        if self.ctx.get("cancel") is not None:
            self.ctx["cancel"].set()
        super().stop_scan()

    def scan_sample(self, ctx, s):
        if ctx["load_error"]:
            return []
        ann = ctx["adapter"].load(s, light=True)
        img = ann.image_path or s.image_path
        if img is None:
            return []
        classes = ctx["classes"]
        boxes = []
        for i, sh in enumerate(ann.shapes):
            if sh.get("label") not in classes:
                continue
            poly = to_polygon(sh)
            pts = poly.tolist() if poly is not None else sh.get("points") or []
            if len(pts) < 2:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            boxes.append((i, sh, (min(xs), min(ys), max(xs), max(ys))))
        if not boxes:
            return []
        norm = ann.normalized
        bkey = lambda b: ",".join((f"{v:.5f}" if norm else f"{v:.1f}") for v in b)
        cache = ctx["cache"]
        data = (cache.get_preds(ctx["sig"], img) if cache is not None else None) or {"boxes": {}}
        todo = [(i, b) for i, _sh, b in boxes if bkey(b) not in data["boxes"]]
        if todo:
            if not ctx["loaded"]:
                try:
                    c = ctx["cfg"]
                    info = ctx["svc"].ensure_loaded(c["python"], c["engine"], c["codes"], c["gpu"], c["mode"],
                                                    c["strategy"], ctx["thr"]["min_conf"])
                    ctx["loaded"] = True
                    ctx["load_info"] = info
                except Exception as exc:
                    ctx["load_error"] = str(exc)
                    raise
            res = ctx["svc"].read([{"path": str(img), "boxes": [list(b) for _i, b in todo], "norm": norm}],
                                  ctx["thr"]["pad"], ctx["thr"]["min_h"], cancel=ctx["cancel"])[0]
            if "error" in res:
                raise RuntimeError(res["error"])
            data["w"], data["h"] = res["w"], res["h"]
            for (_i, b), r in zip(todo, res["boxes"]):
                data["boxes"][bkey(b)] = r
            if cache is not None:
                cache.put_preds(ctx["sig"], img, data)
            ctx["ran"] += 1
        else:
            ctx["cached"] += 1
        W, H = data.get("w", 0), data.get("h", 0)
        rows = []
        for i, sh, b in boxes:
            res = data["boxes"].get(bkey(b), {"error": "no result"})
            box = (b[0] * W, b[1] * H, b[2] * W, b[3] * H) if norm else b
            cc = classes[sh.get("label")]
            gt = sh.get(cc["text_field"]) if cc.get("text_field") else None
            r = Read(s.key, i, sh.get("label"), s.split, box, res, gt if isinstance(gt, str) else None, (W, H))
            ctx["reads"].append(r)
            f = evaluate(r, cc, ctx["thr"], ctx["validators"])
            if f is not None:
                rows.append(self._row(f))
        return rows

    def rescan_sample(self, ctx, s):
        """After an edit: drop the file's old reads first (a full scan never needs this, so it stays linear)."""
        ctx["reads"] = [r for r in ctx.get("reads", []) if r.key != s.key]
        return self.scan_sample(ctx, s)

    def _row(self, f) -> ResultRow:
        k = f.kinds[0]
        title, role = KINDS[k]
        conf = float((f.read.res or {}).get("conf") or 0.0)
        issue = title + (f" +{len(f.kinds) - 1}" if len(f.kinds) > 1 else "")
        return ResultRow(f.read.key, (issue, f.read.label, f.text, round(conf, 3), f.detail, f.read.idx),
                         shape_index=f.read.idx, payload=f, tone=TONE.get(role))

    def finalize(self, ctx):
        if ctx["cache"] is not None:
            ctx["cache"].flush()
        if ctx["load_error"]:
            raise RuntimeError(f"OCR unavailable — {ctx['load_error']}")
        return []

    def on_scan_complete(self, ctx):
        info = ctx.get("load_info") or self.app.ocr.load_info or {}
        if info and not info.get("test_ok", True):
            self.log_note(f"⚠ Engine self-test read “{info.get('test_text')}” instead of “ABC123” — check the engine / "
                          "language settings.")
        self._refresh_insights()

    def log_note(self, text):
        self.app.log_errors([f"[OCR Check] {text}"], quiet=True)

    # ================================================================== live re-evaluation
    def _reevaluate(self):
        if not self.ctx or not self.ctx.get("reads") or (self.worker is not None and self.worker.isRunning()):
            return
        self.ctx["thr"] = self._thr()
        cfg = self._collect()
        self.ctx["classes"] = cfg["classes"]
        self.validators = VL.discover(self.session.root)
        self.ctx["validators"] = self.validators
        rows = []
        for r in self.ctx["reads"]:
            cc = cfg["classes"].get(r.label)
            if cc is None:
                continue
            f = evaluate(r, cc, self.ctx["thr"], self.validators)
            if f is not None:
                rows.append(self._row(f))
        self.table.setRowCount(0)
        self.rows = []
        self._append_rows(rows)
        self._rebuild_filter()
        self.summary.setText(self.summarize())
        self._refresh_insights()

    # ================================================================== views
    def _refresh_cards(self):
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        cards = []
        for row in self.visible_rows()[:5000]:
            f = row.payload
            r = f.read
            col = self.app.label_colors.get(r.label)
            shapes = [Shape(box_points(*r.box), col.name() if col else hexc("info"))]
            if self.ctx.get("thr", {}).get("geometry"):
                for l in (r.res or {}).get("lines") or []:
                    shapes.append(Shape(box_points(*l[:4]), hexc("accent"), dashed=True, width=1.5))
            s = self.session.by_key.get(r.key)
            chips = [(KINDS[k][0], KINDS[k][1]) for k in f.kinds[:3]]
            conf = (r.res or {}).get("conf")
            if conf is not None:
                chips.append((f"{float(conf):.2f}", "muted"))
            read = f"read “{f.text}”" if f.text else "no text read"
            lang = (r.res or {}).get("lang")
            if lang and len(self.cfg.get("codes") or []) > 1:
                read += f" [{lang}]"
            cards.append(Card(row, [Panel(r.key, r.box, shapes)], chips,
                              [read + (f" · {f.passed}" if f.passed else ""),
                               f"{f.detail} — {s.display if s else r.key}"]))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Cards ({len(cards):,})")

    def _refresh_insights(self):
        reads = self.ctx.get("reads") or []
        if not reads:
            return
        cfg = self._collect()
        summ = summarize(reads, cfg["classes"], self._thr(), self.validators)
        self._summ = summ
        # identities
        rows = [(ident, "leakage", ", ".join(f"{sp} ({len(v)})" for sp, v in by.items()),
                 sorted(set().union(*by.values()))) for ident, by in summ.leaks]
        rows += [(ident, "frequent", "", n) for ident, n in summ.frequent[:200]]
        self.ident_tbl.setRowCount(len(rows))
        self._ident_rows = rows
        for i, (ident, kind, splits, files) in enumerate(rows):
            n = files if isinstance(files, int) else len(files)
            for c, v in enumerate((ident, kind, splits, n)):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, v)
                if c == 1:
                    it.setForeground(theme_color("muted" if kind == "frequent" else "error"))
                self.ident_tbl.setItem(i, c, it)
        self.ident_tbl.resizeColumnsToContents()
        self.tabs.setTabText(2, f"Identities ({len(summ.leaks):,} leaked)")
        # summary
        parts = []
        for lab, c in sorted(summ.per_class.items()):
            b = max(c["boxes"], 1)
            line = (f"<b>{lab}</b> — {c['boxes']:,} boxes · readable {c['readable'] / b * 100:.1f}% · "
                    f"confident {c['confident'] / b * 100:.1f}%")
            if c["validated"]:
                line += f" · rules pass {c['passed'] / max(c['validated'], 1) * 100:.1f}%"
            if c["skipped"]:
                line += f" · skipped {c['skipped']:,} (too small / unreadable image)"
            lens = summ.lengths.get(lab)
            if lens:
                top = ", ".join(f"{L} chars × {n:,}" for L, n in lens.most_common(4))
                line += f"<br><span style='color:{hexc('muted')}'>lengths: {top}</span>"
            parts.append(line)
        lang_mix = Counter((r.res or {}).get("lang") for r in reads if (r.res or {}).get("text"))
        lang_mix.pop(None, None)
        if len(lang_mix) > 1:
            parts.append("Read with: " + ", ".join(f"{l} × {n:,}" for l, n in lang_mix.most_common()))
        rare = [ch for ch, n in summ.chars.most_common()[-8:] if n < 10]
        if rare:
            parts.append(f"<span style='color:{hexc('warn')}'>Rare characters (under 10 reads): {' '.join(rare)}</span> — "
                         "a recogniser trained on this data will struggle with them.")
        ran, cached = self.ctx.get("ran", 0), self.ctx.get("cached", 0)
        parts.append(f"<span style='color:{hexc('muted')}'>OCR ran on {ran:,} image(s); {cached:,} came from the cache.</span>")
        self.summary_lbl.setText("<br><br>".join(parts))

    def summarize(self):
        self._refresh_cards()
        live = [r for r in self.rows if r]
        c = Counter(KINDS[r.payload.kinds[0]][0] for r in live)
        return " · ".join(f"{n:,} {k}" for k, n in c.most_common()) or "✅ Every text box reads and passes its checks"

    def _open_identity(self, r, _c):
        rows = getattr(self, "_ident_rows", [])
        if not 0 <= r < len(rows):
            return
        ident, kind, _s, files = rows[r]
        if isinstance(files, int):
            files = sorted({x.key for x in self.ctx["reads"] if VL.identity_key(
                VL.normalize((x.res or {}).get("text") or "", self._collect()["classes"].get(x.label, DEFAULT_CLASS)
                             .get("normalize"))) == ident})
        self.app.apply_file_filter(files, f"🔤 {kind}: {ident}")

    # ================================================================== actions
    def _fix_in_editor(self):
        keys = list(dict.fromkeys(r.key for r in self.target_rows()))
        if not keys:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first.")
            return
        self.app.open_editor_session(keys)

    def _reads_for(self, label):
        cfg = self._collect()
        cc = cfg["classes"].get(label, DEFAULT_CLASS)
        thr = self._thr()
        return [VL.normalize((r.res or {}).get("text") or "", cc.get("normalize"))
                for r in (self.ctx.get("reads") or []) if r.label == label
                and float((r.res or {}).get("conf") or 0) >= thr["min_conf"]]

    def _open_validators(self):
        cfg = self._collect()
        dlg = ValidatorEditor(self, self.session.root, reads_provider=self._reads_for,
                              class_names=list(cfg["classes"]) or list(self._cls_widgets),
                              normalize_opts=cfg["normalize"])
        dlg.exec()
        self.validators = VL.discover(self.session.root)
        for _lab, (_it, vb, _l, _t) in self._cls_widgets.items():
            self._label_validators(vb)
        self._reevaluate()

    def _export_reads(self):
        reads = self.ctx.get("reads") or []
        if not reads:
            QMessageBox.information(self, self.TITLE, "Run a scan first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export OCR reads", "ocr_reads.csv", "CSV (*.csv)")
        if not path:
            return
        cfg = self._collect()
        thr = self._thr()
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "shape", "class", "split", "read", "normalised", "confidence", "validator passed",
                        "issues", "detail"])
            for r in reads:
                cc = cfg["classes"].get(r.label, DEFAULT_CLASS)
                f = evaluate(r, cc, thr, self.validators)
                s = self.session.by_key.get(r.key)
                res = r.res or {}
                w.writerow([s.display if s else r.key, r.idx, r.label, r.split, res.get("text", ""),
                            VL.normalize(res.get("text") or "", cc.get("normalize")), res.get("conf", ""),
                            f.passed if f else "(all checks pass)", ", ".join(KINDS[k][0] for k in f.kinds) if f else "",
                            f.detail if f else ""])
        self.app.statusBar().showMessage(f"Exported {len(reads):,} reads to {path}", 6000)
