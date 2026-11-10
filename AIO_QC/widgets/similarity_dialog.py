# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🔎 Visual Similarity — find mislabels by how objects LOOK, with a vision model you already have (DINOv3 / DINOv2 /
SigLIP2 / CLIP …), running in your Python environment (Models & Environment), never inside the app.

Models: add a model FOLDER, a model ZIP (extracted once into a models folder you choose, reused afterwards), or
download a preset from Hugging Face (DINOv3 is licence-gated: accept it on huggingface.co and paste a token).
Vectors are cached per model (.aio_qc/embeddings, float16) and reused as long as the image / shape is unchanged,
so the second run over the same data only embeds what changed. See core.embeddings for the analysis.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog, QVBoxLayout, QWidget,
)

from AIO_QC.core.embeddings import (
    KINDS, PRESETS, EmbeddingCache, analyze, extract_zip, file_sig, find_model_root, folder_size, geom_id, model_id,
    model_kind,
)
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import newButton, style as apply_style
from .base_scan import BaseScanDialog, ResultRow, dspin, ispin
from .inference_service import FuncWorker

CHUNK = 384                     # crops per server request (progress granularity)


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024


class SimilarityDialog(BaseScanDialog):
    TITLE, ICON = "Visual Similarity", "🔎"
    HELP = ("Every object (or, for classification datasets, every image) is turned into a vector by a vision model; "
            "objects that look alike end up close together. Reported: likely mislabels (its closest look-alikes "
            "carry one other label), objects atypical for their label, and near-identical objects with different "
            "labels. The first run embeds everything (GPU recommended for large models); later runs reuse the "
            "cache and only embed what changed.")
    COLUMNS = ["Finding", "Label", "Suggests", "Score", "Why"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = False

    # ------------------------------------------------------------------ parameters
    def build_params(self, form):
        st = self.app.settings
        self.models: list[dict] = []
        try:
            self.models = json.loads(st.value("embed/models", "[]") or "[]")
        except ValueError:
            self.models = []
        self.model_box = QComboBox()
        self.model_box.currentIndexChanged.connect(lambda _i: self._model_info())
        mrow = QWidget()
        ml = QHBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.model_box, 1)
        for text, slot, tip in (("Folder…", self._add_folder, "A model folder (already extracted)"),
                                ("Zip…", self._add_zip, "A model zip: extracted once into a models folder you choose"),
                                ("Download…", self._download, "Download a preset model from Hugging Face"),
                                ("Remove", self._remove_model, "Forget this model (files stay where they are)")):
            ml.addWidget(newButton(text, slot=slot, tip=tip))
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        apply_style(self.model_note, "color:@muted;")
        self.env_note = QLabel("")
        self.env_note.setWordWrap(True)
        apply_style(self.env_note, "color:@muted;")
        erow = QWidget()
        el = QHBoxLayout(erow)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.env_note, 1)
        el.addWidget(newButton("Environment…", slot=lambda: self.app.open_model_setup(),
                               tip="The Python environment that runs the model (shared with Model Check)"))
        self.scope = QComboBox()
        self.scope.addItems(["All files", "Files in current list"])
        self.label_box = QComboBox()
        self.label_box.addItem("(all labels)", None)
        for lab, _n in self.session.label_totals().most_common():
            self.label_box.addItem(lab, lab)
        self.cap = ispin(0, 10_000_000, int(st.value("embed/cap", 0)))
        self.cap.setSpecialValueText("all")
        self.cap.setToolTip("At most this many objects per label (random sample) — for a quick first look at a huge "
                            "dataset; 'all' checks everything")
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["auto", "cpu", "cuda:0", "cuda:1", "mps"])
        self.half = QCheckBox("fp16 on GPU")
        self.half.setChecked(True)
        self.batch = ispin(1, 1024, int(st.value("embed/batch", 32)))
        self.pad = ispin(0, 100, 10)
        self.pad.setSuffix(" % context")
        self.pad.setToolTip("Margin around each object in its crop")
        self.k = ispin(3, 50, 10)
        self.k.setToolTip("How many closest look-alikes vote on each object's label")
        self.vote = dspin(0.5, 1.0, 0.6, 2, step=0.05)
        self.vote.setToolTip("Report a likely mislabel when look-alikes with ONE other label hold at least this share "
                             "of the (similarity-weighted) vote")
        self.zlim = dspin(2.0, 10.0, 3.5, 1, step=0.5)
        self.zlim.setToolTip("Atypical: this many robust standard deviations below the label's usual similarity")
        self.dup = dspin(0.8, 1.0, 0.97, 3, step=0.005)
        self.dup.setToolTip("Look-alike pair: similarity at least this (mutual nearest neighbours only)")
        grid = QWidget()
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        for c, (t, w) in enumerate((("max / label", self.cap), ("device", self.device), ("", self.half),
                                    ("batch", self.batch), ("crop", self.pad))):
            g.addWidget(QLabel(t), 0, 2 * c)
            g.addWidget(w, 0, 2 * c + 1)
        for c, (t, w) in enumerate((("look-alikes k", self.k), ("vote ≥", self.vote), ("atypical z ≥", self.zlim),
                                    ("look-alike sim ≥", self.dup))):
            g.addWidget(QLabel(t), 1, 2 * c)
            g.addWidget(w, 1, 2 * c + 1)
        form.addRow("Model:", mrow)
        form.addRow("", self.model_note)
        form.addRow("Runs in:", erow)
        sc = QWidget()
        sl = QHBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.scope)
        sl.addWidget(QLabel("Label:"))
        sl.addWidget(self.label_box, 1)
        form.addRow("Scope:", sc)
        form.addRow("Settings:", grid)
        self.add_action("✓ Use suggested label", self._relabel,
                        "Relabel the ticked (or selected) rows to the suggested label — backed up, undoable")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Put back the files changed by the last relabel")
        self.undo_btn.setEnabled(False)
        self.add_action("🧩 Crop Gallery", self._gallery,
                        "Open the Crop Gallery for the selected row's label, least typical first")
        self._last_backup = None
        self._fill_models()
        self._env_text()

    def _fill_models(self, select: str | None = None):
        self.model_box.blockSignals(True)
        self.model_box.clear()
        for m in self.models:
            self.model_box.addItem(m["name"], m["path"])
        if not self.models:
            self.model_box.addItem("No model yet — add a folder, a zip or download one", None)
        i = self.model_box.findData(select) if select else -1
        last = self.app.settings.value("embed/last", "")
        self.model_box.setCurrentIndex(i if i >= 0 else max(0, self.model_box.findData(last)))
        self.model_box.blockSignals(False)
        self._model_info()

    def _save_models(self):
        self.app.settings.setValue("embed/models", json.dumps(self.models))

    def _model_info(self):
        path = self.model_box.currentData()
        if not path:
            self.model_note.setText("DINOv3 ViT-S/16 is a good start (fast); ViT-L/16 or SigLIP2-L for the best "
                                    "quality. Folder or zip: Hugging Face format (config.json + weights), timm, "
                                    "open_clip, or the DINOv3 repository with its .pth.")
            return
        if not Path(path).is_dir():
            self.model_note.setText(f"⚠ Folder not found: {path}")
            return
        m = next((x for x in self.models if x["path"] == path), {})
        self.model_note.setText(f"{m.get('kind', model_kind(path))} · {_human(m.get('size', 0))} · {path}")

    def _env_text(self):
        py = self.app.inference.python
        self.env_note.setText(f"Python: {py}" if py else "⚠ No Python environment chosen yet (Environment…)")

    def _register(self, root: Path, name: str | None = None):
        root = Path(root)
        entry = {"name": name or root.name, "path": str(root), "kind": model_kind(root), "size": folder_size(root)}
        self.models = [m for m in self.models if m["path"] != entry["path"]] + [entry]
        self._save_models()
        self._fill_models(entry["path"])

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Model folder (config.json + weights, or the DINOv3 repo)",
                                             self.app.settings.value("embed/models_dir", str(Path.home())))
        if not d:
            return
        root = find_model_root(d)
        if root is None:
            QMessageBox.warning(self, self.TITLE, "No model found in that folder (expected config.json, "
                                                  "open_clip_config.json or hubconf.py somewhere inside).")
            return
        self._register(root, Path(d).name)

    def _models_dir(self) -> str | None:
        st = self.app.settings
        cur = st.value("embed/models_dir", str(Path.home() / "AIO_QC_models"))
        d = QFileDialog.getExistingDirectory(self, "Folder to keep extracted / downloaded models in "
                                                   "(needs room for the unpacked model)", cur)
        if d:
            st.setValue("embed/models_dir", d)
        return d or None

    def _run_with_progress(self, title, fn, on_done):
        dlg = QProgressDialog(title, "Cancel", 0, 1000, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        cancel = threading.Event()
        dlg.canceled.connect(cancel.set)
        state = {"p": 0}

        def progress(done, total):
            state["p"] = int(1000 * done / max(total, 1))

        from PyQt6.QtCore import QTimer
        timer = QTimer(dlg, interval=150, timeout=lambda: dlg.setValue(state["p"]) if state["p"] else dlg.setRange(0, 0))
        timer.start()
        w = FuncWorker(lambda: fn(progress, cancel), self)

        def finish(res=None, err=None):
            timer.stop()
            dlg.close()
            if err:
                if "cancelled" not in err.lower():
                    QMessageBox.warning(self, self.TITLE, err)
                return
            on_done(res)
        w.done.connect(lambda r: finish(r))
        w.failed.connect(lambda e: finish(err=e))
        w.finished.connect(w.deleteLater)
        w.start()
        self._bg = w

    def _add_zip(self):
        z, _ = QFileDialog.getOpenFileName(self, "Model zip", self.app.settings.value("embed/zip_dir", str(Path.home())),
                                           "Zip (*.zip)")
        if not z:
            return
        self.app.settings.setValue("embed/zip_dir", str(Path(z).parent))
        dest = self._models_dir()
        if not dest:
            return
        free = shutil.disk_usage(dest).free
        need = sum(i.file_size for i in __import__("zipfile").ZipFile(z).infolist())
        if need > free * 0.95:
            QMessageBox.warning(self, self.TITLE, f"Not enough space in {dest}: the model needs {_human(need)}, "
                                                  f"{_human(free)} free.")
            return
        self._run_with_progress(f"Extracting {Path(z).name} ({_human(need)}) — once; reused afterwards …",
                                lambda prog, cancel: extract_zip(z, dest, prog, cancel.is_set),
                                lambda root: self._register(root, Path(z).stem))

    def _download(self):
        py = self.app.inference.python
        if not py:
            QMessageBox.information(self, self.TITLE, "Choose a Python environment first (Environment…).")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Download a model")
        lay = QVBoxLayout(dlg)
        f = QFormLayout()
        preset = QComboBox()
        for title, repo, size, note in PRESETS:
            preset.addItem(f"{title}  ({size})", (repo, note))
        preset.addItem("Other Hugging Face repository…", ("", ""))
        repo = QLineEdit(PRESETS[0][1])
        note = QLabel(PRESETS[0][3])
        apply_style(note, "color:@muted;")
        preset.currentIndexChanged.connect(lambda _i: (repo.setText(preset.currentData()[0]),
                                                       note.setText(preset.currentData()[1])))
        token = QLineEdit()
        token.setEchoMode(QLineEdit.EchoMode.Password)
        token.setPlaceholderText("hf_… (only for gated models such as DINOv3; not stored)")
        f.addRow("Model:", preset)
        f.addRow("Repository:", repo)
        f.addRow("", note)
        f.addRow("Token:", token)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Download")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if not dlg.exec() or not repo.text().strip():
            return
        base = self._models_dir()
        if not base:
            return
        r = repo.text().strip()
        dest = str(Path(base) / r.replace("/", "__"))
        svc, tok = self.app.embedder, token.text().strip()
        if not svc.probe(py).get("huggingface_hub"):
            self._offer_install(["huggingface_hub"], "Downloading needs huggingface_hub in the environment.")
            return
        self._run_with_progress(f"Downloading {r} … (large models take a while)",
                                lambda _p, cancel: svc.download(py, r, dest, tok, cancel),
                                lambda path: self._register(find_model_root(path) or Path(path), r.split("/")[-1]))

    def _remove_model(self):
        path = self.model_box.currentData()
        if path:
            self.models = [m for m in self.models if m["path"] != path]
            self._save_models()
            self._fill_models()

    def _offer_install(self, pkgs: list[str], why: str):
        py = self.app.inference.python
        cmd = f'"{py}" -m pip install ' + " ".join(pkgs)
        box = QMessageBox(self)
        box.setWindowTitle(self.TITLE)
        box.setText(f"{why}\n\nMissing in the environment: {', '.join(pkgs)}\n\n{cmd}")
        inst = box.addButton("Install now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is not inst:
            return
        from .process_runner import StepRunner
        log = QDialog(self)
        log.setWindowTitle("Installing …")
        log.resize(720, 360)
        v = QVBoxLayout(log)
        out = QPlainTextEdit()
        out.setReadOnly(True)
        v.addWidget(out)
        runner = StepRunner([(f"pip install {' '.join(pkgs)}", [py, "-m", "pip", "install", *pkgs])], log)
        runner.output.connect(lambda t: (out.moveCursor(out.textCursor().MoveOperation.End), out.insertPlainText(t)))
        runner.finished.connect(lambda ok, msg: (log.setWindowTitle("Installed ✓" if ok else "Install failed"),
                                                 self.app.embedder.shutdown()))
        runner.start()
        log.exec()

    # ------------------------------------------------------------------ scan
    def samples_to_scan(self):
        if self.scope.currentIndex() == 1:
            keys = set(self.app.file_dock.visible_keys())
            return [s for s in self.session.samples if s.key in keys]
        return list(self.session.samples)

    def make_context(self):
        path = self.model_box.currentData()
        if not path or not Path(path).is_dir():
            raise ValueError("Add a model first (Folder…, Zip… or Download…).")
        py = self.app.inference.python
        if not py:
            self.app.open_model_setup()
            raise ValueError("Choose the Python environment that should run the model (Environment…).")
        st = self.app.settings
        st.setValue("embed/last", path)
        st.setValue("embed/cap", self.cap.value())
        st.setValue("embed/batch", self.batch.value())
        unit = "image" if self.session.is_classification else "shape"
        m = next((x for x in self.models if x["path"] == path), {"name": Path(path).name, "kind": model_kind(path)})
        return {"adapter": self.session.adapter, "unit": unit, "label": self.label_box.currentData(),
                "keys": [], "idx": [], "labels": [], "boxes": [], "wh": [], "images": [], "ids": [],
                "python": py, "path": path, "model": m, "device": self.device.currentText().strip() or "auto",
                "half": self.half.isChecked(), "batch": self.batch.value(), "pad": self.pad.value() / 100,
                "k": self.k.value(), "vote": self.vote.value(), "z": self.zlim.value(), "dup": self.dup.value(),
                "cap": self.cap.value(), "cancel": threading.Event(), "root": str(self.session.root),
                "classes": {s.key: self.session.class_of(s.key) for s in self.session.samples}
                if unit == "image" else None}

    def stop_scan(self):
        if self.ctx and self.ctx.get("cancel") is not None:
            self.ctx["cancel"].set()
        super().stop_scan()

    def scan_sample(self, ctx, s):
        img = s.image_path
        root = ctx["root"]
        rel = os.path.relpath(s.key, root).replace("\\", "/")
        if ctx["unit"] == "image":
            lab = ctx["classes"].get(s.key)
            if not lab or img is None or (ctx["label"] and lab != ctx["label"]):
                return []
            ctx["keys"].append(s.key)
            ctx["idx"].append(-1)
            ctx["labels"].append(lab)
            ctx["boxes"].append(None)
            ctx["wh"].append(None)
            ctx["images"].append(str(img))
            ctx["ids"].append(f"{rel}|image|{file_sig(img)}")
            return []
        ann = ctx["adapter"].load(s, light=True)
        img = ann.image_path or img
        if img is None or not ann.shapes:
            return []
        if ann.normalized:
            size = ctx["adapter"].image_size(s, ann)
            if not size:
                return []
            W, H = float(size[0]), float(size[1])
        else:
            W, H = float(ann.width or 0), float(ann.height or 0)
            if not (W and H):
                size = ctx["adapter"].image_size(s, ann)
                W, H = (float(size[0]), float(size[1])) if size else (0.0, 0.0)
        sx, sy = (W, H) if ann.normalized else (1.0, 1.0)
        isig = file_sig(img)
        for i, sh in enumerate(ann.shapes):
            lab = str(sh.get("label"))
            pts = sh.get("points") or []
            if not pts or sh.get("_error") or (ctx["label"] and lab != ctx["label"]):
                continue
            try:
                xs, ys = [float(p[0]) * sx for p in pts], [float(p[1]) * sy for p in pts]
            except (TypeError, ValueError, IndexError):
                continue
            if (sh.get("shape_type") or "polygon") == "circle" and len(pts) == 2:
                r = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
                box = [xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r]
            else:
                box = [min(xs), min(ys), max(xs), max(ys)]
            if box[2] - box[0] < 2 and box[3] - box[1] < 2:            # points: some context around them
                c = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                d = max(W, H) * 0.02
                box = [c[0] - d, c[1] - d, c[0] + d, c[1] + d]
            ctx["keys"].append(s.key)
            ctx["idx"].append(i)
            ctx["labels"].append(lab)
            ctx["boxes"].append(box)
            ctx["wh"].append((W, H))
            ctx["images"].append(str(img))
            ctx["ids"].append(f"{rel}|{geom_id(sh)}|{isig}")
        return []

    def _progress(self, phase, done, total):
        self.ctx["phase"] = phase
        w = self.worker
        if w is not None:
            w.progress.emit(int(done), int(max(total, 1)))

    def _on_progress(self, i, total):
        super()._on_progress(i, total)
        ph = (self.ctx or {}).get("phase")
        if ph:
            self.status.setText(f"{ph} {i:,} / {total:,}")

    def finalize(self, ctx):
        n_all = len(ctx["ids"])
        if n_all < 3:
            raise ValueError("fewer than 3 objects to compare")
        keep = list(range(n_all))
        if ctx["cap"]:                                                 # random sample per label (seeded)
            by = {}
            for i, l in enumerate(ctx["labels"]):
                by.setdefault(l, []).append(i)
            rnd = random.Random(7)
            keep = sorted(i for idx in by.values() for i in (rnd.sample(idx, ctx["cap"]) if len(idx) > ctx["cap"]
                                                             else idx))
        for f in ("keys", "idx", "labels", "boxes", "wh", "images", "ids"):
            ctx[f] = [ctx[f][i] for i in keep]
        n = len(ctx["ids"])
        svc, py = self.app.embedder, ctx["python"]
        self._progress("Starting the model …", 0, 1)
        info = svc.probe(py)
        miss = svc.missing(ctx["model"].get("kind", "transformers"))
        if not info.get("torch"):
            miss = sorted(set(miss) | {"torch"})
        if miss:
            ctx["env_missing"] = miss
            raise ValueError("the environment lacks: " + ", ".join(miss))
        li = svc.ensure_loaded(py, ctx["path"], ctx["device"], ctx["half"])
        ctx["load_info"] = li
        mid = model_id(ctx["path"]) + f"_p{int(round(ctx['pad'] * 100))}"
        cache = EmbeddingCache(ctx["root"], mid, ctx["unit"])
        rows, missing = cache.get(ctx["ids"])
        dim = int(li["dim"])
        vecs = np.zeros((n, dim), np.float16)
        have = rows >= 0
        if have.any() and cache.vecs is not None and cache.vecs.shape[1] == dim:
            vecs[have] = cache.vecs[rows[have]]
        else:
            missing = list(range(n))
        valid = np.ones(n, bool)
        tmp = Path(tempfile.mkdtemp(prefix="aioqc_embed_"))
        try:
            # group the missing crops by image: each image is decoded once
            order = sorted(missing, key=lambda i: ctx["images"][i])
            done = 0
            for a in range(0, len(order), CHUNK):
                if ctx["cancel"].is_set():
                    raise ValueError("stopped")
                part = order[a:a + CHUNK]
                items, flat = [], []
                for i in part:
                    if items and items[-1]["image"] == ctx["images"][i]:
                        items[-1]["boxes"].append(ctx["boxes"][i])
                    else:
                        items.append({"image": ctx["images"][i], "boxes": [ctx["boxes"][i]]})
                    flat.append(i)
                out = tmp / f"c{a}.npy"
                res = svc.embed(items, out, ctx["batch"], ctx["pad"], cancel=ctx["cancel"])
                arr = np.load(out)
                vecs[flat] = arr[:len(flat)]
                for start, cnt, _err in res.get("failed", []):
                    for q in range(start, start + cnt):
                        valid[flat[q]] = False
                done += len(part)
                self._progress(f"Embedding with {ctx['model'].get('name')}:", done, len(order))
            # keep the cache to what exists: old ids of files still in the dataset + everything computed now
            alive = {os.path.relpath(k, ctx["root"]).replace("\\", "/") for k in self.session.by_key}
            now = set(ctx["ids"])
            keep_old = [(k, r) for k, r in cache.index.items() if k.split("|", 1)[0] in alive and k not in now] \
                if cache.vecs is not None and cache.vecs.shape[1] == dim else []
            ids = ctx["ids"] + [k for k, _r in keep_old]
            allv = np.concatenate([vecs, cache.vecs[[r for _k, r in keep_old]]]) if keep_old else vecs
            ok_ids = [i for i, v in zip(ids, list(valid) + [True] * len(keep_old)) if v]
            cache.save(ok_ids, allv[np.array(list(valid) + [True] * len(keep_old), bool)])
            self._progress("Finding look-alikes …", 0, 1)
            np.save(tmp / "all.npy", vecs)
            svc.knn(tmp / "all.npy", min(ctx["k"], n - 1), str(tmp / "nn"), cancel=ctx["cancel"])
            nn_idx, nn_sim = np.load(tmp / "nn_idx.npy"), np.load(tmp / "nn_sim.npy")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ctx["vecs"], ctx["valid"], ctx["n_new"] = vecs, valid, len(missing)
        found = analyze(vecs, ctx["labels"], nn_idx, nn_sim, ctx["vote"], 0.0, ctx["z"], ctx["dup"], valid=valid)
        tones = {"mislabel": "error", "lookalike": "warning", "atypical": "muted"}
        out = []
        for f in sorted(found, key=lambda f: -f.score):
            i = f.i
            out.append(ResultRow(ctx["keys"][i], (KINDS[f.kind], f.label, f.suggest, round(f.score, 3), f.evidence),
                                 None if ctx["idx"][i] < 0 else ctx["idx"][i], payload=(f, i), tone=tones[f.kind]))
        return out

    def on_scan_complete(self, ctx):
        miss = ctx.get("env_missing")
        if miss:
            self._offer_install(miss, "This model needs packages your environment does not have yet.")
            return
        if ctx.get("vecs") is not None:                                # look-alike orderings for the Crop Gallery
            self.app.embed_index = {"model": ctx["model"].get("name"),
                                    "map": {(k, i): r for r, (k, i) in enumerate(zip(ctx["keys"], ctx["idx"]))},
                                    "vecs": ctx["vecs"]}
        li = ctx.get("load_info") or {}
        logger.info(f"[similarity] {ctx['model'].get('name')} on {li.get('device')} · {len(ctx['ids']):,} objects · "
                    f"{ctx.get('n_new', 0):,} newly embedded")

    def summarize(self):
        c = self.ctx or {}
        if c.get("vecs") is None:
            return super().summarize()
        n, new = len(c["ids"]), c.get("n_new", 0)
        li = c.get("load_info") or {}
        return (f"{n:,} objects compared with {c['model'].get('name')} ({li.get('device', '?')}, {li.get('dim')}-d) · "
                f"{new:,} newly embedded, {n - new:,} from the cache · {len(self.rows):,} finding(s)")

    # ------------------------------------------------------------------ actions
    def _relabel(self):
        rows = [r for r in self.target_rows() if r.payload and r.payload[0].suggest]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows with a suggested label first.")
            return
        if not self.confirm(f"Relabel {len(rows):,} object(s) to the suggested label?\n\nThe originals are backed up "
                            "(Undo)."):
            return
        c = self.ctx
        if c["unit"] == "image":
            done, errs = 0, []
            for r in rows:
                ok, err = self.session.set_class([r.key], r.payload[0].suggest)
                done += len(ok)
                errs += err
            ids = {id(r) for r in rows}
            self.remove_rows(lambda x: id(x) in ids)
            self.report("Done", [f"Moved / relabelled {done:,} image(s)."], errs)
            return
        fixes = {}
        for r in rows:
            f, i = r.payload
            W, H = c["wh"][i]
            b = c["boxes"][i]
            fixes.setdefault(r.key, []).append(("relabel", c["idx"][i], (b[0] / W, b[1] / H, b[2] / W, b[3] / H),
                                                f.suggest))
        st = self.session.apply_fixes(self, fixes, what="similarity")
        if st["files"]:
            self._last_backup = st["backup"]
            self.undo_btn.setEnabled(True)
            self.session.set_review(st["changed"], REVIEW_FLAG, "Visual Similarity: relabelled — check it")
        ids = {id(r) for r in rows}
        self.remove_rows(lambda x: id(x) in ids)
        lines = [f"Relabelled {st['done']:,} object(s) in {st['files']:,} file(s); the files are flagged for a look."]
        if st["missing"]:
            lines.append(f"{st['missing']:,} skipped: the shape changed since the run.")
        if st["refused"]:
            lines.append(f"{st['refused']:,} skipped: the label is not a class of this dataset.")
        self.report("Done", lines, st["errors"])

    def _undo(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last relabel?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.report("Undone", [f"Restored {len(keys):,} file(s)."] +
                    ([f"{len(skipped):,} changed since and left alone."] if skipped else []), [])

    def _gallery(self):
        rows = self.target_rows()
        label = rows[0].payload[0].label if rows and rows[0].payload else self.label_box.currentData()
        self.app.open_tool("crop_gallery")
        dlg = self.app.open_dialogs.get("crop_gallery")
        if dlg is not None and label:
            dlg.show_label(label, sort="Least typical first")
