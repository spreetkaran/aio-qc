# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Assist (Image QC ▸ Edit): the dataset's models suggest labels on the shown image — click one to accept it (it
becomes a real shape, undoable like any edit), right-click to drop it, Accept All takes every one above the
confidence. Suggestions that repeat a label already there (same label, IoU ≥ 0.5) are not shown.

    Suggest Labels (Ctrl+Alt+J)       this image, now
    Suggest on Every Image            … and on every image shown (predictions are cached: instant on a revisit,
                                      shared with Model Check)
    Accept All Suggestions (Ctrl+Alt+Enter)
    Outline Boxes with SAM (Ctrl+Alt+X)   a drawn box — or an accepted box suggestion — becomes the object's outline
                                      (SAM / SAM2 through ultralytics in the models' environment; weights download
                                      on first use)

Models, their classes and the environment come from Models & Environment; nothing is written until a suggestion is
accepted. Runs in the background — the window never waits for the model.
"""
from __future__ import annotations

import threading

from loguru import logger
from PyQt6.QtCore import QObject, pyqtSignal

from AIO_QC.core.diag import ignored as _ignored

MATCH_IOU = 0.5


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def new_suggestions(preds, shapes_px, conf: float) -> list[dict]:
    """preds: [(label, conf, (x1, y1, x2, y2), poly | None, model id)] in image pixels; shapes_px: [(label, box)] of the
    labels already there -> suggestions at or above `conf` that do not repeat one of them, most confident first."""
    out = []
    for label, c, box, poly, mid in sorted(preds, key=lambda p: -p[1]):
        if c < conf:
            continue
        if any(lab == label and _iou(box, b) >= MATCH_IOU for lab, b in shapes_px):
            continue
        if any(s["label"] == label and _iou(box, s["box"]) >= MATCH_IOU for s in out):
            continue                                        # two models, one object
        out.append({"label": label, "conf": float(c), "box": tuple(map(float, box)), "poly": poly, "model": mid})
    return out


class ModelAssist(QObject):
    ready = pyqtSignal(str, list, str)            # image key, suggestions, error ("" = fine)
    outlined = pyqtSignal(object, object, str)    # job, polygon | None, error

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.items: list[dict] = []               # the shown suggestions (canvas pixels)
        self.key = None
        self._busy = False
        self._want = None
        self.last_error = ""
        c = app.canvas
        c.suggestionClicked.connect(self.accept)
        c.suggestionDismissed.connect(self.dismiss)
        self.ready.connect(self._ready)
        self.outlined.connect(self._outlined)

    # ------------------------------------------------------------------ settings
    @property
    def settings(self):
        return self.app.settings

    @property
    def conf(self) -> float:
        return float(self.settings.value("assist/conf", 0.35))

    @property
    def auto(self) -> bool:
        return self.settings.value("assist/auto", False, type=bool)

    @property
    def sam_on(self) -> bool:
        return self.settings.value("assist/sam", False, type=bool)

    @property
    def sam_model(self) -> str:
        return self.settings.value("assist/sam_model", "sam2.1_b.pt") or "sam2.1_b.pt"

    def _models(self):
        from AIO_QC.core.state import load_models_config
        s = self.app.session
        if s is None:
            return []
        ms = [m for m in load_models_config(s.root).get("models", []) if m.get("enabled", True)]
        cls = s.is_classification                    # classification: classifiers; else: detectors
        return [m for m in ms if (m.get("task") == "classify") == cls]

    # ------------------------------------------------------------------ suggest
    def image_shown(self):
        """The window showed another image: suggestions go with the old one; with “every image” on, ask again."""
        if self.app.current_key != self.key:
            self.clear()
            self.key = None
        if self.auto and self.app.current_key and self._models():
            self.suggest(quiet=True)

    def suggest(self, quiet: bool = False):
        app, s = self.app, self.app.session
        key = app.current_key
        if s is None or key is None:
            return
        models = self._models()
        svc = app.inference
        if not models or not svc.python:
            if not quiet:
                from .errors import show_error
                show_error(app, "Suggest Labels", ValueError(
                    "Add a model in ⚙ Models & environment." if not models else
                    "Choose a Python environment in ⚙ Models & environment."), app=app)
            return
        if self._busy:
            self._want = key                         # the latest image wins once the running one is done
            return
        self._busy = True
        self.app.statusBar().showMessage("Model Assist: asking the model …", 3000)
        threading.Thread(target=self._work, args=(key, models), daemon=True, name="model-assist").start()

    def _work(self, key, models):
        """Worker thread: cached predictions, else one server call — then mapped to the dataset's labels."""
        from .model_setup_dialog import server_specs
        from AIO_QC.core.model_check import model_signature
        s, svc = self.app.session, self.app.inference
        if s.is_classification:
            return self._work_cls(key, models)
        try:
            sample = s.by_key[key]
            ann = s.adapter.load(sample, light=True)
            img = ann.image_path or sample.image_path
            if img is None:
                raise ValueError("this file has no image")
            specs = server_specs(models, svc.opts, svc.info)
            if not specs:
                raise ValueError("None of the enabled model files exist.")
            maps = {m["id"]: dict(m.get("class_map") or {}) for m in models}
            cache = s.cache
            got, geo, wh, need = {}, {}, None, []
            for sp in specs:
                hit = cache.get_preds(model_signature(sp), img) if cache is not None else None
                if hit is None:
                    need.append(sp)
                else:
                    got[sp["id"]], wh = hit["boxes"], (hit["w"], hit["h"])
                    if hit.get("geo"):
                        geo[sp["id"]] = hit["geo"]
            if need:
                svc.ensure_loaded(specs)
                ids = [sp["id"] for sp in need]
                classes = {mid: sorted(int(k) for k, v in maps.get(mid, {}).items() if v) for mid in ids}
                res = svc.predict([str(img)], ids, conf=0.05, classes=classes).get(str(img)) or {}
                if res.get("error") and not res.get("w"):
                    raise RuntimeError(res["error"])
                wh = (res.get("w"), res.get("h"))
                for sp in need:
                    boxes = (res.get("by_model") or {}).get(sp["id"]) or []
                    got[sp["id"]] = boxes
                    g = (res.get("geo") or {}).get(sp["id"])
                    if g:
                        geo[sp["id"]] = g
                    if cache is not None and res.get("w"):
                        entry = {"w": res["w"], "h": res["h"], "boxes": boxes}
                        if g:
                            entry["geo"] = g
                        cache.put_preds(model_signature(sp), img, entry)
                if cache is not None:
                    cache.flush()
            preds = []
            for mid, boxes in got.items():
                cmap, gl = maps.get(mid, {}), geo.get(mid) or []
                for n, b in enumerate(boxes or ()):
                    label = cmap.get(str(int(b[5])), "")
                    if label:
                        g = gl[n] if n < len(gl) else None
                        preds.append((label, float(b[4]), tuple(map(float, b[:4])), (g or {}).get("poly"), mid))
            from AIO_QC.core.model_check import gt_boxes
            W, H = (ann.width or (wh or (0, 0))[0]), (ann.height or (wh or (0, 0))[1])
            shapes = [(lab, box) for _i, lab, box in gt_boxes(ann.shapes, W or 1, H or 1, ann.normalized)]
            sugg = new_suggestions(preds, shapes, self.conf)
            for x in sugg:
                x["img_w"] = float((wh or (W, H))[0] or W or 1)
            self.ready.emit(key, sugg, "")
        except Exception as exc:
            _ignored(exc)
            self.ready.emit(key, [], f"{type(exc).__name__}: {exc}" if not isinstance(exc, ValueError) else str(exc))

    def _work_cls(self, key, models):
        """Classification: the classifier's top classes for the image (cached like Model Check's)."""
        from .model_setup_dialog import server_specs
        from AIO_QC.core.cls_model_check import map_probs
        from AIO_QC.core.model_check import model_signature
        s, svc = self.app.session, self.app.inference
        try:
            sample = s.by_key[key]
            ann = s.adapter.load(sample, light=True)
            img = ann.image_path or sample.image_path
            if img is None:
                raise ValueError("this file has no image")
            m = models[0]
            specs = server_specs([m], svc.opts, svc.info)
            if not specs:
                raise ValueError(f"{m['name']}: the model file is missing.")
            sig = model_signature(specs[0]) + "|cls"
            hit = s.cache.get_preds(sig, img) if s.cache is not None else None
            if hit is None:
                svc.ensure_loaded(specs)
                res = svc.predict([str(img)], [m["id"]], conf=0.0).get(str(img)) or {}
                probs = (res.get("probs") or {}).get(m["id"])
                if probs is None:
                    raise RuntimeError(res.get("error") or "the model gave no classes")
                if s.cache is not None:
                    s.cache.put_preds(sig, img, {"w": res.get("w"), "h": res.get("h"), "boxes": [], "probs": probs})
                    s.cache.flush()
            else:
                probs = hit.get("probs") or []
            top = map_probs(probs, dict(m.get("class_map") or {}), dict(m.get("names") or {}))[:3]
            self.ready.emit(key, [{"cls": c, "conf": float(v), "label": c, "model": m.get("id", "")} for c, v in top], "")
        except Exception as exc:
            _ignored(exc)
            self.ready.emit(key, [], f"{type(exc).__name__}: {exc}" if not isinstance(exc, ValueError) else str(exc))

    def _ready(self, key, sugg, err):
        self._busy = False
        self.last_error = err
        if err:
            from .errors import message
            self.app.statusBar().showMessage(f"Model Assist: {message(err)}", 8000)
            logger.warning(f"[model assist] {err}")
        elif key == self.app.current_key:
            self._show(key, sugg)
        want, self._want = self._want, None
        if want is not None and want == self.app.current_key and want != key:
            self.suggest(quiet=True)

    def _show(self, key, sugg):
        if sugg and "cls" in sugg[0]:
            self._show_cls(key, sugg)
            return
        pix_w = self.app.canvas.pixmap_item.pixmap().width()
        self.key = key
        self.items = []
        for x in sugg:
            k = pix_w / x["img_w"] if x.get("img_w") and pix_w else 1.0
            self.items.append(dict(x, box=tuple(v * k for v in x["box"]),
                                   poly=None if x.get("poly") is None else [(px * k, py * k) for px, py in x["poly"]]))
        self._draw()
        n = len(self.items)
        self.app.statusBar().showMessage(
            f"Model Assist: {n} suggestion(s) — click one to accept, right-click to drop, Ctrl+Alt+Enter accepts all"
            if n else "Model Assist: nothing new to suggest on this image", 8000)

    # ------------------------------------------------------------------ classification
    def _show_cls(self, key, sugg):
        """A bar over the image: the model's top classes — click one to give the image that class."""
        from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
        self.key = key
        self.items = [dict(x, box=(0, 0, 0, 0), poly=None) for x in sugg]
        bar = getattr(self, "_cls_bar", None)
        if bar is None:
            bar = self._cls_bar = QWidget(self.app.canvas)
            bar.setObjectName("clsSuggest")
            bar.setStyleSheet("#clsSuggest{background: rgba(20,20,24,215); border-radius: 10px;}"
                              "QLabel{color: white;} QPushButton{padding: 3px 10px;}")
            QHBoxLayout(bar).setContentsMargins(10, 6, 10, 6)
        lay = bar.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        lay.addWidget(QLabel("Model:"))
        have = set(self.app.session.tags_of(key))
        for i, x in enumerate(self.items):
            b = QPushButton(f"{'✓ ' if x['cls'] in have else ''}{x['cls']}  {x['conf']:.0%}")
            b.setToolTip("Give the image this class" + (" (it has it)" if x["cls"] in have else "") +
                         (" — Ctrl+Alt+Enter" if i == 0 else ""))
            b.clicked.connect(lambda _c=False, i=i: self.accept(i))
            lay.addWidget(b)
        close = QPushButton("✕")
        close.clicked.connect(self.clear)
        lay.addWidget(close)
        bar.adjustSize()
        c = self.app.canvas
        bar.move(max(8, c.width() - bar.width() - 14), max(8, c.height() - bar.height() - 14))
        bar.show()
        bar.raise_()
        self._update_actions()
        top = self.items[0] if self.items else None
        if top:
            agree = top["cls"] in have
            self.app.statusBar().showMessage(f"Model Assist: {top['cls']} {top['conf']:.0%}" +
                                             (" — agrees with the label" if agree else " — click it to use it"), 8000)

    def _accept_cls(self, x) -> bool:
        from AIO_QC.core import provenance
        s = self.app.session
        try:
            from AIO_QC.core.model_registry import origin_for
            o = origin_for(x.get("model") or "", s.root, "Model Assist")
        except Exception as exc:                          # noqa: BLE001 — provenance never blocks an edit
            _ignored(exc)
            o = provenance.make_origin("model", "Model Assist")
        with provenance.origin(o):
            if s.cls_mode == "multi":
                done, errors = s.set_tags([self.key], add=[x["cls"]], what=f"+{x['cls']} (model)")
            else:
                done, errors = s.set_tags([self.key], tags=[x["cls"]], what=f"class → {x['cls']} (model)")
        if errors:
            self.app.log_errors([f"[model assist] {e}" for e in errors])
        if done:
            if self.app.current_key not in s.by_key:
                self.app.file_dock.select_key(done[0])
            else:
                self.app.reload_current_annotation()
        return bool(done)

    def _draw(self):
        colors = self.app.label_colors
        from AIO_QC import utils
        items = []
        for x in self.items:
            if x["label"] not in colors:
                utils.update_label_colors(x["label"], colors)
            col = colors.get(x["label"])
            items.append({"box": x["box"], "poly": x["poly"], "text": f"{x['label']} {x['conf']:.2f}",
                          "color": col.name() if hasattr(col, "name") else str(col or "#1e88e5")})
        self.app.canvas.set_suggestions(items)
        self._update_actions()

    def _update_actions(self):
        a = getattr(self.app, "a", None)
        if a is not None and hasattr(a, "assist_accept_all"):
            a.assist_accept_all.setEnabled(bool(self.items))
            a.assist_clear.setEnabled(bool(self.items))

    # ------------------------------------------------------------------ accept / drop
    def _commit(self, x, quiet=False) -> bool:
        from AIO_QC.core import provenance
        from AIO_QC.core.inline_edit import new_box, new_polygon
        from AIO_QC.core.model_registry import origin_for
        ed = self.app.inline
        try:
            o = origin_for(x.get("model") or "", self.app.session.root, "Model Assist")
        except Exception as exc:                          # noqa: BLE001 — provenance never blocks an edit
            _ignored(exc)
            o = provenance.make_origin("model", "Model Assist")
        rec = provenance.shape_record(o, score=x.get("conf"))
        W, H = ed._wh()
        if not self.app.session.adapter.can_use_label(x["label"]):
            ed._say(f"✋ '{x['label']}' is not a class of this dataset", warn=True)
            return False
        if x.get("poly") and len(x["poly"]) >= 3:
            pts = [[float(a), float(b)] for a, b in x["poly"]]

            def fn(ann):
                try:
                    new_polygon(ann, pts, x["label"], W, H)[provenance.KEY] = dict(rec)
                    return f"accepted {x['label']} outline ({x['conf']:.2f})"
                except Exception as exc:            # a detection file: the box instead
                    _ignored(exc)
                    new_box(ann, x["box"], x["label"], W, H)[provenance.KEY] = dict(rec)
                    return f"accepted {x['label']} box ({x['conf']:.2f})"
        else:
            def fn(ann):
                new_box(ann, x["box"], x["label"], W, H)[provenance.KEY] = dict(rec)
                return f"accepted {x['label']} box ({x['conf']:.2f})"
        return ed.commit(fn, quiet=quiet)

    def accept(self, i: int):
        if not (0 <= i < len(self.items)):
            return
        x = self.items[i]
        if "cls" in x:
            if self._accept_cls(x):
                self.clear()
            return
        if self.sam_on and not x.get("poly"):
            self.outline(x["box"], x["label"], then=lambda poly, x=x: self._accept_with(x, poly))
            return
        self._accept_with(x, None)

    def _accept_with(self, x, poly):
        if self._commit(dict(x, poly=poly) if poly else x):
            self.items = [y for y in self.items if y is not x]
            self._draw()

    def accept_all(self) -> int:
        """Every suggestion at or above the confidence — one refresh, each its own undo step."""
        s = self.app.session
        if self.items and "cls" in self.items[0]:          # classification: the top class
            ok = self._accept_cls(self.items[0])
            self.clear()
            return int(ok)
        todo = [x for x in self.items if x["conf"] >= self.conf]
        if not todo:
            return 0
        done = 0
        s.blockSignals(True)
        try:
            for x in todo:
                if self._commit(x, quiet=True):
                    done += 1
        finally:
            s.blockSignals(False)
        key = self.app.current_key
        if done:
            s.annotationsChanged.emit([key])
            s.indexChanged.emit()
        self.items = [x for x in self.items if x not in todo]
        self._draw()
        self.app.inline._update_actions()
        self.app.inline._say(f"✓ Accepted {done} suggestion(s) — Ctrl+Z takes them back one by one" if done else
                             f"✋ Nothing accepted — {self.app.inline._refused or 'not possible'}", warn=not done)
        return done

    def dismiss(self, i: int):
        if 0 <= i < len(self.items):
            del self.items[i]
            self._draw()

    def clear(self):
        self.items = []
        self.app.canvas.clear_suggestions()
        bar = getattr(self, "_cls_bar", None)
        if bar is not None:
            bar.hide()
        self._update_actions()

    # ------------------------------------------------------------------ SAM
    def outline(self, box_px, label, then):
        """SAM outline of a box (canvas pixels) in the background; then(polygon | None) on the UI thread."""
        app, s, svc = self.app, self.app.session, self.app.inference
        if not svc.python:
            from .errors import show_error
            show_error(app, "Outline with SAM", ValueError("Choose a Python environment in ⚙ Models & environment."),
                       app=app)
            then(None)
            return
        key = app.current_key
        sample = s.by_key.get(key)
        pix = app.canvas.pixmap_item.pixmap()
        job = {"then": then, "key": key}
        app.statusBar().showMessage("Outline with SAM … (the first time, ultralytics downloads the weights)", 6000)

        def work():
            try:
                ann = s.adapter.load(sample, light=True)
                img = ann.image_path or sample.image_path
                from AIO_QC.core.io import image_size
                W, _H = image_size(img) or (pix.width(), pix.height())
                k = W / pix.width() if pix.width() else 1.0
                polys = svc.segment(self.sam_model, str(img), [[v * k for v in box_px]])
                poly = polys[0] if polys else None
                self.outlined.emit(job, None if not poly else [(x / k, y / k) for x, y in poly], "")
            except Exception as exc:
                _ignored(exc)
                self.outlined.emit(job, None, f"{type(exc).__name__}: {exc}")
        threading.Thread(target=work, daemon=True, name="model-assist-sam").start()

    def _outlined(self, job, poly, err):
        if job["key"] == self.app.current_key:
            job["then"](poly)
        if err:                                       # said last: the shape's own message does not hide it
            from .errors import message
            self.app.statusBar().showMessage(f"SAM: {message(err)} — kept as a box", 10000)
            logger.warning(f"[model assist] SAM: {err}")

    def box_drawn(self, rect, label) -> bool:
        """Inline editor hook: with “Outline Boxes with SAM” on, a drawn box becomes an outline. -> True = handled."""
        if not self.sam_on:
            return False
        box = (rect.left(), rect.top(), rect.right(), rect.bottom())

        def then(poly):
            x = {"label": label, "conf": 1.0, "box": box, "poly": poly}
            self._commit(x)
        self.outline(box, label, then)
        return True

