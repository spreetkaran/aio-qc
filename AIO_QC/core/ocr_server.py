# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC OCR server — runs INSIDE the user's chosen Python environment (same protocol as infer_server.py).

    {"rid":1,"cmd":"probe"}                                  -> python, installed engines + versions, GPU
    {"rid":2,"cmd":"load","engine":"paddleocr","langs":["en","hi"],"gpu":false,"mode":"read"|"locate",
     "strategy":"best"|"fallback","fallback_conf":0.6}
                                                             -> self-test result (renders "ABC123", reads it back)
    {"rid":3,"cmd":"read","items":[{"path":..., "boxes":[[x1,y1,x2,y2],...], "norm":false}], "pad":0.15,
     "min_h":8}                                              -> per image, per box: {"text","conf","lines":[...]}
    {"rid":4,"cmd":"shutdown"}

Each image is decoded ONCE and all of its boxes are cropped here (no pixels cross the process boundary).
"read" mode: recognition only (fast). "locate" mode: detection + recognition inside the (padded) crop; "lines" then
holds every text line found with its box in IMAGE coordinates (for truncated / loose / merged-line checks).
Standard library + the OCR engine; numpy and Pillow come with every engine.
"""
import json
import sys
import time
import traceback

RESP_PREFIX = "@@AIOQC@@ "
_OUT = sys.stdout
sys.stdout = sys.stderr

ENGINE = None
ENGINES = ["paddleocr", "rapidocr", "easyocr", "tesseract"]
TITLES = {"paddleocr": "PaddleOCR", "rapidocr": "RapidOCR", "easyocr": "EasyOCR", "tesseract": "Tesseract"}


def reply(rid, **payload):
    payload["rid"] = rid
    _OUT.write(RESP_PREFIX + json.dumps(payload) + "\n")
    _OUT.flush()


def log(msg):
    sys.stderr.write(f"[ocr] {msg}\n")


def _version(mod):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return None


def _gpu():
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "cuda (paddle)"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda (torch)"
    except Exception:
        pass
    return None


def probe():
    found = {}
    for key, dist in (("paddleocr", "paddleocr"), ("rapidocr", "rapidocr"), ("rapidocr", "rapidocr_onnxruntime"),
                      ("easyocr", "easyocr"), ("tesseract", "pytesseract")):
        v = _version(dist)
        if v and key not in found:
            found[key] = {"title": TITLES[key], "version": v, "package": dist}
    if "tesseract" in found:
        try:
            import pytesseract
            found["tesseract"]["binary"] = str(pytesseract.get_tesseract_version())
        except Exception as exc:
            found["tesseract"]["error"] = f"tesseract program not found ({exc})"
    return {"python": sys.version.split()[0], "executable": sys.executable, "engines": found, "gpu": _gpu()}


# =============================================================================
# Images
# =============================================================================
def load_image(path):
    """RGB numpy array, EXIF orientation applied (annotation tools show images that way)."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


# =============================================================================
# Engines — each returns, per crop, (text, conf, lines[(x1,y1,x2,y2,text,conf) in crop pixels])
# =============================================================================
def _poly_box(pts):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _join(lines):
    """Lines top-to-bottom, left-to-right -> one string (space-joined), mean confidence."""
    if not lines:
        return "", 0.0
    lines = sorted(lines, key=lambda l: (round(l[1] / max(1.0, (l[3] - l[1]) * 0.6)), l[0]))
    text = " ".join(l[4] for l in lines if l[4])
    conf = sum(l[5] for l in lines) / len(lines)
    return text, float(conf)


class PaddleEngine:
    """PaddleOCR 2.x (ocr(det=, rec=)) and 3.x (predict / TextRecognition)."""

    def __init__(self, lang, gpu, mode):
        import paddleocr
        self.major = int(str(_version("paddleocr") or "2").split(".")[0])
        self.mode = mode
        if self.major >= 3:
            dev = "gpu" if gpu else "cpu"
            self.pipe = paddleocr.PaddleOCR(lang=lang, device=dev, use_doc_orientation_classify=False,
                                            use_doc_unwarping=False, use_textline_orientation=False)
            self.rec = None
            # the stand-alone recogniser is the Chinese / English model: other languages use their full pipeline
            if mode == "read" and lang in ("ch", "en"):
                try:
                    self.rec = paddleocr.TextRecognition(device=dev)
                except Exception as exc:          # older 3.x builds: fall back to the full pipeline
                    log(f"TextRecognition unavailable ({exc}); using the full pipeline")
        else:
            self.pipe = paddleocr.PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=gpu, show_log=False)

    @staticmethod
    def _get(res, key, default=None):
        for acc in (lambda r: r[key], lambda r: getattr(r, key), lambda r: r.json["res"][key]):
            try:
                return acc(res)
            except Exception:
                continue
        return default

    def run(self, crops):
        out = []
        for crop in crops:
            if self.major >= 3:
                if self.rec is not None:
                    res = list(self.rec.predict(crop))[0]
                    t, c = self._get(res, "rec_text", ""), float(self._get(res, "rec_score", 0.0) or 0.0)
                    h, w = crop.shape[:2]
                    out.append((t, c, [(0, 0, w, h, t, c)] if t else []))
                    continue
                res = list(self.pipe.predict(crop))[0]
                texts = self._get(res, "rec_texts", []) or []
                scores = self._get(res, "rec_scores", []) or []
                polys = self._get(res, "rec_polys", None)
                if polys is None:
                    polys = self._get(res, "dt_polys", []) or []
                lines = [(*_poly_box(p), str(t), float(s)) for p, t, s in zip(polys, texts, scores)]
            else:
                if self.mode == "read":
                    r = self.pipe.ocr(crop, det=False, cls=False)
                    item = r[0][0] if r and r[0] else ("", 0.0)
                    t, c = str(item[0]), float(item[1])
                    h, w = crop.shape[:2]
                    out.append((t, c, [(0, 0, w, h, t, c)] if t else []))
                    continue
                r = self.pipe.ocr(crop, det=True, rec=True, cls=False)
                lines = [(*_poly_box(b), str(tc[0]), float(tc[1])) for b, tc in (r[0] or [])] if r else []
            t, c = _join(lines)
            out.append((t, c, lines))
        return out


class RapidEngine:
    """rapidocr (2.x/3.x, RapidOCROutput) and rapidocr_onnxruntime (1.x, (result, elapse))."""

    def __init__(self, lang, gpu, mode):
        self.mode = mode
        if lang in (None, "", "ch", "en"):
            try:
                from rapidocr import RapidOCR
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
            self.eng = RapidOCR()
            return
        try:                                            # rapidocr 3.x: other recognition models via LangRec
            from rapidocr import LangRec, RapidOCR
            self.eng = RapidOCR(params={"Rec.lang_type": LangRec(lang)})
        except Exception as exc:
            raise RuntimeError(f"this RapidOCR version cannot load the '{lang}' model ({exc}). Use PaddleOCR or "
                               "EasyOCR for this language.")

    def _call(self, crop, det):
        res = self.eng(crop, use_det=det, use_cls=False, use_rec=True)
        if isinstance(res, tuple):                              # 1.x: (result, elapse)
            res = res[0] or []
            if det:
                return [(*_poly_box(r[0]), str(r[1]), float(r[2])) for r in res]
            return [(None, None, None, None, str(r[0]), float(r[1])) for r in res]
        txts = list(getattr(res, "txts", None) or [])           # 2.x / 3.x
        scores = list(getattr(res, "scores", None) or [])
        boxes = getattr(res, "boxes", None)
        if det and boxes is not None:
            return [(*_poly_box(b), str(t), float(s)) for b, t, s in zip(boxes, txts, scores)]
        return [(None, None, None, None, str(t), float(s)) for t, s in zip(txts, scores)]

    def run(self, crops):
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            lines = self._call(crop, det=self.mode == "locate")
            lines = [(0, 0, w, h, l[4], l[5]) if l[0] is None else l for l in lines]
            t, c = _join(lines) if self.mode == "locate" else ((lines[0][4], lines[0][5]) if lines else ("", 0.0))
            out.append((t, c, lines if t else []))
        return out


class EasyEngine:
    def __init__(self, lang, gpu, mode):
        import easyocr
        langs = lang if isinstance(lang, list) else [l.strip() for l in str(lang).split(",") if l.strip()]
        self.reader = easyocr.Reader(langs or ["en"], gpu=gpu, verbose=False)
        self.mode = mode

    def run(self, crops):
        out = []
        for crop in crops:
            if self.mode == "locate":
                res = self.reader.readtext(crop, detail=1, paragraph=False)
            else:
                res = self.reader.recognize(crop, detail=1, paragraph=False)
            lines = [(*_poly_box(b), str(t), float(c)) for b, t, c in res]
            t, c = _join(lines)
            out.append((t, c, lines))
        return out


class TesseractEngine:
    def __init__(self, lang, gpu, mode):
        import pytesseract
        self.t = pytesseract
        langs = lang if isinstance(lang, list) else [lang]
        self.lang = "+".join(dict.fromkeys(p for l in langs for p in {"en": "eng"}.get(l, l).split("+")))
        self.mode = mode

    def run(self, crops):
        out = []
        for crop in crops:
            psm = 6 if self.mode == "locate" else 7            # 7 = single text line
            d = self.t.image_to_data(crop, lang=self.lang, config=f"--psm {psm}", output_type=self.t.Output.DICT)
            words = {}
            for i, txt in enumerate(d["text"]):
                conf = float(d["conf"][i])
                if txt.strip() and conf >= 0:
                    k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                    x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                    words.setdefault(k, []).append((x, y, x + w, y + h, txt, conf / 100.0))
            lines = []
            for ws in words.values():
                ws.sort(key=lambda z: z[0])
                lines.append((min(z[0] for z in ws), min(z[1] for z in ws), max(z[2] for z in ws), max(z[3] for z in ws),
                              " ".join(z[4] for z in ws), sum(z[5] for z in ws) / len(ws)))
            t, c = _join(lines)
            out.append((t, c, lines))
        return out


CLASSES = {"paddleocr": PaddleEngine, "rapidocr": RapidEngine, "easyocr": EasyEngine, "tesseract": TesseractEngine}
NATIVE_MULTI = {"easyocr", "tesseract"}


class MultiLang:
    """
    Engines with one model per language (PaddleOCR, RapidOCR): several models, per crop the most confident read.
      best      every model reads every crop
      fallback  the first language reads everything; the others only re-read crops below `fallback_conf` (faster)
    run() returns (text, conf, lines, lang) per crop.
    """

    def __init__(self, cls, langs, gpu, mode, strategy, fallback_conf):
        self.langs = list(langs)
        self.models = [cls(l, gpu, mode) for l in self.langs]
        self.strategy, self.fallback_conf = strategy, fallback_conf

    def run(self, crops):
        best = [(t, c, ls, self.langs[0]) for t, c, ls in self.models[0].run(crops)]
        for lang, model in zip(self.langs[1:], self.models[1:]):
            idx = [i for i, b in enumerate(best)
                   if self.strategy == "best" or not b[0] or b[1] < self.fallback_conf]
            if not idx:
                continue
            for i, (t, c, ls) in zip(idx, model.run([crops[i] for i in idx])):
                if t and (not best[i][0] or c > best[i][1]):
                    best[i] = (t, c, ls, lang)
        return best


class SingleLang:
    """Adapter: engines that read all languages in one pass (or one language) -> (text, conf, lines, lang)."""

    def __init__(self, model, tag):
        self.model, self.tag = model, tag

    def run(self, crops):
        return [(t, c, ls, self.tag) for t, c, ls in self.model.run(crops)]


def _test_image():
    """White strip with 'ABC123' in black — every engine should read it."""
    import numpy as np
    try:
        import cv2
        img = np.full((80, 320, 3), 255, np.uint8)
        cv2.putText(img, "ABC123", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA)
        return img
    except ImportError:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("RGB", (320, 80), "white")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        except OSError:
            font = ImageFont.load_default()
        ImageDraw.Draw(im).text((16, 12), "ABC123", fill="black", font=font)
        return np.asarray(im)


def load(engine, langs, gpu, mode, strategy="best", fallback_conf=0.6):
    global ENGINE
    if engine not in CLASSES:
        raise ValueError(f"unknown engine {engine!r}")
    langs = [l for l in (langs if isinstance(langs, list) else [langs]) if l] or ["en"]
    t0 = time.time()
    cls = CLASSES[engine]
    if engine in NATIVE_MULTI:
        obj = SingleLang(cls(langs, bool(gpu), mode), "+".join(langs))
    elif len(langs) == 1:
        obj = SingleLang(cls(langs[0], bool(gpu), mode), langs[0])
    else:
        obj = MultiLang(cls, langs, bool(gpu), mode, strategy, float(fallback_conf))
    ENGINE = {"name": engine, "mode": mode, "obj": obj, "langs": langs}
    t1 = time.time()
    text, conf, _, _lang = obj.run([_test_image()])[0]
    ok = "".join(ch for ch in text.upper() if ch.isalnum()) == "ABC123"
    latin = any(l.split("+")[0] in ("en", "eng", "ch", "latin", "fr", "german", "de", "es", "it", "pt") for l in langs)
    log(f"{engine} ({', '.join(langs)}) loaded in {t1 - t0:.1f}s · self-test read {text!r} ({conf:.2f}) "
        f"{'✓' if ok else '✗'}")
    return {"load_s": round(t1 - t0, 1), "test_text": text, "test_conf": round(conf, 3), "test_ok": ok or not latin,
            "test_note": "" if latin else "self-test uses Latin text, which a non-Latin model may not read",
            "version": _version({"tesseract": "pytesseract", "rapidocr": "rapidocr"}.get(engine, engine))
            or _version("rapidocr_onnxruntime")}


def read(items, pad, min_h):
    if ENGINE is None:
        raise RuntimeError("no OCR engine loaded")
    eng = ENGINE["obj"]
    out = []
    for it in items:
        try:
            img = load_image(it["path"])
        except Exception as exc:
            out.append({"error": f"cannot read image: {exc}"})
            continue
        H, W = img.shape[:2]
        crops, meta, res = [], [], [None] * len(it["boxes"])
        for i, b in enumerate(it["boxes"]):
            x1, y1, x2, y2 = b
            if it.get("norm"):
                x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
            bh, bw = y2 - y1, x2 - x1
            if bh < min_h or bw < 2:
                res[i] = {"skip": "too small", "h": round(bh, 1)}
                continue
            p = pad * bh
            cx1, cy1 = int(max(0, x1 - p)), int(max(0, y1 - p))
            cx2, cy2 = int(min(W, x2 + p)), int(min(H, y2 + p))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                res[i] = {"skip": "outside the image"}
                continue
            crops.append(img[cy1:cy2, cx1:cx2])
            meta.append((i, cx1, cy1))
        if crops:
            try:
                read_out = eng.run(crops)
            except Exception as exc:
                log(traceback.format_exc())
                read_out = [("", 0.0, [], "")] * len(crops)
                for i, _cx, _cy in meta:
                    res[i] = {"error": f"{type(exc).__name__}: {exc}"}
            for (i, cx, cy), (text, conf, lines, lang) in zip(meta, read_out):
                if res[i] is not None:
                    continue
                res[i] = {"text": text, "conf": round(float(conf), 4), "lang": lang,
                          "lines": [[round(l[0] + cx, 1), round(l[1] + cy, 1), round(l[2] + cx, 1), round(l[3] + cy, 1),
                                     l[4], round(float(l[5]), 4)] for l in lines]}
        out.append({"w": W, "h": H, "boxes": res})
    return out


def main():
    log(f"ready · python {sys.version.split()[0]} · {sys.executable}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid, cmd = req.get("rid"), req.get("cmd")
            if cmd == "probe":
                reply(rid, ok=True, info=probe())
            elif cmd == "load":
                reply(rid, ok=True, **load(req["engine"], req.get("langs") or [req.get("lang", "en")],
                                           req.get("gpu", False), req.get("mode", "read"),
                                           req.get("strategy", "best"), req.get("fallback_conf", 0.6)))
            elif cmd == "read":
                reply(rid, ok=True, results=read(req["items"], float(req.get("pad", 0.15)), float(req.get("min_h", 6))))
            elif cmd == "shutdown":
                reply(rid, ok=True)
                break
            else:
                reply(rid, ok=False, error=f"unknown command {cmd!r}")
        except Exception as exc:
            log(traceback.format_exc())
            reply(rid, ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
