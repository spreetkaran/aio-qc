# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AI annotate: one image in, checked shapes out — and a whole folder of them, concurrently, cached, within a budget
(Qt-free).

    Annotator(cfg, env_python, cache_db, audit_path).image(path) -> Result
    Annotator.close()

Every answer is cached by (what decides the answer, the model, the view's examples, the image's content, the
region), so re-runs, "Try again" and stop/continue never pay twice. Costs come from the service's token counts
and the project's prices.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2

from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
from AIO_QC.tools.frame_extractor.core.ai_annotate import parse as PA
from AIO_QC.tools.frame_extractor.core.ai_annotate import profiles as P
from AIO_QC.tools.frame_extractor.core.ai_annotate import prompt as PR
from AIO_QC.tools.frame_extractor.core.ai_annotate import providers as PV
from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import read_labels
from AIO_QC.core.diag import ignored as _ignored


class AIError(Exception):
    pass


@dataclass
class Result:
    path: str
    width: int = 0
    height: int = 0
    shapes: list = field(default_factory=list)       # {label, box, conf, text, text_ok, polygon}
    flags: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    accepted: bool = True
    reasons: list = field(default_factory=list)
    review: bool = False
    answers: list = field(default_factory=list)      # raw answers (audit / preview)
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    cost: float = 0.0
    cached: int = 0
    calls: int = 0
    seconds: float = 0.0
    error: str = ""
    candidates: list = field(default_factory=list)   # detector proposals (detect_vlm preview)

    def counts(self):
        out = {}
        for s in self.shapes:
            out[s["label"]] = out.get(s["label"], 0) + 1
        return out


# ====================================================================================================== env worker
class LocalWorker:
    """The stand-alone worker in the user's Python environment (one process, one request at a time)."""

    def __init__(self, python: str):
        if not python or not os.path.exists(python):
            raise AIError("This mode runs a model in your Python environment — choose one in Settings ▸ Models "
                          "(Image QC ▸ Models & Environment).")
        script = os.path.join(os.path.dirname(__file__), "local_worker.py")
        flags = 0x08000000 if sys.platform == "win32" else 0          # no console window
        self.p = subprocess.Popen([python, "-u", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                                  creationflags=flags)
        self._lock = threading.Lock()
        self._n = 0
        self._err = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for ln in self.p.stderr:
            self._err.append(ln.rstrip())
            del self._err[:-60]

    def call(self, op, timeout=900, **kw):
        with self._lock:
            if self.p.poll() is not None:
                raise AIError("the Python environment's worker stopped:\n" + "\n".join(self._err[-12:]))
            self._n += 1
            self.p.stdin.write(json.dumps({"id": self._n, "op": op, **kw}) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            if not line:
                raise AIError("the Python environment's worker stopped:\n" + "\n".join(self._err[-12:]))
            res = json.loads(line)
            if not res.get("ok"):
                raise AIError(res.get("error") or "the worker failed")
            return res

    def vlm(self, cfg, system, parts):
        enc = []
        for p in parts:
            if p[0] == "text":
                enc.append({"text": p[1]})
            elif p[0] == "image":
                enc.append({"image": base64.b64encode(p[1]).decode("ascii")})
        r = self.call("vlm", system=system, parts=enc,
                      max_tokens=int((cfg["vlm"].get("params") or {}).get("max_tokens", 2048)))
        return r["text"], {"input": 0, "output": 0}

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.p.stdin.flush()
                self.p.wait(5)
        except Exception as _exc:
            _ignored(_exc)
            try:
                self.p.kill()
            except Exception as _exc:
                _ignored(_exc)


def probe_env(python: str, timeout=60) -> dict:
    """What the environment has (torch, CUDA, transformers …) — runs the worker's probe."""
    w = LocalWorker(python)
    try:
        return w.call("probe", timeout=timeout)
    finally:
        w.close()


# ====================================================================================================== cache
class Cache:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self.db = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
            from AIO_QC.core.sqlite_util import configure
            configure(self.db, path)
            self.db.execute("CREATE TABLE IF NOT EXISTS answers (k TEXT PRIMARY KEY, text TEXT, input INTEGER, "
                            "output INTEGER, at REAL)")
            self.db.commit()

    def get(self, k):
        if not self.db:
            return None
        with self._lock:
            r = self.db.execute("SELECT text, input, output FROM answers WHERE k=?", (k,)).fetchone()
        return r

    def put(self, k, text, usage):
        if not self.db:
            return
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?)",
                            (k, text, usage.get("input", 0), usage.get("output", 0), time.time()))
            self.db.commit()

    def close(self):
        if self.db:
            with self._lock:
                self.db.close()
            self.db = None


def file_hash(path, _memo={}):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    h = _memo.get(key)
    if h is None:
        d = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                d.update(chunk)
        h = _memo[key] = d.hexdigest()
    return h


# ====================================================================================================== annotator
class Annotator:
    def __init__(self, cfg: dict, env_python: str = "", cache_path: str = "", audit_path: str = "",
                 cancelled=lambda: False, base_dir: str = ""):
        self.cfg = C.normalize(cfg)
        self.base = base_dir or os.path.dirname(cfg.get("_path", "") or "") or "."
        self.cancelled = cancelled
        self.refs = PR.References(self.cfg, self.base)
        self.cache = Cache(cache_path)
        self.audit_path = audit_path if self.cfg["output"].get("audit", True) else ""
        self._audit_lock = threading.Lock()
        self.fmt = P.box_format(self.cfg)
        self.prof = P.profile(self.cfg["vlm"].get("model", ""))
        self.key = C.run_key(self.cfg) + "|" + self.cfg["vlm"].get("model", "") + "|" + self.fmt
        self.names = C.class_names(self.cfg)
        self.worker = None
        self.env_python = env_python
        mode = self.cfg["mode"]
        needs_worker = mode in ("detect", "detect_vlm") or self.cfg["connection"].get("provider") == "local" or \
            self.cfg["segment"].get("type", "none") != "none"
        if needs_worker:
            self.worker = LocalWorker(env_python)
            try:
                from AIO_QC.tools.frame_extractor.core.ai_annotate import trust
                if mode in ("detect", "detect_vlm"):
                    d = self.cfg["detector"]
                    ok = trust.trusted(d["model"])
                    if trust.needs_own_code(d["type"], d["model"]) and not ok:
                        raise ValueError(trust.refusal(d["model"]))
                    self.worker.call("load_detector", type=d["type"], model=d["model"], device=d.get("device", "auto"),
                                     trust_remote_code=ok)
                if self.cfg["connection"].get("provider") == "local" and mode != "detect":
                    vm = self.cfg["vlm"]["model"]
                    ok = trust.trusted(vm)
                    if trust.needs_own_code("", vm) and not ok:
                        raise ValueError(trust.refusal(vm))
                    self.worker.call("load_vlm", model=vm, device=self.cfg["detector"].get("device", "auto"),
                                     trust_remote_code=ok)
                if self.cfg["segment"].get("type", "none") != "none":
                    self.worker.call("load_sam", model=self.cfg["segment"]["model"])
            except Exception:
                self.close()
                raise

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self.cache.close()

    # -------------------------------------------------------------- calls
    def _ask(self, system, view, intro, img_bytes, extra_parts=(), cache_key=""):
        parts = list(self.refs.parts(view)) + [("cache", None), ("text", intro), ("image", img_bytes, "image/jpeg")]
        parts += list(extra_parts)
        k = hashlib.sha1((self.key + "|" + view + "|" + cache_key + "|" + system).encode()).hexdigest()
        hit = self.cache.get(k)
        if hit is not None:
            return hit[0], {"input": hit[1], "output": hit[2]}, True
        r = PV.chat(self.cfg, system, parts, self.cancelled, self.worker)
        self.cache.put(k, r["text"], r["usage"])
        return r["text"], r["usage"], False

    def cost_of(self, usage):
        pr = self.cfg.get("pricing") or {}
        return (usage.get("input", 0) * float(pr.get("input") or 0) +
                usage.get("output", 0) * float(pr.get("output") or 0)) / 1e6

    # -------------------------------------------------------------- one image
    def image(self, path: str) -> Result:
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        t0 = time.monotonic()
        res = Result(path)
        img = imread(path)
        if img is None:
            res.error, res.accepted = "the image could not be read", False
            return res
        H, W = img.shape[:2]
        res.width, res.height = W, H
        cfg = self.cfg
        mode = cfg["mode"]
        view = PR.view_for(cfg, path)
        note = (cfg["references"]["views"].get(view) or {}).get("note", "")
        try:
            content = file_hash(path)
            regions = [(None, (0, 0, W, H))]
            roi = cfg["roi"].get("class")
            if roi and mode != "classify":
                pad = float(cfg["roi"].get("padding") or 0)
                regions = []
                for lab, (x1, y1, x2, y2) in read_labels(path):
                    if lab != roi:
                        continue
                    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
                    regions.append((roi, (max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))))
                regions = regions[: max(1, int(cfg["roi"].get("max_crops_per_image") or 4))]
            shapes = []
            for crop_label, (rx1, ry1, rx2, ry2) in regions:
                if self.cancelled():
                    raise PV.ProviderError("stopped")
                sub = img[int(ry1):int(ry2), int(rx1):int(rx2)] if crop_label else img
                if sub.size == 0:
                    continue
                ck = f"{content}|{int(rx1)},{int(ry1)},{int(rx2)},{int(ry2)}"
                if mode == "detect":
                    shapes += self._detect(path, (rx1, ry1, rx2, ry2) if crop_label else None, res)
                    continue
                if mode == "detect_vlm":
                    shapes += self._detect_vlm(path, sub, (rx1, ry1), (rx1, ry1, rx2, ry2) if crop_label else None,
                                               view, note, crop_label, ck, res)
                    continue
                side = int(self.prof.get("max_side") or cfg["vlm"].get("image_max_side") or 1536)
                side = min(side, int(cfg["vlm"].get("image_max_side") or side))
                data, sw, sh, sx, sy = PR.encode(sub, side, int(self.prof.get("round_to") or 0))
                system = PR.system_text(cfg, sw, sh)
                text, usage, cached = self._ask(system, view, PR.image_intro(cfg, sw, sh, crop_label, note), data,
                                                cache_key=ck)
                self._tally(res, text, usage, cached)
                ans = PA.extract_json(text)
                if not isinstance(ans, dict):
                    ans = {"detections": ans} if isinstance(ans, list) else {}
                self._fields(res, ans)
                if mode == "vlm_detect":
                    shapes += self._read_detections(ans, sw, sh, sx, sy, rx1 if crop_label else 0,
                                                    ry1 if crop_label else 0)
            res.shapes = self._finish(shapes, path, W, H, res)
            self._segment(path, res)
            self._check(res)
        except ValueError as exc:
            res.error, res.accepted = f"the answer could not be read: {exc}", False
        except PV.ProviderError as exc:
            res.error, res.accepted = str(exc), False
        except AIError as exc:
            res.error, res.accepted = str(exc), False
        res.seconds = round(time.monotonic() - t0, 2)
        self._audit(res, view)
        return res

    def _tally(self, res, text, usage, cached):
        res.answers.append(text)
        res.calls += 0 if cached else 1
        res.cached += 1 if cached else 0
        if not cached:
            res.usage = {k: res.usage.get(k, 0) + usage.get(k, 0) for k in ("input", "output")}
            res.cost += self.cost_of(usage)

    def _fields(self, res, ans):
        fl = ans.get("flags") or {}
        if isinstance(fl, dict):
            for f in self.cfg["flags"]:
                v = fl.get(f["name"])
                if v is not None:
                    b = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
                    res.flags[f["name"]] = res.flags.get(f["name"], False) or b
        for k in (self.cfg["checks"].get("fields") or {}):
            if k in ans and k not in res.fields:
                res.fields[k] = ans[k]

    def _read_detections(self, ans, sw, sh, sx, sy, ox, oy):
        out = []
        dets = ans.get("detections") or ans.get("objects") or ans.get("boxes") or []
        part_of = {}
        for c in self.cfg["classes"]:
            for p in c.get("parts") or []:
                part_of[p] = c["name"]
        for d in dets if isinstance(dets, list) else []:
            if not isinstance(d, dict):
                continue
            box = d.get("box_2d", d.get("bbox", d.get("box", d.get("bbox_2d"))))
            label = str(d.get("label") or d.get("class") or "")
            if label not in self.names or box is None:
                continue
            try:
                b = PA.to_pixels(box, self.fmt, sw, sh, sx, sy, ox, oy)
            except (ValueError, TypeError):
                continue
            conf = d.get("confidence", d.get("score", 1.0))
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 1.0
            conf = conf / 100 if conf > 1 else conf
            out.append({"label": label, "box": b, "conf": conf, "text": d.get("text")})
            for pname, pbox in (d.get("parts") or {}).items() if isinstance(d.get("parts"), dict) else []:
                if pname in self.names and pbox is not None:
                    try:
                        out.append({"label": pname, "box": PA.to_pixels(pbox, self.fmt, sw, sh, sx, sy, ox, oy),
                                    "conf": conf, "text": None, "_parent": len(out) - 1})
                    except (ValueError, TypeError):
                        pass
        return out

    def _prompts(self):
        d = self.cfg["detector"]
        out = {}
        for c in self.cfg["classes"]:
            p = (d.get("prompts") or {}).get(c["name"])
            out[c["name"]] = p or (c.get("description") or "").split(".")[0].strip() or c["name"].replace("_", " ")
        return out

    def _detect(self, path, crop, res):
        d = self.cfg["detector"]
        r = self.worker.call("detect", path=path, crop=list(crop) if crop else None, prompts=self._prompts(),
                             box_threshold=float(d.get("box_threshold", 0.3)),
                             text_threshold=float(d.get("text_threshold", 0.25)))
        return [{"label": b[5], "box": tuple(b[:4]), "conf": float(b[4]), "text": None} for b in r["boxes"]
                if b[5] in self.names]

    def _detect_vlm(self, path, sub, off, crop, view, note, crop_label, ck, res):
        cands = PA.nms(self._detect(path, crop, res), float(self.cfg["detector"].get("iou", 0.5)))
        res.candidates += cands
        if not cands:
            return []
        shown = sub.copy()
        lw = max(2, int(round(min(shown.shape[:2]) / 300)))
        for i, c in enumerate(cands, 1):
            x1, y1, x2, y2 = (int(c["box"][0] - off[0]), int(c["box"][1] - off[1]), int(c["box"][2] - off[0]),
                              int(c["box"][3] - off[1]))
            cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), lw)
            cv2.putText(shown, str(i), (x1 + 2, max(14, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, lw / 3),
                        (0, 0, 255), max(1, lw // 2 + 1), cv2.LINE_AA)
        side = int(self.cfg["vlm"].get("image_max_side") or 1536)
        data, sw, sh, sx, sy = PR.encode(shown, side)
        system = PR.system_text(self.cfg, sw, sh)
        listed = "\n".join(f"{i}: proposed {c['label']} (detector score {c['conf']:.2f})" for i, c in
                           enumerate(cands, 1))
        text, usage, cached = self._ask(system, view, PR.image_intro(self.cfg, sw, sh, crop_label, note), data,
                                        [("text", "Candidate boxes:\n" + listed)],
                                        cache_key=ck + "|" + json.dumps([c["box"] for c in cands], default=str))
        self._tally(res, text, usage, cached)
        ans = PA.extract_json(text)
        ans = ans if isinstance(ans, dict) else {"boxes": ans}
        self._fields(res, ans)
        out = []
        for v in ans.get("boxes") or []:
            try:
                i = int(v.get("id")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            keep = v.get("keep")
            keep = keep if isinstance(keep, bool) else str(keep).lower() in ("true", "yes", "1")
            if not keep or not 0 <= i < len(cands):
                continue
            label = v.get("label") or cands[i]["label"]
            if label not in self.names:
                continue
            try:
                conf = float(v.get("confidence", cands[i]["conf"]))
            except (TypeError, ValueError):
                conf = cands[i]["conf"]
            out.append({"label": label, "box": cands[i]["box"], "conf": conf, "text": v.get("text")})
        return out

    # -------------------------------------------------------------- post-processing
    def _finish(self, shapes, path, W, H, res):
        by = {c["name"]: c for c in self.cfg["classes"]}
        out = []
        for s in shapes:
            c = by.get(s["label"])
            if c is None:
                continue
            x1, y1, x2, y2 = s["box"]
            x1, x2 = max(0.0, min(W, x1)), max(0.0, min(W, x2))
            y1, y2 = max(0.0, min(H, y1)), max(0.0, min(H, y2))
            ms = max(1, int(c.get("min_size_px") or 0))
            if x2 - x1 < ms or y2 - y1 < ms:
                continue
            if s["conf"] < float(c.get("min_confidence") or 0):
                continue
            s = dict(s, box=(x1, y1, x2, y2))
            if c.get("text"):
                s["text_ok"] = PA.text_ok(s.get("text"), c.get("text_pattern", ""), c.get("text_check", "none"))
                if not s["text_ok"] and self.cfg["checks"].get("on_text_fail", "review") == "drop":
                    continue
            elif s.get("text") is not None and not c.get("text"):
                s["text"] = None
            out.append(s)
        out = PA.nms(out, float(self.cfg["detector"].get("iou", 0.5)))
        # inside: parents from this answer and from the file already there
        parents = {}
        for s in out:
            parents.setdefault(s["label"], []).append(s["box"])
        for lab, b in read_labels(path):
            parents.setdefault(lab, []).append(b)
        kept = []
        for s in out:
            par = by[s["label"]].get("inside")
            if par and not any(PA.inside(s["box"], pb) for pb in parents.get(par, [])):
                continue
            kept.append(s)
        final, per = [], {}
        for s in sorted(kept, key=lambda s: -s["conf"]):
            mx = int(by[s["label"]].get("max_per_image") or 0)
            if mx and per.get(s["label"], 0) >= mx:
                continue
            per[s["label"]] = per.get(s["label"], 0) + 1
            final.append(s)
        thr = float(self.cfg["output"].get("needs_review_below") or 0)
        res.review = any(s["conf"] < thr for s in final) or any(s.get("text_ok") is False for s in final)
        return final

    def _segment(self, path, res):
        if self.cfg["segment"].get("type", "none") == "none" or not res.shapes or self.worker is None:
            return
        r = self.worker.call("segment", path=path, boxes=[list(s["box"]) for s in res.shapes])
        for s, poly in zip(res.shapes, r.get("polygons") or []):
            if poly:
                s["polygon"] = [tuple(p) for p in poly]

    def _check(self, res):
        ch = self.cfg["checks"]
        counts = res.counts()
        if ch.get("require_boxes") and not res.shapes and self.cfg["mode"] != "classify":
            res.accepted = False
            res.reasons.append("no box")
        for rule in ch.get("accept") or []:
            ok, why = PA.rule_ok(rule, res.fields, counts)
            if not ok:
                res.accepted = False
                res.reasons.append(why)

    def _audit(self, res, view):
        if not self.audit_path:
            return
        rec = {"image": res.path, "view": view, "mode": self.cfg["mode"], "model": self.cfg["vlm"].get("model"),
               "answers": res.answers, "shapes": [{k: v for k, v in s.items() if not k.startswith("_")}
                                                  for s in res.shapes],
               "flags": res.flags, "fields": res.fields, "accepted": res.accepted, "reasons": res.reasons,
               "review": res.review, "usage": res.usage, "cost": round(res.cost, 6), "cached": res.cached,
               "seconds": res.seconds, "error": res.error, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with self._audit_lock:
            os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ====================================================================================================== calibration
def calibrate(cfg: dict, image_path: str, env_python="") -> dict:
    """Which box convention the model really uses: ask it to box the known objects of an example image once per
    convention and keep the one whose boxes overlap the known ones best. -> {format: mean IoU}, best."""
    known = read_labels(image_path)
    if not known:
        raise AIError("Choose an example image that already has boxes (a LabelMe JSON or YOLO txt next to it).")
    out = {}
    base = C.normalize(cfg)
    base["mode"] = "vlm_detect"
    base["roi"]["class"] = ""
    base["checks"]["accept"] = []
    base["references"]["views"] = {"default": {"match": "", "images": [], "note": ""}}
    labels = sorted({lab for lab, _ in known})
    base["classes"] = [c for c in base["classes"] if c["name"] in labels] or [C.new_class(lab) for lab in labels]
    for c in base["classes"]:
        c.update(min_confidence=0, inside="", max_per_image=0, min_size_px=0)
    for fmt in ("yxyx_1000", "xyxy_1000", "xyxy_px", "xyxy_norm"):
        c2 = json.loads(json.dumps(base))
        c2["vlm"]["box_format"] = fmt
        a = Annotator(c2, env_python)
        try:
            r = a.image(image_path)
        finally:
            a.close()
        if r.error:
            out[fmt] = 0.0
            continue
        scores = []
        for lab, kb in known:
            best = max((PA.iou(kb, s["box"]) for s in r.shapes if s["label"] == lab), default=0.0)
            scores.append(best)
        out[fmt] = round(sum(scores) / len(scores), 3) if scores else 0.0
    best = max(out, key=out.get)
    return {"scores": out, "best": best if out[best] > 0.1 else None}
