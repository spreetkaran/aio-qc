# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Auto-Annotate's Annotate step with the AI annotator (Qt-free): every frame of the run folder through the project's
AI, concurrently, within the budget and targets; accepted images get YOLO txt / LabelMe files, rejected ones go to
rejected/ with their reasons (put back when the step runs again).
"""
from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
from AIO_QC.tools.frame_extractor.core.ai_annotate import writers as W
from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import AIError, Annotator


def _dur(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def load_project(path):
    from AIO_QC.tools.frame_extractor.core.autoannotate.engine import StepError
    try:
        cfg = C.load(path)
    except FileNotFoundError:
        raise StepError("annotate", f"The AI annotate project was not found: {path}") from None
    except C.ConfigError as exc:
        raise StepError("annotate", f"The AI annotate project has a problem"
                                    f"{f' on line {exc.line}' if exc.line else ''}: {exc}") from None
    except RuntimeError as exc:
        raise StepError("annotate", str(exc)) from None
    probs = C.check(cfg, os.path.dirname(path))
    if probs:
        raise StepError("annotate", "AI annotate: " + " ".join(m for _, m in probs[:4]))
    return cfg


def stage_ai(ws, r, ctx, env_python="") -> dict:
    from AIO_QC.tools.frame_extractor.core.autoannotate.engine import Stopped, StepError, plan_todo
    cfg = load_project(r.ai_config)
    ws.restore(["annotate", "rules"])
    imgs = ws.frame_images()
    if not imgs:
        raise StepError("annotate", "There are no frames yet — run Frames first (or choose an image folder).")
    out = cfg["output"]
    names = C.class_names(cfg)
    lim = cfg["limits"]
    targets = {k: int(v) for k, v in (lim.get("targets") or {}).items() if int(v or 0) > 0}
    if targets or int(lim.get("max_images") or 0):
        random.Random(int(lim.get("seed") or 0)).shuffle(imgs)
    key = "ai:" + C.run_key(cfg) + json.dumps({k: out.get(k) for k in ("format", "labelme_version", "image_data",
                                                                          "write_confidence")}, sort_keys=True)
    import hashlib
    key = hashlib.sha1(key.encode()).hexdigest()[:16]
    ws.set_stage("annotate", status="running", key=key)
    ext = ".json" if out["format"] in ("labelme", "both") else ".txt"
    todo, skipped, kept_yours, kept_other = plan_todo(
        ws, imgs, key, r.existing, ext, "keep" if out.get("merge_with_existing") == "skip" else "process")
    if int(lim.get("max_images") or 0):
        todo = todo[: max(0, int(lim["max_images"]) - skipped)]
    if out["format"] in ("yolo", "both") and names:
        W.write_classes({os.path.dirname(p) for p in imgs} | {ws.frames}, names)
        if out.get("data_yaml", True):
            W.write_data_yaml(ws.frames, names, train=".", val=".")
    from AIO_QC.core import provenance as PV
    vlm = cfg["vlm"].get("model") or ""
    hosted = bool(vlm) and (cfg.get("connection") or {}).get("provider") != "local"
    prov = PV.make_origin("model", "AI Annotate",                      # a hosted service's model: not by name
                          model=PV.HOSTED if hosted else (vlm or cfg["detector"].get("model") or ""),
                          detail=C.MODES.get(cfg["mode"], cfg["mode"]))
    budget = float(lim.get("budget_usd") or 0)
    total = len(todo)
    ctx.progress(0, total, "Loading")
    try:
        ann = Annotator(cfg, env_python, os.path.join(ws.meta, "ai_cache.db"),
                        os.path.join(ws.reports, "ai_audit.jsonl"), ctx.cancel.is_set,
                        os.path.dirname(r.ai_config))
    except AIError as exc:
        raise StepError("annotate", str(exc)) from None
    con = cfg["connection"]
    workers = 1 if ann.worker is not None and cfg["connection"].get("provider") == "local" else \
        max(1, int(con.get("workers") or 4))
    ctx.log(f"AI annotate · {C.MODES.get(cfg['mode'], cfg['mode'])} · {cfg['vlm'].get('model') or cfg['detector']['model']}"
            f" · {total:,} images to do ({skipped:,} up to date) · {workers} at a time")
    st = {"accepted": 0, "rejected": 0, "review": 0, "failed": 0, "boxes": 0, "calls": 0, "cached": 0, "input": 0,
          "output": 0, "cost": 0.0}
    per_class, have_class = {}, {}
    rejected_rows, failed_names = [], []
    lock = threading.Lock()
    t0 = time.monotonic()
    stop_reason = ""

    def targets_met():
        return bool(targets) and all(have_class.get(k, 0) >= v for k, v in targets.items())

    def handle(path, res):
        with lock:
            st["calls"] += res.calls
            st["cached"] += res.cached
            st["input"] += res.usage.get("input", 0)
            st["output"] += res.usage.get("output", 0)
            st["cost"] += res.cost
        if res.error:
            if "stopped" in res.error and ctx.cancel.is_set():
                return
            with lock:
                st["failed"] += 1
                failed_names.append(f"{os.path.basename(path)}: {res.error}")
            ctx.log(f"{os.path.basename(path)}: {res.error}")
            return
        name = os.path.basename(path)
        if res.accepted:
            W.write(path, res.width, res.height, res.shapes, out, names, dict(res.flags,
                                                                             **({"needs_review": True}
                                                                                if res.review else {})),
                    origin=prov)
            outp = os.path.splitext(path)[0] + ext
            if os.path.exists(outp):
                s = os.stat(outp)
                cnt = res.counts()
                ws.db.execute("INSERT OR REPLACE INTO anns VALUES (?,?,?,?,?,?)",
                              (ws.rel(path), key, s.st_mtime_ns, s.st_size, len(res.shapes), json.dumps(cnt)))
            meta = {"texts": [{"label": s["label"], "text": s.get("text"), "text_ok": s.get("text_ok")}
                              for s in res.shapes if s.get("text")],
                    "confidences": [round(s["conf"], 3) for s in res.shapes], "flags": res.flags,
                    "fields": res.fields, "needs_review": res.review}
            ws.db.execute("INSERT OR REPLACE INTO ai_meta VALUES (?,?)", (name, json.dumps(meta, default=str)))
            with lock:
                st["accepted"] += 1
                st["review"] += res.review
                st["boxes"] += len(res.shapes)
                for lab, n in res.counts().items():
                    per_class[lab] = per_class.get(lab, 0) + n
                    have_class[lab] = have_class.get(lab, 0) + 1
        else:
            if cfg["checks"].get("keep_rejected", True):
                W.write(path, res.width, res.height, res.shapes, dict(out, merge_with_existing="replace"), names,
                        dict(res.flags, rejected=True), origin=prov)
                outp = os.path.splitext(path)[0] + ext
                if os.path.exists(outp):                 # recorded as "rejected", so the next run asks again
                    s = os.stat(outp)                    # (from the cache: free) instead of keeping it
                    ws.db.execute("INSERT OR REPLACE INTO anns VALUES (?,?,?,?,?,?)",
                                  (ws.rel(path), "rejected:" + key, s.st_mtime_ns, s.st_size, 0, "{}"))
            for f in [path] + [os.path.splitext(path)[0] + e for e in (".json", ".txt")
                               if os.path.exists(os.path.splitext(path)[0] + e)]:
                ws.move("annotate", f, os.path.join(ws.rejected, os.path.relpath(f, ws.frames)))
            with lock:
                st["rejected"] += 1
                rejected_rows.append((ws.rel(os.path.join(ws.rejected, os.path.relpath(path, ws.frames))),
                                      "; ".join(res.reasons)))

    done = 0
    attempts_cap = int(sum(targets.values()) * float(lim.get("max_attempts_factor") or 0)) if targets else 0
    it = iter(todo)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            running = {}

            def fill():
                while len(running) < (workers if targets or budget else workers * 2) and \
                        not ctx.cancel.is_set() and not stop_reason:
                    p = next(it, None)
                    if p is None:
                        return
                    running[ex.submit(ann.image, p)] = p
            fill()
            while running:
                fin, _ = wait(list(running), timeout=0.5, return_when=FIRST_COMPLETED)
                for f in fin:
                    p = running.pop(f)
                    handle(p, f.result())
                    done += 1
                if fin:
                    ws.db.commit()
                    el = time.monotonic() - t0
                    rate = done / el if el > 0 else 0
                    eta = (total - done) / rate if rate and done < total else 0
                    ctx.progress(done, total, f"{done:,} / {total:,} · kept {st['accepted']:,} · rejected "
                                              f"{st['rejected']:,}" + (f" · ${st['cost']:.2f}" if st["cost"] else "")
                                              + (f" · {_dur(eta)} left" if eta else ""))
                if budget and st["cost"] >= budget and not stop_reason:
                    stop_reason = f"the budget of ${budget:g} is reached (${st['cost']:.2f})"
                if targets_met() and not stop_reason:
                    stop_reason = "every target is reached"
                if attempts_cap and done >= attempts_cap and not stop_reason:
                    stop_reason = f"{done:,} attempts (the limit for these targets)"
                if not ctx.cancel.is_set() and not stop_reason:
                    fill()
    finally:
        ws.db.commit()
        ann.close()
        if rejected_rows:
            os.makedirs(ws.reports, exist_ok=True)
            path = os.path.join(ws.reports, "rejected.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["image", "reasons"])
                w.writerows(rejected_rows)
    if ctx.cancel.is_set():
        raise Stopped()
    if stop_reason.startswith("the budget"):
        ctx.log("stopped: " + stop_reason)
        raise Stopped(stop_reason[0].upper() + stop_reason[1:] + " — raise the budget and Run to continue.")
    if stop_reason:
        ctx.log("finished early: " + stop_reason)
    res = {"frames": len(imgs), "annotated": st["accepted"], "rejected": st["rejected"], "needs_review": st["review"],
           "failed": st["failed"], "up_to_date": skipped, "boxes": st["boxes"], "classes": dict(sorted(per_class.items())),
           "calls": st["calls"], "cached_answers": st["cached"], "tokens_in": st["input"], "tokens_out": st["output"],
           "cost_usd": round(st["cost"], 4), "format": out["format"], "seconds": round(time.monotonic() - t0, 1)}
    if stop_reason:
        res["finished_early"] = stop_reason
    if kept_yours:
        res["edited_kept"] = kept_yours
    if kept_other:
        res["not_ours_kept"] = kept_other
    if failed_names:
        ctx.log("failed: " + "; ".join(failed_names[:5]) + (" …" if len(failed_names) > 5 else ""))
        if st["accepted"] == 0 and st["rejected"] == 0 and len(failed_names) == total:
            raise StepError("annotate", "Every image failed — " + failed_names[0].split(": ", 1)[-1])
    return res
