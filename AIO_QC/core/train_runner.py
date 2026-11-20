# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC training runner — runs INSIDE the user's Python environment (torch + ultralytics), standalone: it imports nothing
from AIO QC, so it works with any environment the app can find. AIO QC copies it next to nothing else and starts it
detached, so a training run survives the app closing (or crashing):

    python -u train_runner.py <run>/aioqc/spec.json          train | resume | val (Extend Model: spec["extend"])
    python -u train_runner.py --probe                        -> one JSON line: versions, devices

spec["engine"] == "torch": a classifier of any PyTorch architecture (timm / torchvision) — aioqc_cls_engine.py, copied
next to this file, does the work (same files, same control); everything else trains with Ultralytics.

It talks to the app only through files in <run>/aioqc/:
    status.json      state, heartbeat, epoch / batch progress, last metrics, best, error (rewritten atomically)
    epochs.jsonl     one line per finished epoch: losses, metrics, learning rate, time
    stop | pause | kill   control files the app creates:
        stop   finish now and keep the results: the current epoch ends early, validation and the final check with
               best.pt run, plots are made (not resumable: the optimizer is stripped)
        pause  stop after the next saved epoch, keeping last.pt resumable
        kill   exit at once (the last saved epoch stays resumable)
The console output (Ultralytics' own log) goes to <run>/aioqc/train.log (the app redirects stdout / stderr there).
"""
import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

STATE = {}
LOCK = threading.Lock()
CTRL = {"stop": False, "pause": False}


def _write_json(path: Path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=_plain, ensure_ascii=False), encoding="utf-8")
    for _ in range(20):                       # Windows: a reader may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.05)


def _plain(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


class Status:
    def __init__(self, meta: Path):
        self.meta = meta
        self.path = meta / "status.json"

    def set(self, **kw):
        with LOCK:
            STATE.update(kw)
            STATE["updated"] = time.time()
            _write_json(self.path, STATE)

    def event(self, row: dict):
        with LOCK:
            with open(self.meta / "epochs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=_plain, ensure_ascii=False) + "\n")


IMPORT_NAME = {"opencv-python": "cv2", "pillow": "PIL", "nvidia-ml-py": "pynvml", "ultralytics-thop": "thop",
               "pyyaml": "yaml", "scikit-learn": "sklearn", "ultralytics-platform": "ultralytics_platform"}
OPTIONAL = {"nvidia-ml-py", "ultralytics-thop", "ultralytics-platform"}      # Ultralytics runs without them


def missing_requirements():
    """Ultralytics' own requirements that are not installed in this environment -> (essential, optional)."""
    import importlib.util
    import re
    from importlib.metadata import PackageNotFoundError, requires, version
    try:
        reqs = requires("ultralytics") or []
    except PackageNotFoundError:
        return [], []
    essential, optional = set(), set()
    for r in reqs:
        spec, _, marker = r.partition(";")
        if "extra" in marker:
            continue
        if marker.strip():
            try:
                from packaging.markers import Marker
                if not Marker(marker.strip()).evaluate():
                    continue
            except Exception:
                pass
        name = re.split(r"[<>=!~\[ (]", spec.strip(), maxsplit=1)[0].strip()
        if not name:
            continue
        try:
            version(name)
            continue
        except PackageNotFoundError:
            pass
        mod = IMPORT_NAME.get(name.lower(), name.lower().replace("-", "_"))
        try:
            if importlib.util.find_spec(mod) is not None:     # e.g. opencv-python-headless provides cv2
                continue
        except (ImportError, ValueError):
            pass
        (optional if name.lower() in OPTIONAL else essential).add(name)
    return sorted(essential), sorted(optional)


def release(code: int):
    """End this run and give everything back, whatever happened: GPU memory, data-loader worker processes, RAM.
    Ends with os._exit — a crashed run can otherwise hang on a worker or a non-daemon thread and keep the GPU."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    except Exception:
        pass
    _end_children()
    try:
        with LOCK:
            STATE["released"] = time.time()
            STATE["updated"] = time.time()
            _write_json(STATUS_PATH[0], STATE) if STATUS_PATH else None
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def _end_children(grace: float = 3.0):
    """Data-loader workers (and anything else this run started): asked to end, then killed."""
    try:
        import psutil                                  # an Ultralytics requirement
        kids = psutil.Process().children(recursive=True)
        for k in kids:
            try:
                k.terminate()
            except Exception:
                pass
        _gone, alive = psutil.wait_procs(kids, timeout=grace)
        for k in alive:
            try:
                k.kill()
            except Exception:
                pass
        return
    except ImportError:
        pass
    except Exception:
        return
    try:                                               # no psutil: the standard library's view of its children
        import multiprocessing as mp
        for c in mp.active_children():
            c.terminate()
            c.join(grace)
            if c.is_alive():
                c.kill()
    except Exception:
        pass


STATUS_PATH: list = []


def _on_signal(signum, _frame):
    try:
        with LOCK:
            if STATE.get("state") in ("starting", "training", "validating", None):
                STATE.update(state="stopped", message="Ended by the system or the app (signal "
                                                      f"{signum}) — the last saved epoch can be resumed",
                             resumable=True, ended=time.time())
    except Exception:
        pass
    release(4)


def probe():
    info = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
            "ultralytics": None, "torch": None, "devices": [{"id": "cpu", "name": "CPU", "mem_gb": 0}],
            "cpus": os.cpu_count() or 1}
    try:
        info["missing"], info["missing_optional"] = missing_requirements()
    except Exception as exc:
        info["missing"], info["missing_optional"] = [], []
        info["missing_error"] = str(exc)
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        info["ultralytics_error"] = f"{type(exc).__name__}: {exc}"
    try:                                               # classifiers of any architecture (aioqc_cls_engine)
        import aioqc_cls_engine
        info.update(aioqc_cls_engine.probe())
    except Exception:
        for mod in ("timm", "torchvision"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                info[mod] = None
    try:                                               # the libraries extra augmentation effects come from
        import importlib.util as _u
        info["aug_libs"] = {n: _u.find_spec(n) is not None for n in ("albumentations", "imgaug")}
    except Exception:
        info["aug_libs"] = {}
    try:                                               # its settings and their defaults (they change by version)
        from ultralytics.cfg import DEFAULT_CFG_DICT
        info["cfg_keys"] = sorted(DEFAULT_CFG_DICT)
        info["cfg"] = {k: v for k, v in DEFAULT_CFG_DICT.items()
                       if v is None or isinstance(v, (bool, int, float, str))}
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({"id": str(i), "name": f"GPU {i} · {p.name} · {p.total_memory / 2**30:.1f} GB",
                                        "mem_gb": round(p.total_memory / 2**30, 1)})
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["devices"].append({"id": "mps", "name": "Apple GPU (MPS)", "mem_gb": 0})
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _watch(meta: Path, status: Status):
    """Heartbeat + control files (every second)."""
    while True:
        time.sleep(1.0)
        for name in ("stop", "pause"):
            if (meta / name).exists() and not CTRL[name]:
                CTRL[name] = True
                status.set(request=name)
        if (meta / "kill").exists():
            status.set(state="stopped", message="Stopped at once (the last saved epoch can be resumed)",
                       resumable=True, ended=time.time())
            release(3)
        try:
            status.set()                                     # heartbeat
        except Exception:
            pass


def _metrics_of(trainer):
    m = {}
    for k, v in (getattr(trainer, "metrics", None) or {}).items():
        try:
            m[k] = round(float(v), 5)
        except (TypeError, ValueError):
            pass
    return m


def _losses_of(trainer):
    try:
        d = trainer.label_loss_items(trainer.tloss, prefix="train")
        return {k: round(float(v), 5) for k, v in d.items()}
    except Exception:
        return {}


def _per_class(metrics, names):
    """Per-class P / R / mAP50 / mAP50-95 for boxes (and masks) from a DetMetrics / SegmentMetrics."""
    out = []
    try:
        parts = [("box", metrics.box)] + ([("mask", metrics.seg)] if hasattr(metrics, "seg") else [])
        idx = list(metrics.box.ap_class_index)
        for j, c in enumerate(idx):
            row = {"id": int(c), "name": str(names.get(int(c), c)) if isinstance(names, dict) else str(c)}
            for tag, m in parts:
                row[f"{tag}_p"] = round(float(m.p[j]), 4)
                row[f"{tag}_r"] = round(float(m.r[j]), 4)
                row[f"{tag}_map50"] = round(float(m.ap50[j]), 4)
                row[f"{tag}_map"] = round(float(m.ap[j]), 4)
            out.append(row)
        nt = getattr(metrics, "nt_per_class", None)
        if nt is not None:
            for row in out:
                row["instances"] = int(nt[row["id"]])
    except Exception:
        pass
    return out


def _per_class_cls(validator, names):
    """A classifier: per class precision / recall (= its accuracy) from the confusion matrix (rows: predicted,
    columns: true). Stored under the box_* keys the run view shows; box_map / box_map50 = the class accuracy."""
    out = []
    try:
        import numpy as np
        m = np.asarray(validator.confusion_matrix.matrix, float)
        nc = min(m.shape[0], len(names) if names else m.shape[0])
        for c in range(nc):
            tp = m[c, c]
            col, row = m[:, c].sum(), m[c, :].sum()
            r = tp / col if col else 0.0
            p = tp / row if row else 0.0
            out.append({"id": c, "name": str(names.get(c, c)) if isinstance(names, dict) else str(c),
                        "box_p": round(p, 4), "box_r": round(r, 4), "box_map50": round(r, 4), "box_map": round(r, 4),
                        "instances": int(col)})
    except Exception:
        pass
    return out


def train(spec: dict, status: Status):
    from ultralytics import YOLO
    import ultralytics
    mode = spec.get("mode", "train")
    status.set(state="starting", phase="loading the model", ultralytics=ultralytics.__version__)
    weights = spec["resume_from"] if mode == "resume" else spec["model"]
    ext = spec.get("extend") if mode != "resume" else None
    if ext:                                       # Extend Model: pre-labels, the enlarged head, stage 1
        weights = extend_prepare(spec, status)
        if weights is None:
            return                                # waiting for the pre-labels to be reviewed
    model = YOLO(weights)
    t0 = time.time()
    batch = {"i": 0, "n": 0, "t": 0.0}

    def on_pretrain_end(tr):
        try:
            n_train = len(tr.train_loader.dataset)
        except Exception:
            n_train = None
        try:
            n_val = len(tr.test_loader.dataset) if tr.test_loader is not None else None
        except Exception:
            n_val = None
        status.set(state="training", phase="training", epochs=int(tr.epochs), start_epoch=int(tr.start_epoch),
                   save_dir=str(tr.save_dir), train_images=n_train, val_images=n_val,
                   batch_size=int(getattr(tr, "batch_size", 0) or 0), device=str(tr.device),
                   names={int(k): str(v) for k, v in (tr.data.get("names") or {}).items()})

    def on_epoch_start(tr):
        batch["i"] = 0
        try:
            batch["n"] = len(tr.train_loader)
        except Exception:
            batch["n"] = 0
        status.set(phase="training", epoch=int(tr.epoch) + 1, batch=0, batches=batch["n"])

    def on_batch_end(tr):
        batch["i"] += 1
        if CTRL["stop"]:
            tr.stop = True                       # ends this epoch early, then validates and keeps the results
        now = time.time()
        if now - batch["t"] > 1.0 or batch["i"] == batch["n"]:
            batch["t"] = now
            extra = {}
            try:
                import torch
                if torch.cuda.is_available() and str(tr.device).startswith(("cuda", "0", "1", "2", "3")):
                    extra["gpu_mem_gb"] = round(torch.cuda.memory_reserved() / 2**30, 2)
            except Exception:
                pass
            status.set(batch=batch["i"], batches=batch["n"], loss=_losses_of(tr), **extra)

    def on_train_epoch_end(tr):
        status.set(phase="validating")

    def on_fit_epoch_end(tr):
        final = batch.get("ended", False)        # the check of best.pt after the last epoch (same callback)
        row = {"epoch": int(tr.epoch) + (0 if final else 1), "epochs": int(tr.epochs), "loss": _losses_of(tr),
               "metrics": _metrics_of(tr), "lr": {k: float(v) for k, v in (getattr(tr, "lr", None) or {}).items()},
               "time": round(float(getattr(tr, "epoch_time", 0) or 0), 2), "fitness": _f(getattr(tr, "fitness", None)),
               "best_fitness": _f(getattr(tr, "best_fitness", None)), "final": bool(final), "t": time.time()}
        status.event(row)
        status.set(metrics=row["metrics"], best_fitness=row["best_fitness"], epoch_time=row["time"],
                   last_epoch=row["epoch"], phase="training" if not tr.stop else "final check with best.pt",
                   elapsed=round(time.time() - t0, 1))
        if tr.stop:
            batch["ended"] = True
        if CTRL["pause"] and Path(tr.last).exists():
            status.set(state="paused", message="Paused after a saved epoch — Resume continues from it",
                       resumable=True, last=str(tr.last), best=str(tr.best), ended=time.time())
            release(0)                            # before the final check strips last.pt's optimizer

    def on_train_end(tr):
        status.set(phase="finishing", best=str(tr.best), last=str(tr.last))

    model.add_callback("on_pretrain_routine_end", on_pretrain_end)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    status.set(state="starting", phase="checking the dataset")
    aug = spec.get("augment") or {}
    if aug.get("recipe"):                              # the augmentation recipe, applied in memory while training
        try:
            import aioqc_augment                       # copied next to this file by AIO QC
            line = aioqc_augment.install_in_ultralytics(aug["recipe"], float(aug.get("share", 1.0)))
            print(line, flush=True)
            status.set(augment=line)
        except Exception as exc:
            print(f"WARNING: the augmentation recipe could not be used ({type(exc).__name__}: {exc}) — training "
                  "goes on with Ultralytics' own augmentation", flush=True)
            status.set(augment=f"not used: {type(exc).__name__}: {exc}")
    if mode == "resume":
        model.train(resume=True)
    else:
        args = dict(spec["args"])
        if ext and ext.get("two_stage"):
            args.setdefault("lr0", float(ext.get("stage2_lr") or 0.001))
            args.setdefault("warmup_epochs", 0)
        model.train(**args)
    tr = model.trainer
    names = getattr(tr, "data", {}).get("names", {}) if getattr(tr, "data", None) else {}
    per = _per_class(tr.validator.metrics, names) if getattr(tr, "validator", None) is not None else []
    if not per and getattr(tr, "validator", None) is not None:
        per = _per_class_cls(tr.validator, names)
    stopped = bool(CTRL["stop"])
    if spec.get("extend") and not stopped:
        try:
            extend_compare(spec, status, str(tr.best))
        except Exception as exc:
            print(f"WARNING: the before / after check did not run ({type(exc).__name__}: {exc})", flush=True)
    status.set(state="stopped" if stopped else "finished", phase="done", per_class=per, metrics=_metrics_of(tr),
               best=str(tr.best), last=str(tr.last), save_dir=str(tr.save_dir), ended=time.time(), resumable=False,
               elapsed=round(time.time() - t0, 1),
               message=("Stopped early — the best epoch so far was checked and kept" if stopped else "Finished"))


# ====================================================================================================== extend
def _last_conv(seq):
    import torch
    convs = [m for m in seq.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


def expand_head(base, names, old_of, out):
    """The base model with a scoring head for `names`: every weight transferred, the old classes' rows of the
    class-scoring convolutions copied (new id <- old id), the new classes' rows freshly initialised. The base model
    detects exactly what it did before with it. -> how many scoring layers were enlarged."""
    import copy
    import torch
    from ultralytics import YOLO
    y = YOLO(base)
    old, task = y.model, y.task
    new = type(old)(copy.deepcopy(old.yaml), nc=len(names), verbose=False)
    new.load(old)
    ho, hn = old.model[-1], new.model[-1]
    n = 0
    if hasattr(ho, "linear") and hasattr(hn, "linear"):          # a classifier: the last linear layer's rows
        with torch.no_grad():
            # softmax: a new class starts silent (zero weights, a low bias) so the old classes score exactly as
            # before; training lifts it from its own examples
            low = float(ho.linear.bias.min()) - 12.0 if ho.linear.bias is not None else -12.0
            for ni in range(hn.linear.weight.shape[0]):
                if str(ni) not in {str(k) for k in old_of}:
                    hn.linear.weight[ni].zero_()
                    if hn.linear.bias is not None:
                        hn.linear.bias[ni] = low
            for ni, oi in old_of.items():
                ni, oi = int(ni), int(oi)
                if ni < hn.linear.weight.shape[0] and oi < ho.linear.weight.shape[0]:
                    hn.linear.weight[ni] = ho.linear.weight[oi]
                    hn.linear.bias[ni] = ho.linear.bias[oi]
        n += 1
    for attr in ("cv3", "one2one_cv3"):
        if not hasattr(ho, attr) or not hasattr(hn, attr):
            continue
        for so, sn in zip(getattr(ho, attr), getattr(hn, attr)):
            co, cn = _last_conv(so), _last_conv(sn)
            with torch.no_grad():
                for ni, oi in old_of.items():
                    ni, oi = int(ni), int(oi)
                    if ni < cn.weight.shape[0] and oi < co.weight.shape[0]:
                        cn.weight[ni] = co.weight[oi]
                        if cn.bias is not None and co.bias is not None:
                            cn.bias[ni] = co.bias[oi]
            n += 1
    new.names = {i: str(x) for i, x in enumerate(names)}
    new.args = getattr(old, "args", {})
    torch.save({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": copy.deepcopy(new).half(),
                "train_args": dict((getattr(y, "ckpt", None) or {}).get("train_args") or {}, task=task),
                # (the task, and the image size / transforms predictions use, come from here)
                "epoch": -1, "optimizer": None, "ema": None, "best_fitness": None}, out)
    return n


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _box_of(parts):
    v = [float(x) for x in parts[1:]]
    if len(v) == 4:
        x, y, w, h = v
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    xs, ys = v[0::2], v[1::2]
    return [min(xs), min(ys), max(xs), max(ys)] if xs else [0, 0, 0, 0]


def prelabel(spec, status):
    """The base model labels ITS classes on the new training images (copies only: every label file is replaced,
    never written through a hard link). An object already labelled (same class, IoU >= 0.5) is left alone."""
    from ultralytics import YOLO
    ext = spec["extend"]
    meta = Path(spec["meta"])
    tree = json.loads((meta / "tree.json").read_text(encoding="utf-8"))
    pairs = tree.get("new_train_labels") or []
    model = YOLO(ext["base"])
    conf = float(ext.get("prelabel_conf") or 0.5)
    seg = str(spec.get("task") or "") == "seg"
    keep = {int(k): int(v) for k, v in (ext.get("old_of") or {}).items()}
    old_to_new = {v: k for k, v in keep.items()}
    added, rows = 0, []
    for i, (img, lab) in enumerate(pairs, 1):
        status.set(phase=f"pre-labelling the old classes on the new images · {i:,} / {len(pairs):,}")
        try:
            r = model.predict(img, conf=conf, verbose=False)[0]
        except Exception as exc:
            print(f"pre-label {img}: {exc}", flush=True)
            continue
        lp = Path(lab)
        lines = lp.read_text(encoding="utf-8").splitlines() if lp.is_file() else []
        have = [(int(float(p[0])), _box_of(p)) for p in (ln.split() for ln in lines) if len(p) >= 5]
        new_lines = []
        boxes = r.boxes
        if boxes is None:
            continue
        for j in range(len(boxes)):
            oc = int(boxes.cls[j])
            if oc not in old_to_new:
                continue
            nc = old_to_new[oc]
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxyn[j]]
            if any(c == nc and _iou(b, [x1, y1, x2, y2]) >= 0.5 for c, b in have):
                continue
            if seg and r.masks is not None and len(r.masks.xyn) > j and len(r.masks.xyn[j]) >= 3:
                pts = " ".join(f"{v:.6f}" for xy in r.masks.xyn[j] for v in xy)
                new_lines.append(f"{nc} {pts}")
            else:
                new_lines.append(f"{nc} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            rows.append([img, model.names.get(oc, str(oc)), round(float(boxes.conf[j]), 3)])
        if new_lines:
            tmp = lp.with_name(lp.name + ".aioqc-tmp")
            tmp.write_text("\n".join(lines + new_lines) + "\n", encoding="utf-8")
            os.replace(tmp, lp)                    # a new file: the dataset's own label (if linked) is untouched
            added += len(new_lines)
    import csv
    with open(meta / "prelabels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "confidence"])
        w.writerows(rows)
    (meta / "prelabels.json").write_text(json.dumps({"added": added, "images": len(pairs), "conf": conf}),
                                         encoding="utf-8")
    return added


def _stage1(spec, status, weights):
    from ultralytics import YOLO
    ext = spec["extend"]
    run = Path(spec["meta"]).parent
    best = run / "stage1_head" / "weights" / "best.pt"
    if best.is_file():
        return str(best)
    a = dict(spec["args"])
    a.update({"epochs": int(ext.get("stage1_epochs") or 5), "freeze": int(ext.get("freeze") or 10),
              "project": str(run), "name": "stage1_head", "exist_ok": True, "patience": 0, "plots": False})
    m = YOLO(weights)

    def epoch_end(tr):
        status.set(phase=f"stage 1 of 2 · the new head only (backbone frozen) · epoch {int(tr.epoch) + 1} / "
                         f"{int(tr.epochs)}", elapsed=None)
        if CTRL["stop"]:
            tr.stop = True
    m.add_callback("on_fit_epoch_end", epoch_end)
    status.set(state="training", phase="stage 1 of 2 · the new head only (backbone frozen)")
    m.train(**a)
    return str(m.trainer.best) if Path(str(m.trainer.best)).is_file() else weights


def extend_prepare(spec, status):
    """-> the weights stage 2 starts from, or None while the pre-labels wait for a review."""
    ext = spec["extend"]
    meta = Path(spec["meta"])
    if ext.get("prelabel") and not ext.get("cls") and not (meta / "prelabels.json").exists():
        status.set(state="starting", phase="pre-labelling the old classes on the new images")
        n = prelabel(spec, status)
        if ext.get("review"):
            status.set(state="review", phase="pre-labels ready", ended=time.time(), resumable=False,
                       message=f"{n:,} pre-label(s) added for the old classes — review them (open the training "
                               "copy), then Continue")
            return None
    expanded = meta / "expanded.pt"
    if not expanded.is_file():
        status.set(state="starting", phase="enlarging the class-scoring head (old classes kept)")
        layers = expand_head(ext["base"], ext["names"], ext["old_of"], str(expanded))
        print(f"Extend: the scoring head now has {len(ext['names'])} classes ({layers} layer(s) enlarged, the "
              f"old classes' rows copied)", flush=True)
    weights = str(expanded)
    if ext.get("two_stage"):
        weights = _stage1(spec, status, weights)
    return weights


def _val_rows(model, data, status, what):
    status.set(phase=f"before / after · {what}")
    m = model.val(data=data, split="val", plots=False, verbose=False)
    names = getattr(model, "names", {}) or {}
    return [{"name": r["name"], "map": r.get("box_map"), "map50": r.get("box_map50"),
             "mask_map": r.get("mask_map")} for r in _per_class(m, names)]


def _cls_accuracy(model, tree, listing, status, what, batch=32):
    """A classifier's accuracy per class on a list of "path<TAB>class" lines (by class NAME)."""
    from collections import Counter
    status.set(phase=f"before / after · {what}")
    rows = [ln.split("\t") for ln in (tree / listing).read_text(encoding="utf-8").splitlines() if "\t" in ln]
    names = getattr(model, "names", {}) or {}
    hit, tot = Counter(), Counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        res = model.predict([str(tree / p) for p, _c in chunk], verbose=False)
        for (p, c), r in zip(chunk, res):
            tot[c] += 1
            hit[c] += str(names.get(int(r.probs.top1), "")) == c
    return [{"name": c, "map": hit[c] / tot[c], "map50": hit[c] / tot[c], "mask_map": None} for c in sorted(tot)]


def extend_compare(spec, status, best):
    """Per-class scores of the base model and the extended one on the old and the new validation images."""
    from ultralytics import YOLO
    ext = spec["extend"]
    meta = Path(spec["meta"])
    tree = Path(ext["tree"])
    out = {"before": {}, "after": {}}
    if ext.get("cls"):                                   # classifiers: accuracy per class, by name
        after, base = YOLO(best), YOLO(ext["base"])
        for split in ("val_old", "val_new"):
            if (tree / f"{split}.txt").is_file():
                out["after"][split] = _cls_accuracy(after, tree, f"{split}.txt", status, f"the extended model on {split}")
                out["before"][split] = _cls_accuracy(base, tree, f"{split}.txt", status, f"the base model on {split}")
        (meta / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out
    after = YOLO(best)
    for split in ("val_old", "val_new"):
        y = tree / f"{split}.yaml"
        if y.is_file():
            out["after"][split] = _val_rows(after, str(y), status, f"the extended model on {split}")
    base = YOLO(ext["base"])
    if ext.get("old_yaml"):
        out["before"]["val_old"] = _val_rows(base, ext["old_yaml"], status, "the base model on val_old")
    if ext.get("goal") == "images" and (tree / "val_new.yaml").is_file() and \
            len(ext["names"]) == len(ext.get("base_names") or []):
        out["before"]["val_new"] = _val_rows(base, str(tree / "val_new.yaml"), status, "the base model on val_new")
    (meta / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _f(v):
    try:
        return round(float(v), 5)
    except (TypeError, ValueError):
        return None


def validate(spec: dict, status: Status):
    from ultralytics import YOLO
    status.set(state="starting", phase="loading the model")
    model = YOLO(spec["model"])
    status.set(state="validating", phase=f"validating on {spec['args'].get('split', 'val')}")
    m = model.val(**spec["args"])
    names = getattr(model, "names", {}) or {}
    res = {}
    for tag, part in (("box", getattr(m, "box", None)), ("mask", getattr(m, "seg", None))):
        if part is not None:
            res[f"{tag}_map50"] = round(float(part.map50), 5)
            res[f"{tag}_map"] = round(float(part.map), 5)
            res[f"{tag}_p"] = round(float(part.mp), 5)
            res[f"{tag}_r"] = round(float(part.mr), 5)
    if getattr(m, "top1", None) is not None:          # a classifier (ClassifyMetrics)
        res["metrics/accuracy_top1"] = round(float(m.top1), 5)
        res["metrics/accuracy_top5"] = round(float(m.top5), 5)
    status.set(state="finished", phase="done", metrics=res, per_class=_per_class(m, names),
               save_dir=str(getattr(m, "save_dir", "")), ended=time.time(), message="Validation finished")


def _provenance_finish(spec):
    """Training ended: what the model came from (written by AIO QC into spec["provenance"]) goes into best.pt and
    best.onnx, and the run folder's record gets the weights' SHA-256. Standalone like the rest of this file."""
    rec = spec.get("provenance")
    if not isinstance(rec, dict) or spec.get("mode") == "val" or STATE.get("state") != "finished":
        return
    try:
        import hashlib
        save = Path(str(STATE.get("save_dir") or ""))
        best = Path(str(STATE.get("best") or (save / "weights" / "best.pt")))
        if not best.is_file():
            return
        rec = dict(rec, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            import torch
            ck = torch.load(str(best), map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                ck["provenance"] = json.dumps(rec, ensure_ascii=False, default=str)
                tmp = best.with_name(best.name + ".prov.tmp")
                torch.save(ck, str(tmp))
                os.replace(tmp, best)
        except Exception as exc:                          # noqa: BLE001 — the weights stay as trained
            print(f"note: provenance not added to {best.name} ({type(exc).__name__})", flush=True)
        onnx = best.with_suffix(".onnx")
        if onnx.is_file():
            try:
                import aioqc_cls_engine as E               # its metadata writer (no onnx package needed)
                E.add_onnx_metadata(onnx, {"provenance": json.dumps(rec, ensure_ascii=False, default=str)})
            except Exception as exc:                      # noqa: BLE001
                print(f"note: provenance not added to {onnx.name} ({type(exc).__name__})", flush=True)

        def sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        weights = {p.name: sha(p) for p in (best, best.with_name("last.pt"), onnx) if p.is_file()}
        side = save / ".aio_qc_provenance.json"
        try:
            doc = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {"format": "aio-qc-provenance/1", "records": []}
        doc.setdefault("records", []).append({"at": rec["finished"], "action": "trained", "tool": "Train Model",
                                              "software": rec.get("software", "AIO QC"), "weights": weights,
                                              "metrics": STATE.get("metrics")})
        _write_json(side, doc)
    except Exception as exc:                              # noqa: BLE001 — never fails a finished run
        print(f"note: provenance not recorded ({type(exc).__name__}: {exc})", flush=True)


def _aioqc_classifier(spec) -> bool:
    """A model trained by aioqc_cls_engine (its best.pt has a best.aioqc.json beside it): that engine goes on with it
    (fine-tuning, validating, extending) — Ultralytics can predict with it, not train it."""
    m = str(spec.get("resume_from") or spec.get("model") or "")
    if not m.lower().endswith((".pt", ".pth")):
        return False
    p = Path(m)
    return p.with_name(p.stem + ".aioqc.json").is_file() or (p.name == "last.pt" and
                                                             p.with_name("best.aioqc.json").is_file())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        print(json.dumps(probe()))
        return
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    meta = spec_path.parent
    status = Status(meta)
    STATUS_PATH.append(status.path)
    import signal
    for sig in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), _on_signal)
            except (ValueError, OSError):
                pass
    STATE.update(json.loads((meta / "status.json").read_text(encoding="utf-8")) if (meta / "status.json").exists()
                 else {})
    for name in ("stop", "pause", "kill"):
        try:
            (meta / name).unlink()
        except OSError:
            pass
    STATE.update({"request": None, "error": None, "error_tail": None})
    status.set(state="starting", phase="importing torch / ultralytics", pid=os.getpid(), mode=spec.get("mode", "train"),
               started=time.time(), python=sys.executable, host=platform.node())
    threading.Thread(target=_watch, args=(meta, status), daemon=True).start()
    if spec.get("cwd"):
        try:
            os.makedirs(spec["cwd"], exist_ok=True)
            os.chdir(spec["cwd"])                    # downloaded weights land here (shared by every run)
        except OSError:
            pass
    torch_engine = spec.get("engine") == "torch" or _aioqc_classifier(spec)
    try:
        miss, _opt = ([], []) if torch_engine else missing_requirements()
    except Exception:
        miss = []
    try:
        if torch_engine:                                  # any PyTorch classifier (timm / torchvision)
            import aioqc_cls_engine                       # copied next to this file by AIO QC
            aioqc_cls_engine.run(spec, status, CTRL, release, meta)
        elif miss:                                          # fail now, not hours later (polars: at the first plot)
            raise ModuleNotFoundError(f"No module named '{miss[0]}' — this environment lacks packages Ultralytics "
                                      f"needs: {', '.join(miss)}")
        elif spec.get("mode") == "val":
            validate(spec, status)
        else:
            train(spec, status)
        _provenance_finish(spec)
    except KeyboardInterrupt:
        status.set(state="stopped", message="Stopped", ended=time.time(), resumable=True)
        release(2)
    except BaseException as exc:                     # noqa: BLE001 - everything is reported to the app
        tb = traceback.format_exc()
        print(tb, flush=True)
        status.set(state="failed", error=f"{type(exc).__name__}: {exc}", error_tail=tb[-4000:], ended=time.time(),
                   resumable=bool(STATE.get("last_epoch")))
        release(1)
    release(0)


if __name__ == "__main__":
    main()
