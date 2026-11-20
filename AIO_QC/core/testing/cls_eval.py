# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
ClsLabRun - one evaluation of image CLASSIFIERS (YOLO -cls / any model whose answer is a class per image), the
classification counterpart of evaluation.LabRun with the same shape (results, ranking, verdict, kpis, to_json …),
shared by the app's results view and the reports. Qt-free.

  * a classifier's cached answer is its top classes as whole-frame rows (detector.cls_rows): conf in column 4,
    model class id in column 5; mapped to the test set's classes by name (AIO QC's class map first)
  * ground truth = the image's class(es) (class folders / LabelMe flags) - or a reference model's top class
    ("vs reference": relative, not true accuracy)
  * per model: top-1 / top-5 accuracy, per-class precision / recall / F1 (macro averages), the confusion matrix,
    every mistake (surest first) - and at an operating confidence: how many images it answers and how often it is
    right on those (answers below it count as "unsure")
  * comparison(a, b): per-class recall deltas, the images where one is right and the other wrong
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from AIO_QC.core.testing import metrics as M


def _norm(s):
    return M._norm(str(s))


class ClsResult:
    """One classifier on the scored images."""

    def __init__(self, name):
        self.name = name
        self.n = 0
        self.top1 = self.top5 = 0.0
        self.precision = self.recall = self.f1 = 0.0          # macro over classes with images
        self.classes: dict = {}                               # name -> support / tp / fp / fn / precision / recall / f1
        self.confusion: dict = {}                             # true name -> {predicted name: n}
        self.per_image: list = []                             # (eval index, gt names, [(name, conf)], right)
        self.speed_ms = None
        self.op_conf = 0.0
        self.answered = 1.0                                   # share with a top class >= op_conf
        self.acc_answered = 0.0                               # top-1 accuracy on those
        self.curve: list = []                                 # [(conf, answered share, accuracy on answered)]
        self.extra: dict = {}

    # LabRun-compatible names for code shared with detection runs
    @property
    def map5095(self):
        return self.top1

    @property
    def map50(self):
        return self.top5

    def summary(self) -> dict:
        return dict(name=self.name, images=self.n, top1=self.top1, top5=self.top5, precision=self.precision,
                    recall=self.recall, f1=self.f1, op_conf=self.op_conf, answered=self.answered,
                    accuracy_answered=self.acc_answered, speed_ms=self.speed_ms)

    def class_rows(self) -> list:
        return [dict(name=k, **v) for k, v in sorted(self.classes.items(), key=lambda kv: kv[1]["f1"])]


def score(name, gts, preds, op_conf=0.0) -> ClsResult:
    """gts: [list of true names]; preds: [[(name, conf), …] best first] (same order)."""
    r = ClsResult(name)
    cls = defaultdict(Counter)
    conf = defaultdict(Counter)
    for i, (gt, p) in enumerate(zip(gts, preds)):
        if not gt or not p:
            continue
        r.n += 1
        top, c = p[0]
        right = top in gt
        r.top1 += right
        r.top5 += any(k in gt for k, _ in p[:5])
        r.per_image.append((i, list(gt), list(p[:5]), bool(right)))
        g = gt[0] if not right else top
        conf[g][top] += 1
        for t in gt:
            cls[t]["support"] += 1
        if right:
            cls[top]["tp"] += 1
        else:
            cls[top]["fp"] += 1
            for t in gt:
                cls[t]["fn"] += 1
    n = max(1, r.n)
    r.top1, r.top5 = r.top1 / n, r.top5 / n
    for k, c in cls.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        r.classes[k] = dict(support=c["support"], tp=tp, fp=fp, fn=fn, precision=p, recall=rc,
                            f1=2 * p * rc / (p + rc) if p + rc else 0.0)
    real = [v for v in r.classes.values() if v["support"]]
    if real:
        r.precision = sum(v["precision"] for v in real) / len(real)
        r.recall = sum(v["recall"] for v in real) / len(real)
        r.f1 = sum(v["f1"] for v in real) / len(real)
    r.confusion = {g: dict(c) for g, c in conf.items()}
    set_operating(r, op_conf)
    tops = np.array([p[0][1] for _i, _g, p, _r in r.per_image], np.float32)
    ok = np.array([rt for *_x, rt in r.per_image], bool)
    r.curve = []
    for t in np.linspace(0, 0.95, 20):
        m = tops >= t
        r.curve.append((float(t), float(m.mean()) if len(m) else 0.0, float(ok[m].mean()) if m.any() else 0.0))
    return r


def set_operating(r: ClsResult, op_conf):
    r.op_conf = float(op_conf or 0.0)
    ans = [rt for _i, _g, p, rt in r.per_image if p[0][1] >= r.op_conf]
    r.answered = len(ans) / max(1, len(r.per_image))
    r.acc_answered = sum(ans) / max(1, len(ans))


class ClsLabRun:
    task = "classify"

    def __init__(self, items, names, specs, preds_raw, model_names, infos=None, ground="labels", reference=None,
                 op_conf=None, source="", only_reviewed=True):
        self.only_reviewed = only_reviewed                   # pre-labels nobody checked yet are not ground truth
        self.items = items
        self.names = dict(names or {})
        self.specs = specs
        self.infos = infos or {}
        self.model_names = model_names
        self.preds_raw = preds_raw
        self.ground = ground
        self.reference = reference
        self.op_conf = op_conf
        self.source = source
        self.unmatched, self.preds = [], []
        self.results, self.eval_idx = [], []
        self._prepare()

    @classmethod
    def from_cache(cls, dataset, specs, cache, keys, infos=None, only_reviewed=True, ground="auto", reference=None,
                   op_conf=None, source=""):
        from AIO_QC.core.testing.runner import load_predictions
        preds_raw, model_names = [], []
        for s in specs:
            p, n = load_predictions(cache, s, keys)
            if getattr(s, "class_map", None):
                from AIO_QC.core.model_registry import names_after_map
                n = names_after_map(n, s.class_map)
            preds_raw.append(p)
            model_names.append(n)
        items = dataset.items
        if ground == "auto":
            ground = "labels" if any(it.tags for it in items if not it.meta.get("excluded")
                                     and (it.reviewed or not only_reviewed)) else (
                "reference" if len(specs) > 1 else "none")
        if ground == "reference" and reference is None:
            reference = 0
        run = cls(items, dataset.names, specs, preds_raw, model_names, infos, ground, reference, op_conf, source,
                  only_reviewed=only_reviewed)
        return run

    # ---------------------------------------------------------- construction
    def _prepare(self):
        """Every model's answers as [(dataset class name, conf)] best first; model classes the test set does not
        have keep their own name (always wrong) and are listed."""
        known = {_norm(v): v for v in self.names.values()}
        for raw, mn in zip(self.preds_raw, self.model_names):
            mn = {int(k): v for k, v in (mn or {}).items()}
            un = set()
            out = []
            for p in raw:
                if p is None or not len(p):
                    out.append(None if p is None else [])
                    continue
                a = np.asarray(p, np.float32).reshape(-1, 6)
                a = a[np.argsort(-a[:, 4])]
                row = []
                for c, k in zip(a[:, 4], a[:, 5].astype(int)):
                    nm = mn.get(int(k), str(int(k)))
                    if nm == "":                              # mapped away (class map "ignore")
                        continue
                    hit = known.get(_norm(nm))
                    if hit is None:
                        un.add(str(nm))
                    row.append((hit if hit is not None else str(nm), float(c)))
                out.append(row)
            self.preds.append(out)
            self.unmatched.append(sorted(un) if self.names else [])
        if not self.names:                                   # videos / reference only: the models' classes
            seen = []
            for pr in self.preds:
                for row in pr:
                    for nm, _c in row or []:
                        if nm not in seen:
                            seen.append(nm)
            self.names = {i: n for i, n in enumerate(seen)}
        self.evaluate()

    def _ref_conf(self):
        return float(self.op_conf) if isinstance(self.op_conf, (int, float)) else 0.0

    def ground_truth(self):
        if self.ground == "labels":
            idx = [i for i, it in enumerate(self.items) if it.tags and not it.meta.get("excluded")
                   and (it.reviewed or not self.only_reviewed)]
            return idx, [list(self.items[i].tags) for i in idx]
        if self.ground == "reference" and self.reference is not None:
            ref = self.preds[self.reference]
            c = self._ref_conf()
            idx = [i for i, p in enumerate(ref) if p and p[0][1] >= c and not self.items[i].meta.get("excluded")]
            return idx, [[ref[i][0][0]] for i in idx]
        return [], []

    def evaluate(self):
        self.eval_idx, gts = self.ground_truth()
        self.results = []
        for mi, s in enumerate(self.specs):
            if not self.eval_idx or (self.ground == "reference" and mi == self.reference):
                self.results.append(None)
                continue
            pr = [self.preds[mi][i] or [] for i in self.eval_idx]
            r = score(s.label, gts, pr, self.op_conf if isinstance(self.op_conf, (int, float)) else 0.0)
            info = self.infos.get(s.key) or {}
            r.speed_ms = info.get("speed_ms")
            r.extra.update(coverage=sum(p is not None for p in self.preds[mi]) / max(1, len(self.items)),
                           backend=info.get("backend", ""), unmatched=self.unmatched[mi])
            self.results.append(r)
        return self.results

    def set_threshold(self, op_conf):
        """Operating confidence: answers below it count as "unsure" (None = answer everything)."""
        self.op_conf = op_conf
        if self.ground == "reference":
            return self.evaluate()
        for r in self.results:
            if r is not None:
                set_operating(r, op_conf)
        return self.results

    # ---------------------------------------------------------- views
    @property
    def scored(self):
        return [(i, r) for i, r in enumerate(self.results) if r is not None]

    def ranking(self):
        return sorted((i for i, _ in self.scored), key=lambda i: (-self.results[i].top1, -self.results[i].f1))

    def mistakes(self, mi, n=None, cls=None):
        """[(item index, true names, [(name, conf)])] the model got wrong, surest first (cls: of that true class)."""
        r = self.results[mi] if 0 <= mi < len(self.results) else None
        if r is None:
            return []
        rows = [(self.eval_idx[i], g, p) for i, g, p, right in r.per_image if not right and (cls is None or cls in g)]
        rows.sort(key=lambda x: -x[2][0][1])
        return rows[:n] if n else rows

    def comparison(self, a, b):
        ra, rb = self.results[a], self.results[b]
        if ra is None or rb is None:
            return None
        names = sorted(set(ra.classes) | set(rb.classes))
        per_class = [dict(name=k, a=(ra.classes.get(k) or {}).get("recall"), b=(rb.classes.get(k) or {}).get("recall"),
                          delta=(rb.classes.get(k) or {}).get("recall", 0) - (ra.classes.get(k) or {}).get("recall", 0))
                     for k in names]
        ea = {i: (rt, p) for i, _g, p, rt in ra.per_image}
        eb = {i: (rt, p) for i, _g, p, rt in rb.per_image}
        a_better, b_better = [], []
        for i, (rta, pa) in ea.items():
            if i not in eb:
                continue
            rtb, pb = eb[i]
            if rta and not rtb:
                a_better.append((self.eval_idx[i], pb[0][1]))
            elif rtb and not rta:
                b_better.append((self.eval_idx[i], pa[0][1]))
        a_better.sort(key=lambda x: -x[1])
        b_better.sort(key=lambda x: -x[1])
        return dict(a=a, b=b, per_class=per_class, a_better=a_better, b_better=b_better,
                    top1_delta=rb.top1 - ra.top1, f1_delta=rb.f1 - ra.f1, map_delta=rb.top1 - ra.top1,
                    map50_delta=rb.top5 - ra.top5, speed_a=ra.speed_ms, speed_b=rb.speed_ms)

    def class_counts(self):
        _idx, gts = self.ground_truth()
        c = Counter(t for g in gts for t in g)
        return dict(c)

    def verdict(self):
        out = []
        rank = self.ranking()
        if not rank:
            if self.ground == "none":
                out.append(("warn", "No ground truth: give the images classes (class folders / LabelMe flags) or "
                                    "pick a reference model to score the others against."))
            return out
        rel = " (relative to the reference model, not true accuracy)" if self.ground == "reference" else ""
        best = self.results[rank[0]]
        txt = f"Best overall: {best.name} — top-1 accuracy {best.top1:.1%}, macro F1 {best.f1:.3f}{rel}."
        if len(rank) > 1:
            second = self.results[rank[1]]
            txt += f" Runner-up {second.name}: {second.top1:.1%} ({(best.top1 - second.top1) * 100:.1f} points lower)."
        out.append(("best", txt))
        timed = [(i, self.results[i].speed_ms) for i in rank if self.results[i].speed_ms]
        if timed:
            fi, ms = min(timed, key=lambda t: t[1])
            f = self.results[fi]
            out.append(("fast", f"Fastest: {f.name} — {ms:.1f} ms per image (≈{1000 / ms:.0f} images/s), top-1 "
                                f"{f.top1:.1%}."))
        weak = [(k, v) for k, v in best.classes.items() if v["support"] >= 5 and v["recall"] < 0.6]
        if weak:
            parts = []
            for k, v in sorted(weak, key=lambda kv: kv[1]["recall"])[:5]:
                mixed = {p: n for p, n in (best.confusion.get(k) or {}).items() if p != k}
                other = max(mixed, key=mixed.get) if mixed else ""
                parts.append(f"{k} ({v['recall']:.0%} found" + (f", taken for {other}" if other else "") + ")")
            out.append(("weak", f"{best.name} struggles with " + ", ".join(parts) + "."))
        small = [k for k, n in self.class_counts().items() if n < 30]
        if small:
            out.append(("warn", f"Few images (< 30) for {', '.join(small[:6])}{'…' if len(small) > 6 else ''} — "
                                "their numbers can swing a lot."))
        for mi, un in enumerate(self.unmatched):
            if un and self.results[mi] is not None:
                out.append(("warn", f"{self.specs[mi].label}: classes the test set does not have (always counted "
                                    f"wrong): {', '.join(un[:8])}{'…' if len(un) > 8 else ''}."))
        mine = sum(1 for it in self.items if it.meta.get("excluded") == "excluded by you")
        if mine:
            out.append(("warn", f"{mine} image(s) you excluded were left out of scoring."))
        return out

    def summary_table(self):
        rows = []
        for rank, i in enumerate(self.ranking(), 1):
            r = self.results[i]
            wrong = sum(1 for *_x, rt in r.per_image if not rt)
            rows.append(dict(rank=rank, idx=i, model=r.name, top1=r.top1, top5=r.top5, precision=r.precision,
                             recall=r.recall, f1=r.f1, op_conf=r.op_conf, answered=r.answered,
                             accuracy_answered=r.acc_answered, speed_ms=r.speed_ms,
                             fps=(1000 / r.speed_ms) if r.speed_ms else None, right=r.n - wrong, wrong=wrong,
                             map5095=r.top1, map50=r.top5))
        return rows

    def kpis(self):
        rows = self.summary_table()
        if not rows:
            return []
        best = rows[0]
        f1 = max(rows, key=lambda x: x["f1"])
        timed = [x for x in rows if x["speed_ms"]]
        fast = min(timed, key=lambda x: x["speed_ms"]) if timed else None
        cc = self.class_counts()
        return [("Best top-1 accuracy", f"{best['top1']:.1%}", best["model"]),
                ("Best macro F1", f"{f1['f1']:.3f}", f1["model"]),
                ("Fastest", f"{fast['speed_ms']:.1f} ms" if fast else "–",
                 f"{fast['model']} · ≈{fast['fps']:.0f} img/s" if fast else "not measured (cached)"),
                ("Images scored", f"{len(self.eval_idx):,}", f"{len(cc)} classes")]

    def to_json(self):
        out = dict(task="classify", ground=self.ground,
                   reference=self.specs[self.reference].label if self.reference is not None else None,
                   names={str(k): v for k, v in self.names.items()}, images=len(self.items),
                   scored_images=len(self.eval_idx), source=self.source, models=[])
        for mi, s in enumerate(self.specs):
            r = self.results[mi]
            m = dict(label=s.label, path=s.path, key=s.key, settings=dict(imgsz=s.imgsz),
                     backend=(self.infos.get(s.key) or {}).get("backend"))
            if r is not None:
                m.update(r.summary())
                m["classes"] = r.class_rows()
                m["confusion"] = r.confusion
                m["unmatched_classes"] = self.unmatched[mi]
            out["models"].append(m)
        return out


def is_classifier_run(run) -> bool:
    return getattr(run, "task", "") == "classify"
