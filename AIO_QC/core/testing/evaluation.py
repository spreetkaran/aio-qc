# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
LabRun - one evaluation: dataset items + every model's cached predictions + metrics, shared by the app's
results view and the reports.

  * predictions are mapped to the dataset's classes by name (per model; unmatched model classes listed)
  * ground truth = the dataset labels (reviewed ones by default) - or, when nothing is labelled, a chosen
    reference model's detections ("vs reference": relative, not true accuracy)
  * the operating threshold per model is its best-F1 confidence, or one fixed value for all
  * comparison(a, b): per-class AP deltas and the images where one model is right and the other wrong
"""
import numpy as np

from AIO_QC.core.testing import metrics as M
from AIO_QC.core.testing.runner import load_predictions


class LabRun:
    def __init__(self, items, names, specs, preds_raw, model_names, infos=None, ground="labels", reference=None,
                 op_conf=None, source=""):
        self.items = items                     # dataset.ImageItem list (all images run)
        self.names = dict(names or {})
        self.specs = specs
        self.infos = infos or {}
        self.model_names = model_names         # per model {id: name}
        self.preds_raw = preds_raw             # per model, aligned with items: (n,6) or None
        self.ground = ground                   # 'labels' | 'reference'
        self.reference = reference
        self.op_conf = op_conf                 # None = best F1 per model, float = fixed
        self.source = source
        self.maps, self.unmatched, self.preds = [], [], []
        self.results = []
        self.eval_idx = []
        self._prepare()

    # ---------------------------------------------------------- construction
    @classmethod
    def from_cache(cls, dataset, specs, cache, keys, infos=None, only_reviewed=True, ground="auto", reference=None,
                   op_conf=None, source=""):
        preds_raw, model_names = [], []
        for s in specs:
            p, n = load_predictions(cache, s, keys)
            if getattr(s, "class_map", None):
                from AIO_QC.core.model_registry import names_after_map
                n = names_after_map(n, s.class_map)
            preds_raw.append(p)
            model_names.append(n)
        items = dataset.items
        for it in items:
            if it.meta.get("excluded") == "unreviewed pre-labels":
                it.meta.pop("excluded")
            if only_reviewed and it.labelled and not it.reviewed and not it.meta.get("excluded"):
                it.meta["excluded"] = "unreviewed pre-labels"
        if ground == "auto":
            ground = "labels" if any(it.labelled and not it.meta.get("excluded") for it in items) else (
                "reference" if len(specs) > 1 else "none")
        if ground == "reference" and reference is None:
            reference = 0
        return cls(items, dataset.names, specs, preds_raw, model_names, infos, ground, reference, op_conf, source)

    def _prepare(self):
        names = self.names
        if not names:                          # no dataset classes (reference / sampled video frames)
            from AIO_QC.core.testing.videoeval import unify_names
            names, _ = unify_names(self.model_names)
            self.names = names
        for mn in self.model_names:
            mp, matched, un = M.build_class_map(mn, self.names)
            self.maps.append(mp)
            self.unmatched.append(un)
        self.preds = [[M.remap(p, mp) if p is not None else None for p in pr] for pr, mp in zip(self.preds_raw, self.maps)]
        self.evaluate()

    def ground_truth(self):
        """(item indices, gts) used for scoring."""
        if self.ground == "labels":
            idx = [i for i, it in enumerate(self.items) if it.labelled and not it.meta.get("excluded")]
            return idx, [(self.items[i].boxes, self.items[i].cls) for i in idx]
        if self.ground == "reference" and self.reference is not None:
            ref = self.preds[self.reference]
            c = self._ref_conf()
            idx = [i for i in range(len(self.items)) if ref[i] is not None and not self.items[i].meta.get("excluded")]
            gts = []
            for i in idx:
                r = ref[i]
                k = r[r[:, 4] >= c]
                gts.append((k[:, :4], k[:, 5].astype(int)))
            return idx, gts
        return [], []

    def _ref_conf(self):
        return float(self.op_conf) if isinstance(self.op_conf, (int, float)) else 0.25

    def evaluate(self):
        self.eval_idx, gts = self.ground_truth()
        self.results = []
        for mi, s in enumerate(self.specs):
            if not self.eval_idx or (self.ground == "reference" and mi == self.reference):
                self.results.append(None)
                continue
            pr = [self.preds[mi][i] if self.preds[mi][i] is not None else np.zeros((0, 6), np.float32)
                  for i in self.eval_idx]
            r = M.evaluate(s.label, self.names, gts, pr, op_conf=self.op_conf if isinstance(self.op_conf, (int, float)) else None)
            info = self.infos.get(s.key) or {}
            r.speed_ms = info.get("speed_ms")
            r.extra.update(coverage=sum(p is not None for p in self.preds[mi]) / max(1, len(self.items)),
                           backend=info.get("backend", ""), unmatched=self.unmatched[mi])
            self.results.append(r)
        return self.results

    def set_threshold(self, op_conf):
        """Change the operating point (None = best F1 per model) - only the cheap part is recomputed."""
        self.op_conf = op_conf
        idx, gts = self.ground_truth()
        if self.ground == "reference":
            return self.evaluate()
        for mi, r in enumerate(self.results):
            if r is None:
                continue
            r.op_conf = float(r.best_conf if op_conf is None else op_conf)
            pr = [self.preds[mi][i] if self.preds[mi][i] is not None else np.zeros((0, 6), np.float32) for i in idx]
            M.analyse(r, gts, pr, max(self.names) + 1 if self.names else None)
        return self.results

    # ---------------------------------------------------------- views
    @property
    def scored(self):
        return [(i, r) for i, r in enumerate(self.results) if r is not None]

    def ranking(self):
        """Model indices best first (mAP50-95, then mAP50)."""
        return sorted((i for i, _ in self.scored), key=lambda i: (-self.results[i].map5095, -self.results[i].map50))

    def image_errors(self, mi):
        """Per scored image: (item index, tp, fp, fn, pred types, kept pred indices, gt types)."""
        r = self.results[mi]
        if r is None:
            return []
        return [(self.eval_idx[p["i"]], p["tp"], p["fp"], p["fn"], p["pred_type"], p["keep"], p["gt_type"])
                for p in r.per_image]

    def worst_images(self, mi, n=24):
        rows = self.image_errors(mi)
        rows.sort(key=lambda x: -(x[2] + x[3] * 1.5))
        return [r for r in rows if r[2] + r[3] > 0][:n]

    def comparison(self, a, b):
        """Model a vs model b on the same images."""
        ra, rb = self.results[a], self.results[b]
        if ra is None or rb is None:
            return None
        cls = sorted(set(ra.classes.tolist()) | set(rb.classes.tolist()))
        apa = {int(c): float(v) for c, v in zip(ra.classes, ra.ap[:, 0])}
        apb = {int(c): float(v) for c, v in zip(rb.classes, rb.ap[:, 0])}
        per_class = [dict(cls=c, name=self.names.get(c, str(c)), a=apa.get(c), b=apb.get(c),
                          delta=(apb.get(c, 0) - apa.get(c, 0))) for c in cls]
        ea = {p["i"]: p for p in ra.per_image}
        eb = {p["i"]: p for p in rb.per_image}
        a_better, b_better = [], []
        for i in ea:
            if i not in eb:
                continue
            sa = ea[i]["tp"] - ea[i]["fp"] - ea[i]["fn"]
            sb = eb[i]["tp"] - eb[i]["fp"] - eb[i]["fn"]
            if sa > sb:
                a_better.append((self.eval_idx[i], sa - sb))
            elif sb > sa:
                b_better.append((self.eval_idx[i], sb - sa))
        a_better.sort(key=lambda x: -x[1])
        b_better.sort(key=lambda x: -x[1])
        return dict(a=a, b=b, per_class=per_class, a_better=a_better, b_better=b_better,
                    map_delta=rb.map5095 - ra.map5095, map50_delta=rb.map50 - ra.map50,
                    speed_a=ra.speed_ms, speed_b=rb.speed_ms)

    def class_counts(self):
        idx, gts = self.ground_truth()
        out = {}
        for _, c in gts:
            for k in np.asarray(c, int):
                out[int(k)] = out.get(int(k), 0) + 1
        return out

    def verdict(self):
        """Plain-language findings (list of (kind, text)); kind: 'best' | 'fast' | 'tradeoff' | 'weak' | 'warn'."""
        out = []
        rank = self.ranking()
        if not rank:
            if self.ground == "none":
                out.append(("warn", "No ground truth: add labels (YOLO / LabelMe) or pick a reference model to score "
                                    "the others against."))
            return out
        rel = " (relative to the reference model, not true accuracy)" if self.ground == "reference" else ""
        best = self.results[rank[0]]
        txt = f"Best overall: {best.name} — mAP50-95 {best.map5095:.3f}, mAP50 {best.map50:.3f}{rel}."
        if len(rank) > 1:
            second = self.results[rank[1]]
            gap = (best.map5095 - second.map5095) / max(best.map5095, 1e-9) * 100
            txt += (f" Runner-up {second.name}: {second.map5095:.3f} ({gap:.1f}% lower)"
                    + (" — a difference this small may be noise." if gap < 2 else "."))
        out.append(("best", txt))
        timed = [(i, self.results[i].speed_ms) for i in rank if self.results[i].speed_ms]
        if timed:
            fi, fms = min(timed, key=lambda x: x[1])
            f = self.results[fi]
            out.append(("fast", f"Fastest: {f.name} at {fms:.1f} ms per image (≈{1000 / fms:.0f} images/s)"
                                + ("." if fi != rank[0] else " — also the most accurate.")))
            if fi != rank[0] and len(timed) > 2:
                # best accuracy per millisecond among the Pareto front
                front = [i for i, ms in timed if not any(self.results[j].map5095 >= self.results[i].map5095 and
                                                         ms2 < ms for j, ms2 in timed if j != i)]
                if len(front) > 2:
                    mid = sorted(front, key=lambda i: self.results[i].speed_ms)[len(front) // 2]
                    out.append(("tradeoff", f"Balanced choice: {self.results[mid].name} "
                                            f"(mAP50-95 {self.results[mid].map5095:.3f} at {self.results[mid].speed_ms:.1f} ms)."))
        for i in rank:
            r = self.results[i]
            rows = [x for x in r.class_rows() if x["gt"] >= 5]
            if len(rows) >= 2:
                w = min(rows, key=lambda x: x["ap50"])
                errs = {k: v for k, v in r.errors.items() if k != "tp" and v}
                common = max(errs, key=errs.get) if errs else None
                txt = f"{r.name}: weakest class {w['name']} (AP50 {w['ap50']:.2f}, {w['gt']} objects)"
                if common:
                    txt += f"; most common error: {M.ERROR_LABELS[common].lower()} ({errs[common]})"
                out.append(("weak", txt + "."))
        small = [self.names.get(c, str(c)) for c, n in self.class_counts().items() if n < 30]
        if small:
            out.append(("warn", f"Few examples (< 30 objects) for {', '.join(small[:6])}{'…' if len(small) > 6 else ''}"
                                " — their numbers can swing a lot; add more labelled frames."))
        for mi, un in enumerate(self.unmatched):
            if un and self.results[mi] is not None:
                out.append(("warn", f"{self.specs[mi].label}: classes not in the dataset and ignored: "
                                    f"{', '.join(un[:8])}{'…' if len(un) > 8 else ''}."))
        unrev = sum(1 for it in self.items if it.meta.get("excluded") == "unreviewed pre-labels")
        mine = sum(1 for it in self.items if it.meta.get("excluded") == "excluded by you")
        if unrev:
            out.append(("warn", f"{unrev} image(s) with unreviewed pre-labels were left out of scoring."))
        if mine:
            out.append(("warn", f"{mine} image(s) you excluded were left out of scoring."))
        return out

    def summary_table(self):
        rows = []
        for rank, i in enumerate(self.ranking(), 1):
            r = self.results[i]
            tp, fp, fn = r.tp_fp_fn()
            rows.append(dict(rank=rank, idx=i, model=r.name, map5095=r.map5095, map50=r.map50, precision=r.mp,
                             recall=r.mr, f1=r.mf1, best_conf=r.best_conf, op_conf=r.op_conf, speed_ms=r.speed_ms,
                             fps=(1000 / r.speed_ms) if r.speed_ms else None, tp=tp, fp=fp, fn=fn))
        return rows

    def kpis(self):
        """Headline tiles [(label, value, sub)] shared by the app and the reports."""
        rows = self.summary_table()
        if not rows:
            return []
        best = rows[0]
        top50 = max(rows, key=lambda x: x["map50"])
        timed = [x for x in rows if x["speed_ms"]]
        fast = min(timed, key=lambda x: x["speed_ms"]) if timed else None
        cc = self.class_counts()
        return [("Best mAP50-95", f"{best['map5095']:.3f}", best["model"]),
                ("Best mAP50", f"{top50['map50']:.3f}", top50["model"]),
                ("Fastest", f"{fast['speed_ms']:.1f} ms" if fast else "–",
                 f"{fast['model']} · ≈{fast['fps']:.0f} img/s" if fast else "not measured (cached)"),
                ("Objects scored", f"{sum(cc.values()):,}", f"{len(self.eval_idx):,} images · {len(cc)} classes")]

    def to_json(self):
        out = dict(ground=self.ground, reference=self.specs[self.reference].label if self.reference is not None else None,
                   names={str(k): v for k, v in self.names.items()}, images=len(self.items),
                   scored_images=len(self.eval_idx), source=self.source, models=[])
        for mi, s in enumerate(self.specs):
            r = self.results[mi]
            m = dict(label=s.label, path=s.path, key=s.key, settings=dict(imgsz=s.imgsz, conf_floor=s.conf, nms_iou=s.iou,
                                                                             max_det=s.max_det, half=s.half),
                     backend=(self.infos.get(s.key) or {}).get("backend"))
            if r is not None:
                m.update(r.summary())
                m["classes"] = r.class_rows()
                m["errors"] = r.errors
                m["recall_by_size"] = r.recall_by_size
                m["confusion"] = r.confusion.tolist()
                m["unscored_classes"] = [self.names.get(c, str(c)) for c in r.unscored_classes]
            out["models"].append(m)
        return out
