# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Statistical QC sampling: review a (stratified) random sample and estimate the dataset error rate.

    n = sample_size(N, confidence, margin, p)            how many files to review
    plan = draw_sample(strata, n, allocation, seed)      reproducible selection
    est = estimate(strata_sizes, reviewed, flagged, confidence)   error-rate estimate + CI + verdict
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def z_value(confidence: float) -> float:
    return Z.get(round(confidence, 2), 1.96)


def sample_size(population: int, confidence: float = 0.95, margin: float = 0.05, p: float = 0.5) -> int:
    """Cochran's formula with finite-population correction."""
    if population <= 0:
        return 0
    z = z_value(confidence)
    n0 = z * z * p * (1 - p) / (margin * margin)
    n = n0 / (1 + (n0 - 1) / population)
    return min(population, max(1, math.ceil(n)))


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """(estimate, low, high) — well behaved for small n and p near 0."""
    if n == 0:
        return 0.0, 0.0, 1.0
    z = z_value(confidence)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class SamplePlan:
    name: str
    strategy: str
    seed: int
    keys: list[str]
    strata: dict[str, list[str]] = field(default_factory=dict)       # stratum -> sampled keys
    population: dict[str, int] = field(default_factory=dict)         # stratum -> N_h
    params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "strategy": self.strategy, "seed": self.seed, "keys": self.keys,
                "strata": self.strata, "population": self.population, "params": self.params}

    @classmethod
    def from_json(cls, d: dict) -> "SamplePlan":
        return cls(d["name"], d["strategy"], int(d["seed"]), list(d["keys"]), dict(d.get("strata", {})),
                   {k: int(v) for k, v in d.get("population", {}).items()}, dict(d.get("params", {})))


def allocate(pop: dict[str, int], n: int, mode: str = "proportional", min_per: int = 3) -> dict[str, int]:
    """
    proportional: n_h ∝ N_h (largest-remainder rounding)
    equal:        n / H each (capped by N_h; leftover redistributed)
    Every non-empty stratum gets at least min(min_per, N_h).
    """
    strata = [h for h, N in pop.items() if N > 0]
    if not strata or n <= 0:
        return {h: 0 for h in pop}
    total = sum(pop[h] for h in strata)
    out = {h: min(min_per, pop[h]) for h in strata}
    remaining = max(0, n - sum(out.values()))
    if mode == "equal":
        while remaining > 0:
            open_ = [h for h in strata if out[h] < pop[h]]
            if not open_:
                break
            share = max(1, remaining // len(open_))
            for h in open_:
                add = min(share, pop[h] - out[h], remaining)
                out[h] += add
                remaining -= add
                if remaining == 0:
                    break
    else:
        room = {h: pop[h] - out[h] for h in strata}
        raw = {h: remaining * pop[h] / total for h in strata}
        base = {h: min(int(raw[h]), room[h]) for h in strata}
        left = remaining - sum(base.values())
        for h in sorted(strata, key=lambda h: raw[h] - int(raw[h]), reverse=True):
            if left <= 0:
                break
            if base[h] < room[h]:
                base[h] += 1
                left -= 1
        for h in strata:
            out[h] += base[h]
    for h in pop:
        out.setdefault(h, 0)
    return out


def draw_sample(strata: dict[str, list[str]], n: int, allocation: str = "proportional", seed: int = 0,
                min_per: int = 3) -> tuple[dict[str, list[str]], dict[str, int]]:
    rng = random.Random(seed)
    pop = {h: len(keys) for h, keys in strata.items()}
    alloc = allocate(pop, n, allocation, min_per)
    picked = {h: sorted(rng.sample(sorted(keys), alloc[h])) for h, keys in strata.items() if alloc.get(h)}
    return picked, pop


@dataclass
class Estimate:
    rate: float
    low: float
    high: float
    reviewed: int
    flagged: int
    target: int
    verdict: str            # "pass" | "fail" | "undecided" | "not started"


def estimate(plan: SamplePlan, status_of, confidence: float = 0.95, max_error: float = 0.05) -> tuple[Estimate, dict]:
    """
    status_of(key) -> "ok" | "flag" | "fixed" | None (fixed counts as an error found). Stratified estimate: Σ W_h p_h with W_h = N_h / N,
    variance Σ W_h² p_h(1-p_h)/n_h · fpc_h. A single stratum reduces to the Wilson interval.
    Verdict: pass if the upper bound ≤ max_error, fail if the lower bound > max_error.
    """
    per = {}
    N = sum(plan.population.get(h, 0) for h in plan.strata) or 1
    tot_r = tot_f = 0
    for h, keys in plan.strata.items():
        st = [status_of(k) for k in keys]
        r = sum(1 for s in st if s is not None)
        f = sum(1 for s in st if s in ("flag", "fixed"))       # fixed = an error that was found and corrected
        tot_r += r
        tot_f += f
        per[h] = wilson(f, r, confidence) + (r, f, len(keys), plan.population.get(h, 0))
    target = len(plan.keys)
    if tot_r == 0:
        return Estimate(0.0, 0.0, 1.0, 0, 0, target, "not started"), per
    if len(plan.strata) <= 1:
        rate, lo, hi = wilson(tot_f, tot_r, confidence)
    else:
        z = z_value(confidence)
        rate = var = 0.0
        covered = 0.0
        for h, (p, _l, _h, r, f, _n, Nh) in per.items():
            if r == 0:
                continue
            W = Nh / N
            covered += W
            rate += W * p
            fpc = max(0.0, 1 - r / Nh) if Nh else 1.0
            # use a pseudo-count so all-OK strata still contribute uncertainty
            pp = (f + 0.5) / (r + 1)
            var += W * W * pp * (1 - pp) / r * fpc
        if covered < 1.0:           # unreviewed strata: be conservative
            rate = rate / covered if covered else 0.0
            var += (1 - covered) ** 2 * 0.25
        half = z * math.sqrt(var)
        lo, hi = max(0.0, rate - half), min(1.0, rate + half)
    verdict = "pass" if hi <= max_error else ("fail" if lo > max_error else "undecided")
    return Estimate(rate, lo, hi, tot_r, tot_f, target, verdict), per


# =============================================================================
# Label-error audit: estimate how many labels are wrong from a small reviewed sample of FINDINGS
# =============================================================================
ERROR_KINDS = {                     # finding kind -> what a confirmed finding means
    "class mismatch": "wrong class",
    "missed by model": "spurious / doubtful label",
    "poor box fit": "bad box",
    "missing annotation": "missing label",
}


@dataclass
class Audit:
    name: str
    created: str
    seed: int
    thresholds: dict
    population: dict              # "label|kind" -> number of findings of that kind for that class
    gt_counts: dict               # label -> number of labels (denominator of the error rate)
    items: list                   # sampled findings: {id, key, kind, label, shape_index, box, pred_label, conf, model}
    verdicts: dict = field(default_factory=dict)      # id -> "error" | "ok"

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict) -> "Audit":
        return cls(**{k: d[k] for k in ("name", "created", "seed", "thresholds", "population", "gt_counts", "items")},
                   verdicts=dict(d.get("verdicts", {})))


def finding_label(f: dict) -> str:
    """Missing annotations belong to the predicted class; everything else to the labelled class."""
    return f["pred_label"] if f["kind"] == "missing annotation" else f["label"]


def create_audit(name: str, findings: list[dict], gt_counts: dict, per_stratum: int, seed: int,
                 kinds: set, thresholds: dict, created: str) -> Audit:
    rng = random.Random(seed)
    strata: dict[str, list] = {}
    for f in findings:
        if f["kind"] in kinds:
            strata.setdefault(f"{finding_label(f)}|{f['kind']}", []).append(f)
    items = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda f: f["id"])
        items.extend(rng.sample(pool, min(per_stratum, len(pool))))
    return Audit(name, created, seed, dict(thresholds), {k: len(v) for k, v in strata.items()},
                 dict(gt_counts), items)


def _scaled(p_lo_hi, n):
    p, lo, hi = p_lo_hi
    return p * n, lo * n, hi * n


def estimate_audit(a: Audit, confidence: float = 0.95) -> tuple[list[dict], dict]:
    """
    Per class: for each finding kind, precision of the finder (Wilson CI from the reviewed sample) × number of
    findings = estimated real errors. Wrong-label kinds are summed into an estimated label-error RATE
    (÷ labels of that class); missing annotations are reported separately (they are not existing labels).
    Stratum bounds are summed, which is conservative (wider than a joint interval).
    """
    by_class: dict[str, dict] = {}
    for key, N in a.population.items():
        label, kind = key.split("|", 1)
        ids = [it["id"] for it in a.items if finding_label(it) == label and it["kind"] == kind]
        v = [a.verdicts.get(i) for i in ids]
        r = sum(1 for x in v if x is not None)
        e = sum(1 for x in v if x == "error")
        c = by_class.setdefault(label, {"label": label, "labels": int(a.gt_counts.get(label, 0)), "kinds": {},
                                        "err": [0.0, 0.0, 0.0], "missing": [0.0, 0.0, 0.0],
                                        "sampled": 0, "reviewed": 0, "confirmed": 0, "findings": 0})
        est = _scaled(wilson(e, r, confidence), N) if r else (0.0, 0.0, float(N))
        c["kinds"][kind] = {"N": N, "sampled": len(ids), "reviewed": r, "errors": e, "est": est}
        target = c["missing"] if kind == "missing annotation" else c["err"]
        for i in range(3):
            target[i] += est[i]
        c["sampled"] += len(ids)
        c["reviewed"] += r
        c["confirmed"] += e
        c["findings"] += N
    rows = []
    for c in sorted(by_class.values(), key=lambda c: -c["err"][0]):
        n = max(c["labels"], 1)
        c["rate"] = tuple(min(x / n, 1.0) for x in c["err"])
        c["complete"] = c["reviewed"] >= c["sampled"]
        rows.append(c)
    tot_labels = sum(c["labels"] for c in rows) or 1
    tot = {"labels": sum(c["labels"] for c in rows), "reviewed": sum(c["reviewed"] for c in rows),
           "sampled": sum(c["sampled"] for c in rows),
           "err": tuple(sum(c["err"][i] for c in rows) for i in range(3)),
           "missing": tuple(sum(c["missing"][i] for c in rows) for i in range(3))}
    tot["rate"] = tuple(min(x / tot_labels, 1.0) for x in tot["err"])
    return rows, tot
