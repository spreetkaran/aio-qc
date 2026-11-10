# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Column profiling (Qt-free, pandas + numpy): what is in each column, what is wrong with it, and the step that would
fix it. Heavy statistics run on a seeded sample of at most 200,000 rows (exact counts stay exact).

profile(df) -> {"rows", "cols", "missing_cells", "duplicate_rows", "memory", "score", "columns": [Col…],
                "correlations": [(a, b, r)…]}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

STAT_ROWS = 200_000
HIST_BINS = 24


@dataclass
class Col:
    name: str
    dtype: str
    kind: str                         # number · integer · text · boolean · date · category · empty
    count: int
    missing: int
    distinct: int
    top: list = field(default_factory=list)          # [(value, n)]
    stats: dict = field(default_factory=dict)        # numeric / text / date statistics
    hist: tuple | None = None                        # (counts, edges) for numbers / dates (as numbers)
    warnings: list = field(default_factory=list)     # [(code, text)]
    fixes: list = field(default_factory=list)        # [(label, step)] one-click steps

    @property
    def missing_pct(self) -> float:
        return self.missing / max(1, self.count + self.missing)


def kind_of(s) -> str:
    import pandas as pd
    t = s.dtype
    if pd.api.types.is_bool_dtype(t):
        return "boolean"
    if pd.api.types.is_integer_dtype(t):
        return "integer"
    if pd.api.types.is_numeric_dtype(t):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(t):
        return "date"
    if isinstance(t, pd.CategoricalDtype):
        return "category"
    return "text"


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:,.4g}" if abs(v) < 1e6 else f"{v:,.0f}"
    return str(v)


def profile(df, sample_rows: int | None = None, seed: int = 7) -> dict:
    import numpy as np
    import pandas as pd
    n = len(df)
    sample_rows = sample_rows or STAT_ROWS                 # read now: Settings ▸ Profile can change it
    sdf = df.sample(sample_rows, random_state=seed) if n > sample_rows else df
    cols = []
    for name in df.columns:
        s = df[name]
        miss = int(s.isna().sum())
        cnt = n - miss
        try:
            distinct = int(s.nunique(dropna=True))
        except TypeError:                                              # unhashable cells (lists / dicts)
            distinct = int(s.astype("string").nunique(dropna=True))
        k = kind_of(s) if cnt else "empty"
        c = Col(str(name), str(s.dtype), k, cnt, miss, distinct)
        ss = sdf[name].dropna()
        try:
            vc = s.value_counts(dropna=True).head(10)
            c.top = [(v, int(m)) for v, m in vc.items()]
        except TypeError:
            c.top = []
        if k in ("number", "integer") and len(ss):
            a = pd.to_numeric(ss, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            a = a[np.isfinite(a)]
            if len(a):
                q1, med, q3 = np.percentile(a, [25, 50, 75])
                iqr = q3 - q1
                out = int(((a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
                sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                skew = float(((a - a.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
                c.stats = {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": sd,
                           "median": float(med), "q1": float(q1), "q3": float(q3), "zeros": int((a == 0).sum()),
                           "negative": int((a < 0).sum()), "outliers": out, "outlier_share": out / len(a),
                           "skew": skew}
                lo, hi = float(a.min()), float(a.max())
                c.hist = tuple(x.tolist() for x in np.histogram(a, bins=HIST_BINS if hi > lo else 1,
                                                                range=(lo, hi) if hi > lo else (lo - 0.5, lo + 0.5)))
        elif k == "date" and len(ss):
            d = pd.to_datetime(ss, errors="coerce").dropna()
            if len(d):
                c.stats = {"min": str(d.min()), "max": str(d.max())}
                v = d.astype("int64").to_numpy() / 1e9
                c.hist = tuple(x.tolist() for x in np.histogram(v, bins=HIST_BINS if v.max() > v.min() else 1))
        elif k in ("text", "category") and len(ss):
            t = ss.astype("string")
            lens = t.str.len()
            c.stats = {"min_len": int(lens.min()), "max_len": int(lens.max()), "mean_len": float(lens.mean()),
                       "empty": int((t.str.strip() == "").sum()),
                       "padded": int((t != t.str.strip()).sum())}
            num = pd.to_numeric(t.str.replace(",", "", regex=False), errors="coerce").notna().mean()
            c.stats["numeric_share"] = float(num)
            if num < 0.9 and t.str.len().between(6, 40).mean() > 0.9:
                c.stats["date_share"] = float(pd.to_datetime(t.head(2000), errors="coerce", format="mixed")
                                              .notna().mean())
        _judge(c, n)
        cols.append(c)
    corr = []
    num = [c.name for c in cols if c.kind in ("number", "integer") and c.distinct > 2][:40]
    if len(num) >= 2:
        m = sdf[num].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                r = m.loc[a, b]
                if r == r and abs(r) >= 0.5:
                    corr.append((a, b, float(r)))
        corr.sort(key=lambda t: -abs(t[2]))
    try:
        dups = int(df.duplicated().sum()) if n <= 5_000_000 else int(sdf.duplicated().sum())
    except TypeError:
        dups = 0
    missing_cells = int(sum(c.missing for c in cols))
    pen = 0.0
    for c in cols:
        pen += min(1.0, c.missing_pct) * 2 + 1.5 * sum(1 for w, _t in c.warnings if w in ("constant", "as_text",
                                                                                            "dates_as_text", "padded"))
    pen += 10 * (dups / max(1, n))
    score = max(0, round(100 - pen * 100 / max(1, len(cols) * 4)))
    return {"rows": n, "cols": len(cols), "missing_cells": missing_cells, "duplicate_rows": dups,
            "memory": int(df.memory_usage(deep=False).sum()), "score": score, "columns": cols,
            "correlations": corr[:30], "sampled": n > sample_rows}


def _judge(c: Col, n: int):
    W, F = c.warnings, c.fixes
    if c.kind == "empty":
        W.append(("empty", "every value is missing"))
        F.append(("Remove the column", {"op": "drop", "p": {"cols": [c.name]}}))
        return
    if c.distinct == 1 and n > 1:
        W.append(("constant", "the same value in every row"))
        F.append(("Remove the column", {"op": "drop", "p": {"cols": [c.name]}}))
    if c.missing_pct >= 0.5:
        W.append(("mostly_missing", f"{c.missing_pct:.0%} missing"))
    elif c.missing:
        how = "median" if c.kind in ("number", "integer") else "most common"
        F.append((f"Fill missing with the {how}", {"op": "fill", "p": {"cols": [c.name], "how": how, "value": ""}}))
    if c.kind in ("text", "category"):
        st = c.stats
        if c.distinct == c.count and c.count > 50:
            W.append(("unique", "every value is different (an ID?)"))
        elif c.distinct > 0.5 * c.count and c.count > 200:
            W.append(("high_cardinality", f"{c.distinct:,} different values"))
        if st.get("numeric_share", 0) >= 0.9:
            W.append(("as_text", f"{st['numeric_share']:.0%} look like numbers stored as text"))
            F.append(("Change to number", {"op": "cast", "p": {"cols": [c.name], "type": "number"}}))
        elif st.get("date_share", 0) >= 0.9:
            W.append(("dates_as_text", f"{st['date_share']:.0%} look like dates stored as text"))
            F.append(("Change to date / time", {"op": "cast", "p": {"cols": [c.name], "type": "date / time"}}))
        if st.get("padded"):
            W.append(("padded", f"{st['padded']:,} value(s) with spaces at the ends"))
            F.append(("Trim spaces", {"op": "text", "p": {"cols": [c.name], "trim": True, "spaces": True,
                                                          "case": "keep"}}))
        if st.get("empty"):
            W.append(("blank", f"{st['empty']:,} blank value(s)"))
    if c.kind in ("number", "integer") and c.stats:
        st = c.stats
        if st.get("outlier_share", 0) > 0.05:
            W.append(("outliers", f"{st['outlier_share']:.0%} outliers (IQR)"))
            F.append(("Flag outliers", {"op": "outliers", "p": {"col": c.name, "method": "IQR", "k": 1.5,
                                                                "action": "flag"}}))
        if abs(st.get("skew", 0)) > 2:
            W.append(("skewed", f"strongly skewed ({st['skew']:+.1f})"))


def describe_value(v) -> str:
    return _fmt(v)
