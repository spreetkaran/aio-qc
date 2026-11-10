# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Recipe steps (Qt-free). A step is plain data — {"op": "filter", "p": {...}, "on": true} — and every op knows how to
write itself as pandas code AND as PySpark code. The app runs the pandas code it generates (with only df / pd / np /
tables in scope), so what you see in the app, the exported pandas script and the PySpark script are the same
recipe. Every user value reaches the code through repr(): names and values cannot inject code.

    OPS[op].pandas(p) -> [code lines]     OPS[op].spark(p) -> [code lines]     OPS[op].describe(p) -> "plain words"
    apply(df, step, tables) -> new DataFrame
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import expr as X

AGG_FUNCS = ["count", "rows", "count distinct", "sum", "mean", "median", "min", "max", "std", "first", "last"]
TYPES = ["number", "integer", "text", "boolean", "date / time", "category"]
FILTER_OPS = ["=", "≠", ">", "≥", "<", "≤", "contains", "starts with", "ends with", "matches pattern",
              "is one of", "between", "is missing", "is not missing"]
BOOL_TRUE = ["true", "1", "yes", "y", "t"]
BOOL_FALSE = ["false", "0", "no", "n", "f"]


@dataclass
class Param:
    key: str
    kind: str          # column · columns · choice · text · number · int · bool · formula · table · aggs · values
    label: str
    default: object = None
    options: list = field(default_factory=list)
    tip: str = ""
    optional: bool = False


class Op:
    key = ""
    title = ""
    group = ""
    params: list = []
    help = ""

    def describe(self, p) -> str:
        return self.title

    def pandas(self, p) -> list[str]:
        raise NotImplementedError

    def spark(self, p) -> list[str]:
        raise NotImplementedError

    def check(self, p, columns) -> str | None:
        """A problem with these parameters for the current columns (None = fine)."""
        for prm in self.params:
            v = p.get(prm.key)
            if prm.kind == "column" and not prm.optional and v not in columns:
                return f"{prm.label}: choose a column"
            if prm.kind == "columns" and not prm.optional:
                if not v:
                    return f"{prm.label}: choose at least one column"
                miss = [c for c in v if c not in columns]
                if miss:
                    return f"{prm.label}: no column {', '.join(map(repr, miss))}"
        return None


def _cols(v):
    return list(v or [])


def _q(v) -> str:
    return repr(v)


OPS: dict[str, Op] = {}


def op(cls):
    OPS[cls.key] = cls()
    return cls


# ================================================================================================ columns
@op
class Select(Op):
    key, title, group = "select", "Keep columns", "Columns"
    params = [Param("cols", "columns", "Keep")]

    def describe(self, p):
        return f"Keep {len(p['cols'])} column(s): {', '.join(p['cols'][:6])}{' …' if len(p['cols']) > 6 else ''}"

    def pandas(self, p):
        return [f"df = df[{_q(_cols(p['cols']))}]"]

    def spark(self, p):
        return [f"df = df.select(*{_q(_cols(p['cols']))})"]


@op
class Drop(Op):
    key, title, group = "drop", "Remove columns", "Columns"
    params = [Param("cols", "columns", "Remove")]

    def describe(self, p):
        return f"Remove {', '.join(p['cols'][:6])}{' …' if len(p['cols']) > 6 else ''}"

    def pandas(self, p):
        return [f"df = df.drop(columns={_q(_cols(p['cols']))})"]

    def spark(self, p):
        return [f"df = df.drop(*{_q(_cols(p['cols']))})"]


@op
class Rename(Op):
    key, title, group = "rename", "Rename column", "Columns"
    params = [Param("col", "column", "Column"), Param("new", "text", "New name")]

    def check(self, p, columns):
        return super().check(p, columns) or (None if str(p.get("new", "")).strip() else "New name: type a name")

    def describe(self, p):
        return f"Rename {p['col']} → {p['new']}"

    def pandas(self, p):
        return [f"df = df.rename(columns={{{_q(p['col'])}: {_q(p['new'])}}})"]

    def spark(self, p):
        return [f"df = df.withColumnRenamed({_q(p['col'])}, {_q(p['new'])})"]


@op
class Cast(Op):
    key, title, group = "cast", "Change type", "Columns"
    params = [Param("cols", "columns", "Columns"), Param("type", "choice", "To", "number", TYPES)]

    def describe(self, p):
        return f"{', '.join(p['cols'])} → {p['type']}"

    def pandas(self, p):
        out = []
        for c in _cols(p["cols"]):
            s = f"df[{_q(c)}]"
            t = p["type"]
            if t == "number":
                e = f"pd.to_numeric({s}, errors='coerce')"
            elif t == "integer":
                e = f"pd.to_numeric({s}, errors='coerce').round().astype('Int64')"
            elif t == "text":
                e = f"{s}.astype('string')"
            elif t == "boolean":
                m = {**{k: True for k in BOOL_TRUE}, **{k: False for k in BOOL_FALSE}}
                e = f"{s}.astype('string').str.strip().str.lower().map({_q(m)}).astype('boolean')"
            elif t == "date / time":
                e = f"pd.to_datetime({s}, errors='coerce', format='mixed')"
            else:
                e = f"{s}.astype('category')"
            out.append(f"df[{_q(c)}] = {e}")
        return out

    def spark(self, p):
        out = []
        for c in _cols(p["cols"]):
            s = f"F.col({_q(c)})"
            t = p["type"]
            if t == "number":
                e = f"{s}.cast('double')"
            elif t == "integer":
                e = f"F.round({s}.cast('double')).cast('long')"
            elif t in ("text", "category"):
                e = f"{s}.cast('string')"
            elif t == "boolean":
                v = f"F.lower(F.trim({s}.cast('string')))"
                e = f"F.when({v}.isin({_q(BOOL_TRUE)}), F.lit(True)).when({v}.isin({_q(BOOL_FALSE)}), F.lit(False))"
            else:
                e = f"F.to_timestamp({s})"
            out.append(f"df = df.withColumn({_q(c)}, {e})")
        return out


@op
class Derive(Op):
    key, title, group = "derive", "New column (formula)", "Columns"
    params = [Param("name", "text", "Name"), Param("formula", "formula", "Formula",
                                                    tip="e.g. price * qty · upper(name) · \"big\" if amount > 1000 "
                                                        "else \"small\" · year(date)")]
    help = X.__doc__

    def check(self, p, columns):
        if not str(p.get("name", "")).strip():
            return "Name: type a name"
        try:
            X.to_pandas(p.get("formula", ""), columns)
        except X.ExprError as exc:
            return f"Formula: {exc}"
        return None

    def describe(self, p):
        return f"{p['name']} = {p['formula']}"

    def pandas(self, p):
        return [f"df[{_q(p['name'])}] = {X.to_pandas(p['formula'], None, p.get('text_cols', ()))}"]

    def spark(self, p):
        return [f"df = df.withColumn({_q(p['name'])}, {X.to_spark(p['formula'], None, p.get('text_cols', ()))})"]


@op
class SplitColumn(Op):
    key, title, group = "split", "Split column", "Columns"
    params = [Param("col", "column", "Column"), Param("sep", "text", "Separator", ","),
              Param("parts", "int", "Into parts", 2), Param("prefix", "text", "New names start with", "", optional=True)]

    def describe(self, p):
        return f"Split {p['col']} at “{p['sep']}” into {p['parts']}"

    def _names(self, p):
        pre = p.get("prefix") or f"{p['col']}_"
        return [f"{pre}{i + 1}" for i in range(int(p["parts"]))]

    def pandas(self, p):
        n = int(p["parts"])
        return [f"_parts = df[{_q(p['col'])}].astype('string').str.split({_q(p['sep'])}, n={n - 1}, expand=True, "
                f"regex=False).reindex(columns=range({n}))",
                f"df[{_q(self._names(p))}] = _parts.values"]

    def spark(self, p):
        n = int(p["parts"])
        lines = [f"_parts = F.split(F.col({_q(p['col'])}).cast('string'), {_q(re.escape(p['sep']))}, {n})"]
        for i, name in enumerate(self._names(p)):
            lines.append(f"df = df.withColumn({_q(name)}, _parts.getItem({i}))")
        return lines


# ================================================================================================ rows
@op
class Filter(Op):
    key, title, group = "filter", "Filter rows", "Rows"
    params = [Param("col", "column", "Column"), Param("cmp", "choice", "Condition", "=", FILTER_OPS),
              Param("value", "text", "Value", "", optional=True), Param("value2", "text", "and", "", optional=True),
              Param("ignore_case", "bool", "Ignore case (text)", True),
              Param("mode", "choice", "Rows that match", "keep", ["keep", "remove"])]

    def describe(self, p):
        v = "" if p["cmp"] in ("is missing", "is not missing") else f" {p.get('value')!r}" + \
            (f" and {p.get('value2')!r}" if p["cmp"] == "between" else "")
        return f"{'Keep' if p['mode'] == 'keep' else 'Remove'} rows where {p['col']} {p['cmp']}{v}"

    def _cond_pd(self, p):
        s, v, cmp = f"df[{_q(p['col'])}]", p.get("value"), p["cmp"]
        ic = bool(p.get("ignore_case", True))
        txt = f"{s}.astype('string')"
        if cmp in ("=", "≠", ">", "≥", "<", "≤"):
            sym = {"=": "==", "≠": "!=", ">": ">", "≥": ">=", "<": "<", "≤": "<="}[cmp]
            return f"({s} {sym} {_q(v)})"
        if cmp == "contains":
            return f"{txt}.str.contains({_q(str(v))}, case={not ic}, regex=False, na=False)"
        if cmp in ("starts with", "ends with"):
            fn = "startswith" if cmp == "starts with" else "endswith"
            if ic:
                return f"{txt}.str.lower().str.{fn}({_q(str(v).lower())}, na=False)"
            return f"{txt}.str.{fn}({_q(str(v))}, na=False)"
        if cmp == "matches pattern":
            return f"{txt}.str.contains({_q(str(v))}, case={not ic}, regex=True, na=False)"
        if cmp == "is one of":
            return f"{s}.isin({_q(list(v) if isinstance(v, list) else [x.strip() for x in str(v).split(',')])})"
        if cmp == "between":
            return f"{s}.between({_q(v)}, {_q(p.get('value2'))})"
        if cmp == "is missing":
            return f"{s}.isna()"
        return f"{s}.notna()"

    def _cond_sp(self, p):
        s, v, cmp = f"F.col({_q(p['col'])})", p.get("value"), p["cmp"]
        ic = bool(p.get("ignore_case", True))
        txt = f"{s}.cast('string')"
        if cmp in ("=", "≠", ">", "≥", "<", "≤"):
            sym = {"=": "==", "≠": "!=", ">": ">", "≥": ">=", "<": "<", "≤": "<="}[cmp]
            return f"({s} {sym} F.lit({_q(v)}))"
        if cmp in ("contains", "starts with", "ends with"):
            fn = {"contains": "contains", "starts with": "startswith", "ends with": "endswith"}[cmp]
            if ic:
                return f"F.lower({txt}).{fn}({_q(str(v).lower())})"
            return f"{txt}.{fn}({_q(str(v))})"
        if cmp == "matches pattern":
            return f"{txt}.rlike({_q(('(?i)' if ic else '') + str(v))})"
        if cmp == "is one of":
            return f"{s}.isin({_q(list(v) if isinstance(v, list) else [x.strip() for x in str(v).split(',')])})"
        if cmp == "between":
            return f"{s}.between({_q(v)}, {_q(p.get('value2'))})"
        if cmp == "is missing":
            return f"{s}.isNull()"
        return f"{s}.isNotNull()"

    def pandas(self, p):
        c = self._cond_pd(p)
        return [f"_m = ({c}).fillna(False).astype('bool')",
                "df = df[_m]" if p["mode"] == "keep" else "df = df[~_m]"]

    def spark(self, p):
        c = f"F.coalesce({self._cond_sp(p)}, F.lit(False))"
        return [f"df = df.filter({c})" if p["mode"] == "keep" else f"df = df.filter(~{c})"]


@op
class FillMissing(Op):
    key, title, group = "fill", "Fill missing values", "Clean"
    params = [Param("cols", "columns", "Columns"),
              Param("how", "choice", "With", "value", ["value", "mean", "median", "most common", "previous row",
                                                     "next row"]),
              Param("value", "text", "Value", "", optional=True)]

    def describe(self, p):
        what = repr(p.get("value")) if p["how"] == "value" else p["how"]
        return f"Fill missing in {', '.join(p['cols'])} with {what}"

    def pandas(self, p):
        out = []
        for c in _cols(p["cols"]):
            s = f"df[{_q(c)}]"
            h = p["how"]
            if h == "value":
                e = f"{s}.fillna({_q(p.get('value'))})"
            elif h == "mean":
                e = f"{s}.fillna({s}.mean())"
            elif h == "median":
                e = f"{s}.fillna({s}.median())"
            elif h == "most common":
                e = f"{s}.fillna({s}.mode().iloc[0] if {s}.notna().any() else {s})"
            elif h == "previous row":
                e = f"{s}.ffill()"
            else:
                e = f"{s}.bfill()"
            out.append(f"df[{_q(c)}] = {e}")
        return out

    def spark(self, p):
        out, h = [], p["how"]
        if h in ("previous row", "next row"):
            out.append("df = df.withColumn('__row', F.monotonically_increasing_id())   # Spark has no row order: "
                       "the read order")
            if h == "previous row":
                out.append("_w = W.orderBy('__row').rowsBetween(W.unboundedPreceding, 0)")
            else:
                out.append("_w = W.orderBy('__row').rowsBetween(0, W.unboundedFollowing)")
        for c in _cols(p["cols"]):
            s = f"F.col({_q(c)})"
            if h == "value":
                out.append(f"df = df.withColumn({_q(c)}, F.coalesce({s}, F.lit({_q(p.get('value'))})))")
            elif h == "mean":
                out.append(f"df = df.withColumn({_q(c)}, F.coalesce({s}, F.lit(df.select(F.mean({s})).first()[0])))")
            elif h == "median":
                out.append(f"df = df.withColumn({_q(c)}, F.coalesce({s}, F.lit(df.select(F.percentile_approx({s}, "
                           f"0.5)).first()[0])))")
            elif h == "most common":
                out.append(f"_top = df.where({s}.isNotNull()).groupBy({s}).count().orderBy(F.desc('count')).first()")
                out.append(f"df = df.withColumn({_q(c)}, F.coalesce({s}, F.lit(_top[0] if _top else None)))")
            elif h == "previous row":
                out.append(f"df = df.withColumn({_q(c)}, F.last({s}, ignorenulls=True).over(_w))")
            else:
                out.append(f"df = df.withColumn({_q(c)}, F.first({s}, ignorenulls=True).over(_w))")
        if h in ("previous row", "next row"):
            out.append("df = df.drop('__row')")
        return out


@op
class DropMissing(Op):
    key, title, group = "dropna", "Remove rows with missing values", "Clean"
    params = [Param("cols", "columns", "Looking at", optional=True),
              Param("how", "choice", "Remove when", "any", ["any", "all"])]

    def describe(self, p):
        cols = p.get("cols") or []
        if not cols:
            return "Remove rows with a missing value" if p["how"] == "any" else "Remove completely empty rows"
        return f"Remove rows missing {'any' if p['how'] == 'any' else 'all'} of: {', '.join(cols)}"

    def pandas(self, p):
        return [f"df = df.dropna(subset={_q(_cols(p.get('cols')) or None)}, how={_q(p['how'])})"]

    def spark(self, p):
        return [f"df = df.dropna(how={_q(p['how'])}, subset={_q(_cols(p.get('cols')) or None)})"]


@op
class Dedupe(Op):
    key, title, group = "dedupe", "Remove duplicate rows", "Clean"
    params = [Param("cols", "columns", "Compare", optional=True),
              Param("keep", "choice", "Keep", "first", ["first", "last", "none"])]

    def describe(self, p):
        return f"Remove duplicates of {', '.join(p.get('cols') or []) or 'whole rows'} (keep {p['keep']})"

    def pandas(self, p):
        keep = {"first": "'first'", "last": "'last'", "none": "False"}[p["keep"]]
        return [f"df = df.drop_duplicates(subset={_q(_cols(p.get('cols')) or None)}, keep={keep})"]

    def spark(self, p):
        cols = _q(_cols(p.get("cols"))) if p.get("cols") else "df.columns"
        lines = ["df = df.withColumn('__row', F.monotonically_increasing_id())", f"_by = [c for c in {cols} if c != "
                 "'__row']"]
        if p["keep"] == "none":
            lines += ["df = df.withColumn('__n', F.count(F.lit(1)).over(W.partitionBy(*_by))).filter('__n = 1')"
                      ".drop('__n')"]
        else:
            order = "F.col('__row')" if p["keep"] == "first" else "F.col('__row').desc()"
            lines += [f"df = df.withColumn('__k', F.row_number().over(W.partitionBy(*_by).orderBy({order})))"
                      ".filter('__k = 1').drop('__k')"]
        lines.append("df = df.orderBy('__row').drop('__row')")
        return lines


@op
class Replace(Op):
    key, title, group = "replace", "Replace a value", "Clean"
    params = [Param("col", "column", "Column"), Param("old", "text", "Value"), Param("new", "text", "Becomes")]

    def describe(self, p):
        return f"{p['col']}: {p['old']!r} → {p['new']!r}"

    def pandas(self, p):
        return [f"df[{_q(p['col'])}] = df[{_q(p['col'])}].replace({_q(p['old'])}, {_q(p['new'])})"]

    def spark(self, p):
        s = f"F.col({_q(p['col'])})"
        return [f"df = df.withColumn({_q(p['col'])}, F.when({s} == F.lit({_q(p['old'])}), F.lit({_q(p['new'])}))"
                f".otherwise({s}))"]


@op
class TextClean(Op):
    key, title, group = "text", "Clean text", "Clean"
    params = [Param("cols", "columns", "Columns"), Param("trim", "bool", "Trim spaces", True),
              Param("spaces", "bool", "Collapse repeated spaces", True),
              Param("case", "choice", "Case", "keep", ["keep", "lower", "UPPER", "Title"])]

    def describe(self, p):
        bits = [b for b, on in (("trim", p.get("trim")), ("collapse spaces", p.get("spaces"))) if on]
        if p.get("case", "keep") != "keep":
            bits.append(p["case"])
        return f"Clean {', '.join(p['cols'])}: {', '.join(bits) or 'nothing'}"

    def pandas(self, p):
        out = []
        for c in _cols(p["cols"]):
            e = f"df[{_q(c)}].astype('string')"
            if p.get("spaces"):
                e += r".str.replace(r'\s+', ' ', regex=True)"
            if p.get("trim"):
                e += ".str.strip()"
            e += {"lower": ".str.lower()", "UPPER": ".str.upper()", "Title": ".str.title()"}.get(p.get("case"), "")
            out.append(f"df[{_q(c)}] = {e}")
        return out

    def spark(self, p):
        out = []
        for c in _cols(p["cols"]):
            e = f"F.col({_q(c)}).cast('string')"
            if p.get("spaces"):
                e = rf"F.regexp_replace({e}, r'\s+', ' ')"
            if p.get("trim"):
                e = f"F.trim({e})"
            e = {"lower": f"F.lower({e})", "UPPER": f"F.upper({e})", "Title": f"F.initcap({e})"}.get(p.get("case"), e)
            out.append(f"df = df.withColumn({_q(c)}, {e})")
        return out


@op
class Outliers(Op):
    key, title, group = "outliers", "Handle outliers", "Clean"
    params = [Param("col", "column", "Column"), Param("method", "choice", "Detect by", "IQR", ["IQR", "z-score"]),
              Param("k", "number", "Strength", 1.5, tip="IQR: 1.5 (usual) · 3 (only extreme) · z-score: 3"),
              Param("action", "choice", "Then", "flag", ["flag", "clip", "remove"])]

    def describe(self, p):
        return f"{p['action'].capitalize()} outliers of {p['col']} ({p['method']} × {p['k']})"

    def pandas(self, p):
        s, k = f"df[{_q(p['col'])}]", float(p["k"])
        if p["method"] == "IQR":
            lines = [f"_q1, _q3 = {s}.quantile(0.25), {s}.quantile(0.75)",
                     f"_lo, _hi = _q1 - {k} * (_q3 - _q1), _q3 + {k} * (_q3 - _q1)"]
        else:
            lines = [f"_mu, _sd = {s}.mean(), {s}.std()", f"_lo, _hi = _mu - {k} * _sd, _mu + {k} * _sd"]
        a = p["action"]
        if a == "flag":
            lines.append(f"df[{_q(p['col'] + '_outlier')}] = ({s} < _lo) | ({s} > _hi)")
        elif a == "clip":
            lines.append(f"df[{_q(p['col'])}] = {s}.clip(_lo, _hi)")
        else:
            lines.append(f"df = df[~(({s} < _lo) | ({s} > _hi))]")
        return lines

    def spark(self, p):
        s, k = f"F.col({_q(p['col'])})", float(p["k"])
        if p["method"] == "IQR":
            lines = [f"_q1, _q3 = df.approxQuantile({_q(p['col'])}, [0.25, 0.75], 0.001)",
                     f"_lo, _hi = _q1 - {k} * (_q3 - _q1), _q3 + {k} * (_q3 - _q1)"]
        else:
            lines = [f"_mu, _sd = df.select(F.mean({s}), F.stddev({s})).first()",
                     f"_lo, _hi = _mu - {k} * _sd, _mu + {k} * _sd"]
        a = p["action"]
        if a == "flag":
            lines.append(f"df = df.withColumn({_q(p['col'] + '_outlier')}, F.coalesce(({s} < _lo) | ({s} > _hi), "
                         "F.lit(False)))")
        elif a == "clip":
            lines.append(f"df = df.withColumn({_q(p['col'])}, F.least(F.greatest({s}, F.lit(_lo)), F.lit(_hi)))")
        else:
            lines.append(f"df = df.filter(~F.coalesce(({s} < _lo) | ({s} > _hi), F.lit(False)))")
        return lines


@op
class Sort(Op):
    key, title, group = "sort", "Sort", "Rows"
    params = [Param("cols", "columns", "By"), Param("order", "choice", "Order", "ascending", ["ascending",
                                                                                           "descending"])]

    def describe(self, p):
        return f"Sort by {', '.join(p['cols'])} ({p['order']})"

    def pandas(self, p):
        return [f"df = df.sort_values({_q(_cols(p['cols']))}, ascending={p['order'] == 'ascending'}, "
                "na_position='last', kind='stable')"]

    def spark(self, p):
        d = "asc_nulls_last" if p["order"] == "ascending" else "desc_nulls_last"
        return [f"df = df.orderBy(*[F.col(c).{d}() for c in {_q(_cols(p['cols']))}])"]


@op
class Sample(Op):
    key, title, group = "sample", "Sample rows", "Rows"
    params = [Param("fraction", "number", "Fraction (0–1)", 0.1), Param("seed", "int", "Seed", 42)]

    def describe(self, p):
        return f"Random {float(p['fraction']):.0%} of the rows (seed {p['seed']})"

    def pandas(self, p):
        return [f"df = df.sample(frac={float(p['fraction'])}, random_state={int(p['seed'])}).sort_index()"]

    def spark(self, p):
        return [f"df = df.sample(fraction={float(p['fraction'])}, seed={int(p['seed'])})"]


@op
class Limit(Op):
    key, title, group = "limit", "First rows", "Rows"
    params = [Param("n", "int", "Rows", 1000)]

    def describe(self, p):
        return f"First {int(p['n']):,} rows"

    def pandas(self, p):
        return [f"df = df.head({int(p['n'])})"]

    def spark(self, p):
        return [f"df = df.limit({int(p['n'])})"]


# ================================================================================================ combine / reshape
_AGG_PD = {"count": "count", "rows": "size", "count distinct": "nunique", "sum": "sum", "mean": "mean",
           "median": "median", "min": "min", "max": "max", "std": "std", "first": "first", "last": "last"}


def _agg_sp(func, c):
    s = f"F.col({_q(c)})"
    return {"count": f"F.count({s})", "rows": "F.count(F.lit(1))", "count distinct": f"F.countDistinct({s})",
            "sum": f"F.sum({s})", "mean": f"F.avg({s})", "median": f"F.percentile_approx({s}, 0.5)",
            "min": f"F.min({s})", "max": f"F.max({s})", "std": f"F.stddev({s})",
            "first": f"F.first({s}, ignorenulls=True)", "last": f"F.last({s}, ignorenulls=True)"}[func]


def agg_name(col, func):
    return "rows" if func == "rows" else f"{col}_{func.replace(' ', '_')}"


@op
class Group(Op):
    key, title, group = "group", "Group & summarise", "Reshape"
    params = [Param("by", "columns", "Group by", optional=True), Param("aggs", "aggs", "Summaries")]

    def check(self, p, columns):
        if not p.get("aggs"):
            return "Summaries: add at least one"
        for c, f in p["aggs"]:
            if f != "rows" and c not in columns:
                return f"Summaries: no column {c!r}"
        miss = [c for c in p.get("by") or [] if c not in columns]
        return f"Group by: no column {miss[0]!r}" if miss else None

    def describe(self, p):
        a = ", ".join(("rows" if f == "rows" else f"{f}({c})") for c, f in p["aggs"])
        return f"By {', '.join(p.get('by') or []) or 'everything'}: {a}"

    def pandas(self, p):
        by = _cols(p.get("by"))
        named = ", ".join(f"{agg_name(c, f)!r}: ({(c if f != 'rows' else (by[0] if by else c))!r}, "
                          f"{_AGG_PD[f]!r})" for c, f in p["aggs"])
        if by:
            return [f"df = df.groupby({_q(by)}, dropna=False, sort=True).agg(**{{{named}}}).reset_index()"]
        parts = ", ".join(f"{agg_name(c, f)!r}: [len(df)]" if f == "rows" else
                          f"{agg_name(c, f)!r}: [df[{c!r}].agg({_AGG_PD[f]!r})]" for c, f in p["aggs"])
        return [f"df = pd.DataFrame({{{parts}}})"]

    def spark(self, p):
        by = _cols(p.get("by"))
        aggs = ", ".join(f"{_agg_sp(f, c)}.alias({agg_name(c, f)!r})" for c, f in p["aggs"])
        if by:
            return [f"df = df.groupBy(*{_q(by)}).agg({aggs}).orderBy(*[F.col(c).asc_nulls_last() for c in {_q(by)}])"]
        return [f"df = df.agg({aggs})"]


@op
class Join(Op):
    key, title, group = "join", "Join another table", "Reshape"
    params = [Param("table", "table", "Table"), Param("left", "columns", "Match these columns"),
              Param("right", "text", "…with these columns of the other table (comma-separated; empty = same names)",
                    "", optional=True),
              Param("how", "choice", "Keep", "left", ["left", "inner", "right", "outer"])]

    def check(self, p, columns):
        return super().check(p, columns) or (None if p.get("table") else "Table: choose one")

    def _right(self, p):
        r = [x.strip() for x in str(p.get("right") or "").split(",") if x.strip()]
        return r or _cols(p["left"])

    def describe(self, p):
        return f"{p['how'].capitalize()} join {p['table']} on {', '.join(p['left'])}"

    def pandas(self, p):
        return [f"df = df.merge(tables[{_q(p['table'])}], left_on={_q(_cols(p['left']))}, "
                f"right_on={_q(self._right(p))}, how={_q(p['how'])}, suffixes=('', '_right'))"]

    def spark(self, p):
        left, right = _cols(p["left"]), self._right(p)
        how = {"outer": "full"}.get(p["how"], p["how"])
        lines = [f"_r = tables[{_q(p['table'])}]"]
        if left == right:
            lines += [f"_r = _r.select(*[F.col(c).alias(c + '_right') if c in df.columns and c not in {_q(left)} "
                      "else F.col(c) for c in _r.columns])",
                      f"df = df.join(_r, on={_q(left)}, how={_q(how)})"]
        else:
            lines += [f"_r = _r.select(*[F.col(c).alias(c + '_right') if c in df.columns else F.col(c) for c in "
                      "_r.columns])",
                      f"_rk = [c + '_right' if c in df.columns else c for c in {_q(right)}]",
                      f"df = df.join(_r, on=[df[a] == _r[b] for a, b in zip({_q(left)}, _rk)], how={_q(how)})"]
        return lines


@op
class Bin(Op):
    key, title, group = "bin", "Bin numbers", "Features"
    params = [Param("col", "column", "Column"), Param("bins", "int", "Bins", 5),
              Param("method", "choice", "Bins of", "equal width", ["equal width", "equal count"]),
              Param("name", "text", "New column", "", optional=True)]

    def describe(self, p):
        return f"{p['col']} → {p['bins']} bins of {p['method']}"

    def _name(self, p):
        return p.get("name") or f"{p['col']}_bin"

    def pandas(self, p):
        s, n = f"df[{_q(p['col'])}]", int(p["bins"])
        if p["method"] == "equal width":
            return [f"df[{_q(self._name(p))}] = pd.cut({s}, bins={n}, labels=False, include_lowest=True)"]
        return [f"df[{_q(self._name(p))}] = pd.qcut({s}, q={n}, labels=False, duplicates='drop')"]

    def spark(self, p):
        s, n = f"F.col({_q(p['col'])})", int(p["bins"])
        if p["method"] == "equal width":
            return [f"_lo, _hi = df.select(F.min({s}), F.max({s})).first()",
                    f"_w = ((_hi - _lo) / {n}) or 1.0",
                    f"df = df.withColumn({_q(self._name(p))}, F.least(F.floor(({s} - F.lit(_lo)) / F.lit(_w)), "
                    f"F.lit({n - 1})).cast('int'))"]
        return [f"_edges = sorted(set(df.approxQuantile({_q(p['col'])}, [i / {n} for i in range(1, {n})], 0.001)))",
                "from pyspark.ml.feature import Bucketizer",
                f"df = Bucketizer(splits=[-float('inf')] + _edges + [float('inf')], inputCol={_q(p['col'])}, "
                f"outputCol={_q(self._name(p))}, handleInvalid='keep').transform(df)"]


@op
class OneHot(Op):
    key, title, group = "onehot", "One-hot encode", "Features"
    params = [Param("col", "column", "Column"), Param("top", "int", "Most common values (rest → other)", 10),
              Param("drop", "bool", "Remove the original column", False)]

    def describe(self, p):
        return f"One-hot {p['col']} (top {p['top']})"

    def pandas(self, p):
        c, k = p["col"], int(p["top"])
        lines = [f"_top = df[{_q(c)}].value_counts().index[:{k}].tolist()",
                 f"for _v in _top:\n    df[{_q(c + '=')} + str(_v)] = (df[{_q(c)}] == _v).astype('int8')",
                 f"df[{_q(c + '=other')}] = (~df[{_q(c)}].isin(_top) & df[{_q(c)}].notna()).astype('int8')"]
        if p.get("drop"):
            lines.append(f"df = df.drop(columns=[{_q(c)}])")
        return lines

    def spark(self, p):
        c, k = p["col"], int(p["top"])
        s = f"F.col({_q(c)})"
        lines = [f"_top = [r[0] for r in df.where({s}.isNotNull()).groupBy({s}).count().orderBy(F.desc('count'))"
                 f".limit({k}).collect()]",
                 f"for _v in _top:\n    df = df.withColumn({_q(c + '=')} + str(_v), ({s} == F.lit(_v)).cast('int'))",
                 f"df = df.withColumn({_q(c + '=other')}, (~{s}.isin(_top) & {s}.isNotNull()).cast('int'))"]
        if p.get("drop"):
            lines.append(f"df = df.drop({_q(c)})")
        return lines


@op
class Scale(Op):
    key, title, group = "scale", "Scale numbers", "Features"
    params = [Param("cols", "columns", "Columns"),
              Param("method", "choice", "Method", "standard (z-score)", ["standard (z-score)", "min–max (0–1)",
                                                                        "robust (median / IQR)"])]

    def describe(self, p):
        return f"Scale {', '.join(p['cols'])}: {p['method']}"

    def pandas(self, p):
        out = []
        for c in _cols(p["cols"]):
            s = f"df[{_q(c)}]"
            m = p["method"]
            if m.startswith("standard"):
                e = f"({s} - {s}.mean()) / ({s}.std() or 1)"
            elif m.startswith("min"):
                e = f"({s} - {s}.min()) / (({s}.max() - {s}.min()) or 1)"
            else:
                e = f"({s} - {s}.median()) / (({s}.quantile(0.75) - {s}.quantile(0.25)) or 1)"
            out.append(f"df[{_q(c)}] = {e}")
        return out

    def spark(self, p):
        out = []
        for c in _cols(p["cols"]):
            s = f"F.col({_q(c)})"
            m = p["method"]
            if m.startswith("standard"):
                out += [f"_a, _b = df.select(F.mean({s}), F.stddev({s})).first()",
                        f"df = df.withColumn({_q(c)}, ({s} - F.lit(_a)) / F.lit(_b or 1))"]
            elif m.startswith("min"):
                out += [f"_a, _b = df.select(F.min({s}), F.max({s})).first()",
                        f"df = df.withColumn({_q(c)}, ({s} - F.lit(_a)) / F.lit((_b - _a) or 1))"]
            else:
                out += [f"_q1, _md, _q3 = df.approxQuantile({_q(c)}, [0.25, 0.5, 0.75], 0.001)",
                        f"df = df.withColumn({_q(c)}, ({s} - F.lit(_md)) / F.lit((_q3 - _q1) or 1))"]
        return out


DATE_PARTS = ["year", "month", "day", "weekday", "hour", "minute", "quarter", "week of year"]


@op
class DateParts(Op):
    key, title, group = "dateparts", "Date parts", "Features"
    params = [Param("col", "column", "Date column"),
              Param("parts", "columns", "Parts", ["year", "month", "day", "weekday"], options=DATE_PARTS)]

    def describe(self, p):
        return f"{p['col']} → {', '.join(p['parts'])}"

    def pandas(self, p):
        lines = [f"_dt = pd.to_datetime(df[{_q(p['col'])}], errors='coerce', format='mixed')"]
        attr = {"year": "year", "month": "month", "day": "day", "weekday": "weekday", "hour": "hour",
                "minute": "minute", "quarter": "quarter", "week of year": "isocalendar().week"}
        for part in _cols(p["parts"]):
            lines.append(f"df[{_q(p['col'] + '_' + part.replace(' ', '_'))}] = _dt.dt.{attr[part]}.astype('Int64')")
        return lines

    def spark(self, p):
        lines = [f"_dt = F.to_timestamp(F.col({_q(p['col'])}))"]
        fn = {"year": "F.year(_dt)", "month": "F.month(_dt)", "day": "F.dayofmonth(_dt)",
              "weekday": "F.pmod(F.dayofweek(_dt) + 5, F.lit(7))", "hour": "F.hour(_dt)", "minute": "F.minute(_dt)",
              "quarter": "F.quarter(_dt)", "week of year": "F.weekofyear(_dt)"}
        for part in _cols(p["parts"]):
            lines.append(f"df = df.withColumn({_q(p['col'] + '_' + part.replace(' ', '_'))}, {fn[part]})")
        return lines

    def check(self, p, columns):
        if p.get("col") not in columns:
            return "Date column: choose one"
        return None if p.get("parts") else "Parts: choose at least one"


# ================================================================================================ running
PRELUDE_PD = "import numpy as np\nimport pandas as pd\n"
PRELUDE_SP = ("from pyspark.sql import SparkSession, Window as W\nfrom pyspark.sql import functions as F\n")


def code_lines(step, engine: str) -> list[str]:
    o = OPS[step["op"]]
    return o.pandas(step["p"]) if engine == "pandas" else o.spark(step["p"])


def apply(df, step, tables=None):
    """Run one step on a pandas DataFrame: the generated pandas code, with only df / pd / np / tables in scope."""
    import numpy as np
    import pandas as pd
    o = OPS[step["op"]]
    err = o.check(step["p"], list(df.columns))
    if err:
        raise ValueError(err)
    # a shallow copy: steps assign whole columns, so the input (cached for earlier steps) is never changed
    ns = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": df.copy(deep=False), "tables": tables or {}}
    exec(compile("\n".join(o.pandas(step["p"])), f"<step {step['op']}>", "exec"), ns)
    out = ns["df"]
    if not isinstance(out, pd.DataFrame):
        raise ValueError("the step did not produce a table")
    return out


_SAFE_BUILTINS = {"len": len, "str": str, "range": range, "list": list, "float": float, "int": int, "set": set,
                  "sorted": sorted, "min": min, "max": max, "zip": zip, "True": True, "False": False, "None": None}
