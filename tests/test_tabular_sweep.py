# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Data Studio recipe steps and formulas, every op and every option: the generated pandas code runs on a small
table (for at least one fitting column), and the generated PySpark code is valid Python. Plus the formula language:
each function evaluates as documented, and anything outside it is refused."""
import ast
import itertools

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from AIO_QC.core.tabular import expr as E  # noqa: E402
from AIO_QC.core.tabular import steps as S  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": [" Ann  Lee", "bob", "Bob", None, "cara", "dan"],
        "price": [1.5, 2.0, np.nan, 40.0, 3.0, 2.0],
        "qty": [1, 2, 3, 4, 100, 2],
        "when": ["2024-01-05 10:30", "2024-02-10 08:00", None, "2024-03-01 00:00", "bad", "2024-12-31 23:59"],
        "tags": ["a,b", "b", "a,c,d", "", None, "c"],
    })


COLUMN_TRY = ["price", "qty", "name", "when", "tags"]
COLUMNS_TRY = [["price", "qty"], ["name"], ["when"], ["qty"]]
TEXT = {"new": "renamed", "name": "made", "sep": ",", "prefix": "part", "old": "bob", "value": "2", "value2": "30",
        "right": ""}


def _variants(op):
    """Defaults, then each choice / bool option in turn."""
    base = {}
    for prm in op.params:
        if prm.kind in ("choice",):
            base[prm.key] = prm.default if prm.default is not None else prm.options[0]
        elif prm.kind == "bool":
            base[prm.key] = bool(prm.default)
        elif prm.kind in ("number", "int"):
            base[prm.key] = prm.default
        elif prm.kind == "text":
            base[prm.key] = TEXT.get(prm.key, prm.default if prm.default is not None else "x")
        elif prm.kind == "formula":
            base[prm.key] = "price * qty"
        elif prm.kind == "table":
            base[prm.key] = "other"
        elif prm.kind == "aggs":
            base[prm.key] = [["price", "sum"], ["qty", "mean"], ["name", "rows"]]
    yield dict(base)
    for prm in op.params:
        if prm.kind == "choice":
            for o in prm.options:
                if o != base[prm.key]:
                    yield {**base, prm.key: o}
        elif prm.kind == "bool":
            yield {**base, prm.key: not base[prm.key]}


def _with_columns(op, p):
    for prm in op.params:
        if prm.kind == "columns" and prm.options:                     # a pick list of its own (date parts …)
            p = {**p, prm.key: list(prm.options)}
    cols = [prm for prm in op.params if prm.kind in ("column", "columns") and not prm.options]
    pools = [COLUMN_TRY if prm.kind == "column" else COLUMNS_TRY for prm in cols]
    for combo in itertools.product(*pools):
        q = dict(p)
        for prm, v in zip(cols, combo):
            q[prm.key] = v
        yield q


def test_every_step_option_runs_in_pandas_and_writes_valid_spark(df):
    other = pd.DataFrame({"qty": [1, 2, 3], "colour": ["r", "g", "b"]})
    tried = 0
    for key, op in S.OPS.items():
        for p in _variants(op):
            ok, errs = False, []
            for q in _with_columns(op, p):
                step = {"op": key, "p": q, "on": True}
                if op.check(q, list(df.columns)):
                    continue
                spark = "\n".join(S.code_lines(step, "spark"))
                ast.parse(spark)                                       # valid Python, whatever the data
                assert op.describe(q)
                try:
                    out = S.apply(df, step, {"other": other})
                except Exception as exc:                               # this column does not fit this option
                    errs.append(f"{q}: {type(exc).__name__}: {exc}")
                    continue
                assert isinstance(out, pd.DataFrame)
                ok = True
                tried += 1
                break
            assert ok, f"{key} {p}: no column worked\n" + "\n".join(errs[:4])
    assert tried > 60


def test_steps_check_and_leave_the_input_alone(df):
    before = df.copy()
    with pytest.raises(ValueError, match="choose a column"):
        S.apply(df, {"op": "filter", "p": {"col": "nope", "cmp": "=", "mode": "keep"}})
    with pytest.raises(ValueError, match="at least one"):
        S.apply(df, {"op": "select", "p": {"cols": []}})
    with pytest.raises(ValueError):
        S.apply(df, {"op": "rename", "p": {"col": "qty", "new": "  "}})
    with pytest.raises(ValueError):
        S.apply(df, {"op": "group", "p": {"by": ["name"], "aggs": []}})
    out = S.apply(df, {"op": "derive", "p": {"name": "total", "formula": "price * qty"}})
    assert list(out["total"].fillna(-1)) == [1.5, 4.0, -1, 160.0, 300.0, 4.0]
    out = S.apply(df, {"op": "filter", "p": {"col": "qty", "cmp": "between", "value": 2, "value2": 4,
                                             "mode": "keep", "ignore_case": True}})
    assert list(out["qty"]) == [2, 3, 4, 2]
    out = S.apply(df, {"op": "filter", "p": {"col": "name", "cmp": "is one of", "value": "bob, dan",
                                             "mode": "remove", "ignore_case": False}})
    assert "bob" not in list(out["name"]) and len(out) == 4
    pd.testing.assert_frame_equal(df, before)
    assert S.PRELUDE_PD and S.PRELUDE_SP


FORMULAS = {
    "price * qty": [1.5, 4.0, None, 160.0, 300.0, 4.0],
    "qty // 3": [0, 0, 1, 1, 33, 0],
    "-qty % 3": [2, 1, 0, 2, 2, 1],
    "qty ** 2": [1, 4, 9, 16, 10000, 4],
    "'big' if qty > 3 else 'small'": ["small", "small", "small", "big", "big", "small"],
    "upper(strip(name))": ["ANN  LEE", "BOB", "BOB", None, "CARA", "DAN"],
    "len(name)": [9, 3, 3, None, 4, 3],
    "contains(lower(name), 'b')": [False, True, True, False, False, False],       # missing: no match
    "startswith(name, 'B')": [False, False, True, False, False, False],
    "replace(name, 'b', 'p')": [" Ann  Lee", "pop", "Bop", None, "cara", "dan"],
    "substr(name, 0, 2)": [" A", "bo", "Bo", None, "ca", "da"],
    "concat(name, '-', to_text(qty))": [" Ann  Lee-1", "bob-2", "Bob-3", "-4", "cara-100", "dan-2"],   # missing: ""
    "coalesce(price, 0)": [1.5, 2.0, 0.0, 40.0, 3.0, 2.0],
    "is_missing(price)": [False, False, True, False, False, False],
    "if_(qty > 2, qty, 0)": [0, 0, 3, 4, 100, 0],
    "round(price / 3, 1)": [0.5, 0.7, None, 13.3, 1.0, 0.7],
    "abs(0 - qty)": [1, 2, 3, 4, 100, 2],
    "least(qty, 3)": [1, 2, 3, 3, 3, 2],
    "greatest(qty, 3)": [3, 3, 3, 4, 100, 3],
    "floor(price)": [1, 2, None, 40, 3, 2],
    "ceil(price)": [2, 2, None, 40, 3, 2],
    "sqrt(qty)": [1.0, 2 ** 0.5, 3 ** 0.5, 2.0, 10.0, 2 ** 0.5],
    "year(to_date(when))": [2024, 2024, None, 2024, None, 2024],
    "month(to_date(when))": [1, 2, None, 3, None, 12],
    "day(to_date(when))": [5, 10, None, 1, None, 31],
    "hour(to_date(when))": [10, 8, None, 0, None, 23],
    "minute(to_date(when))": [30, 0, None, 0, None, 59],
    "weekday(to_date(when))": [4, 5, None, 4, None, 1],
    "to_number('12') + qty": [13, 14, 15, 16, 112, 14],
    "qty > 2 and not price == 40": [False, False, True, False, True, False],
    "qty == 1 or qty == 100": [True, False, False, False, True, False],
}


def _same(a, b):
    if b is None:
        return a is None or (isinstance(a, float) and np.isnan(a)) or a is pd.NA or a is pd.NaT or pd.isna(a)
    if isinstance(b, float):
        return a is not None and not pd.isna(a) and abs(float(a) - b) < 1e-6
    return not pd.isna(a) and a == b


@pytest.mark.parametrize("text", list(FORMULAS))
def test_formula_functions(df, text):
    got = list(E.evaluate(text, df))
    want = FORMULAS[text]
    assert all(_same(a, b) for a, b in zip(got, want)), (text, got)
    spark = E.to_spark(text, list(df.columns), E.text_columns_of(df))
    ast.parse(spark)
    assert set(E.columns_used(text)) <= set(df.columns)


@pytest.mark.parametrize("bad", ["__import__('os')", "price.real", "open('x')", "lambda: 1", "[1, 2]", "price[0]",
                                 "nope + 1", "upper()", "", "price +"])
def test_formula_refuses_what_is_not_in_the_language(df, bad):
    with pytest.raises(E.ExprError):
        E.evaluate(bad, df)


def test_back_ticks_and_col():
    d = pd.DataFrame({"unit price": [2.0, 3.0], "n": [1, 2]})
    assert list(E.evaluate("`unit price` * n", d)) == [2.0, 6.0]
    assert list(E.evaluate('col("unit price") + 1', d)) == [3.0, 4.0]
    assert E.columns_used("`unit price` * n") == ["unit price", "n"]
    assert "unit price" in E.to_pandas("`unit price` * n", list(d.columns))
