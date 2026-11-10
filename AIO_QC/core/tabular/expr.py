# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Formulas for "New column" (Qt-free): one small, safe expression language compiled to pandas AND PySpark code, so a
formula means the same thing in the app, in exported pandas code and on Spark.

    price * qty                      arithmetic: + - * / // % ** (// floors, % is always ≥ 0 for a positive divisor)
    `unit price` * 1.18              columns with spaces / symbols in back-ticks (or col("unit price"))
    a > 10 and not b == "x"          comparisons, and / or / not
    "big" if amount > 1000 else "small"
    upper(name) + "-" + to_text(id)  text: upper lower strip len contains startswith endswith replace substr concat
    year(date)  month(date)  day(date)  weekday(date) (Mon = 0)  hour(date)  minute(date)  to_date(text)
    abs round(x, n) floor ceil sqrt log log10 exp  least(a, b, …) greatest(a, b, …)
    to_number(x)  to_text(x)  coalesce(a, b, …)  is_missing(x)  if_(cond, a, b)

Only these names exist: no attribute access, no imports, no other calls — the text is parsed (ast) and translated,
never executed as written. All literals are emitted with repr(), so names and values cannot inject code.
"""
from __future__ import annotations

import ast
import re

FUNCS = {  # name: (min args, max args, help)
    "abs": (1, 1, "absolute value"), "round": (1, 2, "round(x, digits)"), "floor": (1, 1, ""), "ceil": (1, 1, ""),
    "sqrt": (1, 1, ""), "log": (1, 1, "natural log"), "log10": (1, 1, ""), "exp": (1, 1, ""),
    "upper": (1, 1, ""), "lower": (1, 1, ""), "strip": (1, 1, "trim spaces"), "len": (1, 1, "text length"),
    "contains": (2, 2, "contains(text, part)"), "startswith": (2, 2, ""), "endswith": (2, 2, ""),
    "replace": (3, 3, "replace(text, old, new) — plain text"), "substr": (3, 3, "substr(text, start (1-based), length)"),
    "concat": (1, 20, "joins texts; missing parts are skipped"),
    "year": (1, 1, ""), "month": (1, 1, ""), "day": (1, 1, ""), "weekday": (1, 1, "Monday = 0"), "hour": (1, 1, ""),
    "minute": (1, 1, ""), "to_date": (1, 1, "text -> date / time"),
    "to_number": (1, 1, "text -> number (invalid -> missing)"), "to_text": (1, 1, ""),
    "coalesce": (2, 20, "first value that is not missing"), "is_missing": (1, 1, ""),
    "least": (2, 20, "smallest"), "greatest": (2, 20, "largest"), "if_": (3, 3, "if_(condition, then, else)"),
    "col": (1, 1, 'a column by name: col("unit price")'),
}
_BACKTICK = re.compile(r"`([^`]*)`")


class ExprError(ValueError):
    pass


def parse(text: str) -> ast.AST:
    src = _BACKTICK.sub(lambda m: f"col({m.group(1)!r})", text.strip())
    if not src:
        raise ExprError("empty formula")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"not a valid formula ({exc.msg})") from None
    return tree.body


def columns_used(text: str) -> list[str]:
    out = []

    def walk(n):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "col" and n.args \
                and isinstance(n.args[0], ast.Constant):
            out.append(str(n.args[0].value))
            return
        for c in ast.iter_child_nodes(n):
            walk(c)
    walk(parse(text))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------------------------- compiler
class _Compiler:
    BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}

    TEXT_FUNCS = {"upper", "lower", "strip", "replace", "substr", "concat", "to_text"}

    def __init__(self, columns=None, text_columns=()):
        self.columns = set(columns) if columns is not None else None
        self.text_columns = set(text_columns)

    def is_text(self, n) -> bool:
        """Does this sub-expression produce text? (decides whether + concatenates)"""
        if isinstance(n, ast.Constant):
            return isinstance(n.value, str)
        if isinstance(n, ast.Name):
            return n.id in self.text_columns
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == "col" and n.args and isinstance(n.args[0], ast.Constant):
                return n.args[0].value in self.text_columns
            if n.func.id in self.TEXT_FUNCS:
                return True
            if n.func.id in ("coalesce", "if_"):
                return any(self.is_text(a) for a in n.args[1 if n.func.id == "if_" else 0:])
            return False
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return self.is_text(n.left) or self.is_text(n.right)
        if isinstance(n, ast.IfExp):
            return self.is_text(n.body) or self.is_text(n.orelse)
        return False

    def column(self, name: str) -> str:
        if self.columns is not None and name not in self.columns:
            raise ExprError(f"no column “{name}”")
        return self.col(name)

    def node(self, n) -> str:
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, str, bool)) or n.value is None:
                return self.lit(n.value)
            raise ExprError("unsupported constant")
        if isinstance(n, ast.Name):
            if n.id in ("True", "False", "None"):
                return self.lit({"True": True, "False": False, "None": None}[n.id])
            return self.column(n.id)
        if isinstance(n, ast.BinOp) and isinstance(n.op, self.BIN):
            if isinstance(n.op, ast.Add) and (self.is_text(n.left) or self.is_text(n.right)):
                return self.text_add(self.node(n.left), self.node(n.right))
            return self.binop(type(n.op), self.node(n.left), self.node(n.right))
        if isinstance(n, ast.UnaryOp):
            v = self.node(n.operand)
            if isinstance(n.op, ast.USub):
                return f"(-{v})"
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return self.not_(v)
        if isinstance(n, ast.BoolOp):
            parts = [self.node(v) for v in n.values]
            return "(" + (" & " if isinstance(n.op, ast.And) else " | ").join(f"({p})" for p in parts) + ")"
        if isinstance(n, ast.Compare):
            parts, left = [], self.node(n.left)
            for op, right_n in zip(n.ops, n.comparators):
                if type(op) not in self.CMP:
                    raise ExprError("only == != < <= > >= comparisons")
                right = self.node(right_n)
                parts.append(f"({left} {self.CMP[type(op)]} {right})")
                left = right
            return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"
        if isinstance(n, ast.IfExp):
            return self.if_(self.node(n.test), self.node(n.body), self.node(n.orelse))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            name = n.func.id
            if name not in FUNCS:
                raise ExprError(f"unknown function “{name}” — available: {', '.join(sorted(FUNCS))}")
            lo, hi, _h = FUNCS[name]
            if not lo <= len(n.args) <= hi:
                raise ExprError(f"{name}() takes {lo}" + (f"–{hi}" if hi != lo else "") + " argument(s)")
            if name == "col":
                a = n.args[0]
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                    raise ExprError('col() needs a column name in quotes: col("unit price")')
                return self.column(a.value)
            return self.call(name, n.args)
        raise ExprError(f"not allowed in a formula: {type(n).__name__}")

    def const(self, n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            return -n.operand.value
        raise ExprError("this argument must be a plain value")


class PandasCompiler(_Compiler):
    """-> a pandas expression over `df` (needs `pd`, `np` in scope)."""

    def col(self, name):
        return f"df[{name!r}]"

    def lit(self, v):
        return repr(v) if v is not None else "np.nan"

    def binop(self, op, a, b):
        sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
               ast.Pow: "**"}[op]
        return f"({a} {sym} {b})"

    def not_(self, v):
        return f"(~({v}).astype('bool'))"

    def text_add(self, a, b):
        return f"({self.s(a)} + {self.s(b)})"

    def if_(self, c, a, b):
        return f"pd.Series(np.where({c}, {a}, {b}), index=df.index)"

    def s(self, a):
        return f"pd.Series({a}, index=df.index).astype('string')"

    def call(self, name, args):
        x = lambda i: self.node(args[i])
        if name in ("abs", "floor", "ceil", "sqrt", "log", "log10", "exp"):
            return f"np.{name}({x(0)})"
        if name == "round":
            return f"np.round({x(0)}, {int(self.const(args[1])) if len(args) > 1 else 0})"
        if name in ("upper", "lower", "strip", "len"):
            return f"{self.s(x(0))}.str.{name}()"
        if name in ("contains", "startswith", "endswith"):
            part = self.const(args[1])
            if name == "contains":
                return f"{self.s(x(0))}.str.contains({str(part)!r}, regex=False).fillna(False).astype('bool')"
            return f"{self.s(x(0))}.str.{name}({str(part)!r}).fillna(False).astype('bool')"
        if name == "replace":
            return f"{self.s(x(0))}.str.replace({str(self.const(args[1]))!r}, {str(self.const(args[2]))!r}, regex=False)"
        if name == "substr":
            st, ln = int(self.const(args[1])), int(self.const(args[2]))
            return f"{self.s(x(0))}.str.slice({max(0, st - 1)}, {max(0, st - 1) + ln})"
        if name == "concat":
            return "(" + " + ".join(f"{self.s(x(i))}.fillna('')" for i in range(len(args))) + ")"
        if name in ("year", "month", "day", "weekday", "hour", "minute"):
            return f"pd.to_datetime({x(0)}, errors='coerce', format='mixed').dt.{name}.astype('Int64')"
        if name == "to_date":
            return f"pd.to_datetime({x(0)}, errors='coerce', format='mixed')"
        if name == "to_number":
            return f"pd.to_numeric({x(0)}, errors='coerce')"
        if name == "to_text":
            return self.s(x(0))
        if name == "coalesce":
            out = f"pd.Series({x(0)}, index=df.index)"
            for i in range(1, len(args)):
                out = f"{out}.fillna({x(i)})"
            return out
        if name == "is_missing":
            return f"pd.Series({x(0)}, index=df.index).isna()"
        if name in ("least", "greatest"):
            cols = ", ".join(f"pd.Series({x(i)}, index=df.index)" for i in range(len(args)))
            return f"pd.concat([{cols}], axis=1).{'min' if name == 'least' else 'max'}(axis=1)"
        if name == "if_":
            return self.if_(x(0), x(1), x(2))
        raise ExprError(name)


class SparkCompiler(_Compiler):
    """-> a PySpark Column expression (needs `F` = pyspark.sql.functions in scope)."""

    def col(self, name):
        return f"F.col({name!r})"

    def lit(self, v):
        return f"F.lit({v!r})"

    def binop(self, op, a, b):
        if op is ast.FloorDiv:
            return f"F.floor({a} / {b})"
        if op is ast.Mod:
            return f"F.pmod({a}, {b})"
        if op is ast.Pow:
            return f"F.pow({a}, {b})"
        sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[op]
        return f"({a} {sym} {b})"

    def not_(self, v):
        return f"(~({v}))"

    def text_add(self, a, b):                    # text + text: joined (missing if either part is missing, as pandas)
        return f"F.concat(({a}).cast('string'), ({b}).cast('string'))"

    def if_(self, c, a, b):
        return f"F.when({c}, {a}).otherwise({b})"

    def call(self, name, args):
        x = lambda i: self.node(args[i])
        simple = {"abs": "abs", "floor": "floor", "ceil": "ceil", "sqrt": "sqrt", "log": "log", "log10": "log10",
                  "exp": "exp", "upper": "upper", "lower": "lower", "strip": "trim", "len": "length",
                  "year": "year", "month": "month", "day": "dayofmonth", "hour": "hour", "minute": "minute"}
        if name in simple:
            return f"F.{simple[name]}({x(0)})"
        if name == "round":
            return f"F.round({x(0)}, {int(self.const(args[1])) if len(args) > 1 else 0})"
        if name == "weekday":                                         # Spark: Sunday = 1 … Saturday = 7
            return f"F.pmod(F.dayofweek({x(0)}) + 5, F.lit(7))"
        if name in ("contains", "startswith", "endswith"):
            return f"F.coalesce(({x(0)}).cast('string').{name}({str(self.const(args[1]))!r}), F.lit(False))"
        if name == "replace":
            return (f"F.regexp_replace(({x(0)}).cast('string'), {re.escape(str(self.const(args[1])))!r}, "
                    f"{str(self.const(args[2])).replace(chr(92), chr(92) * 2).replace('$', chr(92) + '$')!r})")
        if name == "substr":
            return f"F.substring(({x(0)}).cast('string'), {int(self.const(args[1]))}, {int(self.const(args[2]))})"
        if name == "concat":
            return "F.concat_ws('', " + ", ".join(f"({x(i)}).cast('string')" for i in range(len(args))) + ")"
        if name == "to_date":
            return f"F.to_timestamp({x(0)})"
        if name == "to_number":
            return f"({x(0)}).cast('double')"
        if name == "to_text":
            return f"({x(0)}).cast('string')"
        if name == "coalesce":
            return "F.coalesce(" + ", ".join(x(i) for i in range(len(args))) + ")"
        if name == "is_missing":
            return f"({x(0)}).isNull()"
        if name in ("least", "greatest"):
            return f"F.{name}(" + ", ".join(x(i) for i in range(len(args))) + ")"
        if name == "if_":
            return self.if_(x(0), x(1), x(2))
        raise ExprError(name)


def text_columns_of(df) -> list[str]:
    """Columns that hold text (object / string dtype) — so `+` joins them in every engine."""
    return [c for c in df.columns if str(df[c].dtype) in ("object", "string", "str") or str(df[c].dtype).startswith("string")]


def to_pandas(text: str, columns=None, text_columns=()) -> str:
    return PandasCompiler(columns, text_columns).node(parse(text))


def to_spark(text: str, columns=None, text_columns=()) -> str:
    return SparkCompiler(columns, text_columns).node(parse(text))


def evaluate(text: str, df):
    """Evaluate a formula on a pandas DataFrame (the generated code, with only df / pd / np in scope)."""
    import numpy as np
    import pandas as pd
    code = to_pandas(text, list(df.columns), text_columns_of(df))
    return eval(compile(code, "<formula>", "eval"), {"__builtins__": {}, "pd": pd, "np": np, "df": df})
