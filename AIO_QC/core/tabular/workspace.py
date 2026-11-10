# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
A Data Studio workspace (Qt-free): tables, each = a source + a list of steps (the recipe). Results are computed on
demand and cached per step, so editing step 7 of 10 re-runs from step 6, not from the file. Saved as a small JSON
file (.aioqc-data.json); the data itself is never copied into it.

    ws = Workspace()
    t = ws.add_source("/data/sales.csv")            # sniffed: separator, encoding, sheet …
    t.steps.append({"op": "filter", "p": {...}, "on": True})
    df = ws.result("sales")                         # pandas DataFrame after every enabled step
    ws.code("pandas" | "pyspark", out_path, fmt)    # one runnable script for the whole recipe
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field

from . import io as TIO
from . import steps as S

CACHE_BUDGET = 1_500 * 1024 ** 2          # intermediate results kept in memory (bytes); older ones are recomputed


@dataclass
class Table:
    name: str
    source: dict                           # io source dict, or {"format": "memory"} for in-app tables
    steps: list = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "steps": self.steps, "note": self.note}


class Workspace:
    def __init__(self, full: bool = False):
        self.full = full                                 # True: read whole files (exports), never a sample
        self.tables: dict[str, Table] = {}
        self.path: str | None = None
        self._raw: dict[str, tuple] = {}                 # name -> (DataFrame, info) as loaded
        self._cache: dict[str, dict[int, object]] = {}   # name -> {step count: DataFrame}
        self._gen: dict[str, int] = {}                    # bumped on every change: a result computed for an older
        self._lock = threading.RLock()                    # recipe (in a background thread) is never cached
        self.dirty = False

    # ------------------------------------------------------------------ tables
    def unique_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.tables:
            name = f"{base}_{i}"
            i += 1
        return name

    def add_source(self, path: str, source: dict | None = None, name: str | None = None) -> Table:
        src = source or TIO.detect(path)
        t = Table(self.unique_name(name or TIO.table_name(path)), src)
        self.tables[t.name] = t
        self.dirty = True
        return t

    def add_frame(self, df, name: str, note: str = "") -> Table:
        """An in-app table (e.g. the image dataset's annotations): kept in memory; saved recipes reference it by
        name and ask for it again."""
        t = Table(self.unique_name(name), {"format": "memory", "path": "", "opts": {}}, note=note)
        self.tables[t.name] = t
        self._raw[t.name] = (df, {"rows": len(df), "sampled": False, "bytes": 0})
        self.dirty = True
        return t

    def remove(self, name: str):
        self.tables.pop(name, None)
        self._raw.pop(name, None)
        self._cache.pop(name, None)
        self.dirty = True

    def rename(self, old: str, new: str):
        if new in self.tables:
            raise ValueError(f"a table called {new!r} exists")
        t = self.tables.pop(old)
        t.name = new
        self.tables[new] = t
        for k in (self._raw, self._cache):
            if old in k:
                k[new] = k.pop(old)
        for other in self.tables.values():                 # joins that use it follow the new name
            for st in other.steps:
                if st["op"] == "join" and st["p"].get("table") == old:
                    st["p"]["table"] = new
        self.dirty = True

    # ------------------------------------------------------------------ data
    def raw(self, name: str):
        with self._lock:
            have = self._raw.get(name)
        if have is not None:
            return have
        if name not in self._raw:
            t = self.tables[name]
            if t.source.get("format") == "memory":
                raise ValueError(f"{name}: an in-app table that is not loaded — open it from its source again")
            self._raw[name] = TIO.read(t.source, 0 if self.full else None)
        return self._raw[name]

    def info(self, name: str) -> dict:
        return self.raw(name)[1]

    def invalidate(self, name: str, from_step: int = 0):
        """Steps from index `from_step` on changed: drop their cached results (and of tables that join this one)."""
        with self._lock:
            self._gen[name] = self._gen.get(name, 0) + 1
            c = self._cache.get(name, {})
            for k in [k for k in c if k > from_step]:
                del c[k]
            for other in self.tables.values():
                if other.name != name and any(st["op"] == "join" and st["p"].get("table") == name
                                              for st in other.steps):
                    self._gen[other.name] = self._gen.get(other.name, 0) + 1
                    self._cache.pop(other.name, None)
        self.dirty = True

    def reload(self, name: str):
        with self._lock:
            self._gen[name] = self._gen.get(name, 0) + 1
            self._raw.pop(name, None)
            self._cache.pop(name, None)

    def result(self, name: str, upto: int | None = None, _stack=()):
        """DataFrame of `name` after its first `upto` steps (default: all). Disabled steps are skipped."""
        if name in _stack:
            raise ValueError("tables join each other in a circle: " + " → ".join(_stack + (name,)))
        t = self.tables[name]
        with self._lock:
            gen = self._gen.get(name, 0)
            steps = [dict(st) for st in t.steps]                  # a snapshot: the UI may edit meanwhile
            n = len(steps) if upto is None else min(upto, len(steps))
            cache = self._cache.setdefault(name, {})
            if n in cache:
                return cache[n]
            start = max([k for k in cache if k <= n], default=0)
            df = cache[start] if start else None
        if df is None:
            df = self.raw(name)[0]
        for i in range(start, n):
            st = steps[i]
            if st.get("on", True):
                tables = {}
                if st["op"] == "join":
                    other = st["p"].get("table")
                    if other not in self.tables:
                        raise ValueError(f"step {i + 1}: no table {other!r}")
                    tables[other] = self.result(other, None, _stack + (name,))
                try:
                    df = S.apply(df, st, tables)
                except Exception as exc:
                    raise StepError(i, exc) from exc
            with self._lock:
                if self._gen.get(name, 0) == gen:
                    self._cache.setdefault(name, {})[i + 1] = df
        with self._lock:
            if self._gen.get(name, 0) == gen:
                self._cache.setdefault(name, {})[n] = df
                self._trim()
        return df

    def _trim(self):
        total, entries = 0, []
        for name, c in self._cache.items():
            for k, df in c.items():
                try:
                    b = int(df.memory_usage(deep=False).sum())
                except Exception:
                    b = 0
                entries.append((name, k, b))
                total += b
        if total <= CACHE_BUDGET:
            return
        for name, k, b in sorted(entries, key=lambda e: e[1]):          # keep the latest results longest
            t = self.tables.get(name)
            if t is not None and k == len(t.steps):
                continue
            self._cache[name].pop(k, None)
            total -= b
            if total <= CACHE_BUDGET:
                break

    def columns_at(self, name: str, upto: int):
        return list(self.result(name, upto).columns)

    # ------------------------------------------------------------------ save / open
    def to_json(self) -> dict:
        return {"app": "AIO QC Data Studio", "version": 1, "saved": time.strftime("%Y-%m-%d %H:%M"),
                "tables": [t.to_json() for t in self.tables.values()]}

    def save(self, path: str):
        from ..io import atomic_write_text
        atomic_write_text(path, json.dumps(self.to_json(), indent=1, ensure_ascii=False, default=str))
        self.path, self.dirty = path, False

    def full_copy(self) -> "Workspace":
        """The same recipe over the whole files (in-app tables are shared as they are)."""
        ws = Workspace(full=True)
        for name, t in self.tables.items():
            ws.tables[name] = Table(t.name, dict(t.source), [dict(st) for st in t.steps], t.note)
            if t.source.get("format") == "memory" and name in self._raw:
                ws._raw[name] = self._raw[name]
        return ws

    @classmethod
    def open(cls, path: str) -> "Workspace":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        ws = cls()
        for td in d.get("tables") or []:
            ws.tables[td["name"]] = Table(td["name"], td.get("source") or {}, list(td.get("steps") or []),
                                          td.get("note", ""))
        ws.path, ws.dirty = path, False
        return ws

    # ------------------------------------------------------------------ code
    def order(self, target: str) -> list[str]:
        """Tables needed to build `target`, dependencies first."""
        seen, out = set(), []

        def visit(n, stack=()):
            if n in seen:
                return
            if n in stack:
                raise ValueError("tables join each other in a circle")
            for st in self.tables[n].steps:
                if st.get("on", True) and st["op"] == "join" and st["p"].get("table") in self.tables:
                    visit(st["p"]["table"], stack + (n,))
            seen.add(n)
            out.append(n)
        visit(target)
        return out

    def code(self, target: str, engine: str, out_path: str | None = None, out_fmt: str = "parquet") -> str:
        """One runnable script that rebuilds `target` from the files (full data — no sampling)."""
        L = [f"# {target} — generated by AIO QC Data Studio ({time.strftime('%Y-%m-%d %H:%M')})",
             "# Reads the full source files (the app may have shown a sample) and runs every enabled step."]
        if engine == "pandas":
            L += [S.PRELUDE_PD.strip(), "", "tables = {}"]
        else:
            L += [S.PRELUDE_SP.strip(), "import argparse", "",
                  "ap = argparse.ArgumentParser()",
                  "ap.add_argument('--master', default=None, help='e.g. local[*], yarn, spark://host:7077')",
                  "ap.add_argument('--remote', default=None, help='Spark Connect, e.g. sc://host:15002')",
                  "args, _ = ap.parse_known_args()",
                  "b = SparkSession.builder.appName('AIO QC Data Studio')",
                  "if args.remote:\n    b = b.remote(args.remote)\nelif args.master:\n    b = b.master(args.master)",
                  "spark = b.getOrCreate()", "", "tables = {}"]
        for name in self.order(target):
            t = self.tables[name]
            L += ["", f"# ---- {name}"]
            if t.source.get("format") == "memory":
                L += [f"# {name} was made inside the app ({t.note or 'in-memory table'}): export it once and read it",
                      f"# here instead, e.g. df = pd.read_parquet('{name}.parquet')",
                      f"raise SystemExit({('the table ' + name + ' exists only inside the app — export it first')!r})"]
                continue
            L += TIO.read_code(t.source, engine)
            for i, st in enumerate(t.steps, 1):
                if not st.get("on", True):
                    L.append(f"# step {i} (off): {S.OPS[st['op']].describe(st['p'])}")
                    continue
                L.append(f"# {i}. {S.OPS[st['op']].describe(st['p'])}")
                L += S.code_lines(st, engine)
            L.append(f"tables[{name!r}] = df")
        L += ["", f"df = tables[{target!r}]"]
        if out_path:
            L += TIO.write_code(out_path, out_fmt, engine)
        elif engine == "pandas":
            L += ["print(df.shape)", "print(df.head())"]
        else:
            L += ["df.printSchema()", "df.show(20, truncate=False)"]
        return "\n".join(L) + "\n"


class StepError(Exception):
    def __init__(self, index: int, exc: Exception):
        super().__init__(f"step {index + 1}: {exc}")
        self.index, self.exc = index, exc


def default_workspace_path(folder: str, name: str = "recipe") -> str:
    return os.path.join(folder, f"{name}.aioqc-data.json")
