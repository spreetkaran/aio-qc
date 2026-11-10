# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Reading and writing tables (Qt-free, pandas). Every reader also writes itself as code — pandas and PySpark — so an
exported recipe loads its data exactly the way the app did.

    source = {"path": ..., "format": "csv", "opts": {"sep": ",", "encoding": "utf-8", "sheet": ..., "table": ...}}
    detect(path) -> source with sniffed options
    read(source, max_rows=None) -> (DataFrame, info)          info: rows read, sampled?, total rows (when known)
    read_code(source, engine) -> [lines]  ·  write(df, path, fmt)  ·  write_code(path, fmt, engine) -> [lines]
"""
from __future__ import annotations

import csv
import os

FORMATS = {
    "csv": "CSV / text", "parquet": "Parquet", "feather": "Feather", "excel": "Excel", "json": "JSON (records)",
    "jsonl": "JSON lines", "sqlite": "SQLite table",
}
EXT = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".tab": "csv", ".dat": "csv", ".parquet": "parquet",
       ".pq": "parquet", ".feather": "feather", ".arrow": "feather", ".xlsx": "excel", ".xlsm": "excel",
       ".xls": "excel", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".db": "sqlite", ".sqlite": "sqlite",
       ".sqlite3": "sqlite"}
OPEN_FILTER = ("Tables (*.csv *.tsv *.txt *.tab *.dat *.parquet *.pq *.feather *.arrow *.xlsx *.xlsm *.xls *.json "
               "*.jsonl *.ndjson *.db *.sqlite *.sqlite3);;All files (*)")
WRITE_FORMATS = {"csv": ".csv", "parquet": ".parquet", "excel": ".xlsx", "json": ".json", "jsonl": ".jsonl"}
BIG_BYTES = 400 * 1024 ** 2          # files above this open as a sample (the full file is used on export / Spark)
SAMPLE_ROWS = 1_000_000


def table_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    out = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_") or "table"
    return out if not out[0].isdigit() else f"t_{out}"


# ------------------------------------------------------------------------------------------------- sniffing
def _encoding(head: bytes) -> str:
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for enc in ("utf-8", "cp1252"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError as exc:
            if enc == "utf-8" and exc.start > len(head) - 4:       # cut in the middle of a character
                return enc
    return "latin-1"


def sniff_csv(path) -> dict:
    with open(path, "rb") as f:
        head = f.read(256 * 1024)
    enc = _encoding(head)
    text = head.decode(enc, errors="replace")
    lines = text.splitlines()[:50]
    sep = ","
    try:
        sep = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sum(line.count(d) for line in lines) for d in (",", ";", "\t", "|")}
        sep = max(counts, key=counts.get) if any(counts.values()) else ","
    return {"sep": sep, "encoding": enc, "header": True}


def detect(path: str) -> dict:
    fmt = EXT.get(os.path.splitext(path)[1].lower(), "csv")
    opts = {}
    if fmt == "csv":
        opts = sniff_csv(path)
    elif fmt == "excel":
        sh = sheets(path)
        opts = {"sheet": sh[0] if sh else 0}
    elif fmt == "json":
        with open(path, "rb") as f:
            first = f.read(4096).lstrip()[:1]
        if first != b"[":
            fmt = "jsonl"
    elif fmt == "sqlite":
        t = sqlite_tables(path)
        opts = {"table": t[0] if t else ""}
    return {"path": os.path.abspath(path), "format": fmt, "opts": opts}


def sheets(path) -> list[str]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            return list(wb.sheetnames)
        finally:
            wb.close()
    except Exception:
        return []


def sqlite_tables(path) -> list[str]:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return [r[0] for r in con.execute("select name from sqlite_master where type in ('table', 'view') "
                                              "order by name")]
        finally:
            con.close()
    except Exception:
        return []


def count_lines(path, cancel=None) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                return -1
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            n += b.count(b"\n")
    return n


# ------------------------------------------------------------------------------------------------- read
def read(source: dict, max_rows: int | None = None):
    """-> (DataFrame, info). max_rows=None: a sample of the first SAMPLE_ROWS rows for big text files, else all;
    max_rows=0: everything."""
    import pandas as pd
    path, fmt, o = source["path"], source["format"], source.get("opts") or {}
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    info = {"sampled": False, "bytes": size}
    limit = max_rows or None
    if max_rows is None and fmt in ("csv", "jsonl") and size > BIG_BYTES:
        limit = SAMPLE_ROWS
    if fmt == "csv":
        df = pd.read_csv(path, sep=o.get("sep", ","), encoding=o.get("encoding", "utf-8"),
                         header=0 if o.get("header", True) else None, nrows=limit, low_memory=False,
                         on_bad_lines="warn", encoding_errors="replace")
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "feather":
        df = pd.read_feather(path)
    elif fmt == "excel":
        df = pd.read_excel(path, sheet_name=o.get("sheet", 0), nrows=limit)
    elif fmt == "json":
        df = pd.read_json(path)
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, nrows=limit)
    elif fmt == "sqlite":
        import sqlite3
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            q = f'select * from "{str(o.get("table", "")).replace(chr(34), chr(34) * 2)}"' + \
                (f" limit {int(limit)}" if limit else "")
            df = pd.read_sql_query(q, con)
        finally:
            con.close()
    else:
        raise ValueError(f"unknown format {fmt}")
    if o.get("header", True) is False:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c) for c in df.columns]
    if limit and len(df) >= limit:
        info["sampled"] = True
    info["rows"] = len(df)
    return df, info


def read_code(source: dict, engine: str, var: str = "df") -> list[str]:
    path, fmt, o = source["path"], source["format"], source.get("opts") or {}
    p = repr(path)
    if engine == "pandas":
        if fmt == "csv":
            return [f"{var} = pd.read_csv({p}, sep={o.get('sep', ',')!r}, encoding={o.get('encoding', 'utf-8')!r}, "
                    f"header={0 if o.get('header', True) else None}, low_memory=False)"]
        if fmt == "parquet":
            return [f"{var} = pd.read_parquet({p})"]
        if fmt == "feather":
            return [f"{var} = pd.read_feather({p})"]
        if fmt == "excel":
            return [f"{var} = pd.read_excel({p}, sheet_name={o.get('sheet', 0)!r})"]
        if fmt == "json":
            return [f"{var} = pd.read_json({p})"]
        if fmt == "jsonl":
            return [f"{var} = pd.read_json({p}, lines=True)"]
        if fmt == "sqlite":
            return ["import sqlite3", f"with sqlite3.connect({p}) as _con:",
                    f"    {var} = pd.read_sql_query({('select * from ' + chr(34) + str(o.get('table', '')) + chr(34))!r}, "
                    "_con)"]
    else:
        if fmt == "csv":
            enc = o.get("encoding", "utf-8").replace("utf-8-sig", "utf-8")
            return [f"{var} = spark.read.csv({p}, header={bool(o.get('header', True))}, inferSchema=True, "
                    f"sep={o.get('sep', ',')!r}, encoding={enc!r}, multiLine=True, escape='\"')"]
        if fmt == "parquet":
            return [f"{var} = spark.read.parquet({p})"]
        if fmt == "jsonl":
            return [f"{var} = spark.read.json({p})"]
        if fmt == "json":
            return [f"{var} = spark.read.option('multiLine', True).json({p})"]
        # Excel / Feather / SQLite: read with pandas on the driver, then hand to Spark
        pd_lines = read_code(source, "pandas", "_pdf")
        return ["import pandas as pd"] + pd_lines + [f"{var} = spark.createDataFrame(_pdf)"]
    raise ValueError(fmt)


# ------------------------------------------------------------------------------------------------- write
def write(df, path: str, fmt: str):
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "excel":
        if len(df) > 1_048_575:
            raise ValueError("Excel holds at most 1,048,575 rows — export as CSV or Parquet")
        df.to_excel(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", date_format="iso", force_ascii=False, indent=1)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, date_format="iso", force_ascii=False)
    else:
        raise ValueError(fmt)


def write_code(path: str, fmt: str, engine: str, var: str = "df") -> list[str]:
    p = repr(path)
    if engine == "pandas":
        return {"csv": [f"{var}.to_csv({p}, index=False)"], "parquet": [f"{var}.to_parquet({p}, index=False)"],
                "excel": [f"{var}.to_excel({p}, index=False)"],
                "json": [f"{var}.to_json({p}, orient='records', date_format='iso', indent=1)"],
                "jsonl": [f"{var}.to_json({p}, orient='records', lines=True, date_format='iso')"]}[fmt]
    # Spark writes a FOLDER of part files (one per partition); coalesce(1) makes it a single part file
    w = f"{var}.coalesce(1).write.mode('overwrite')"
    return {"csv": [f"{w}.option('header', True).csv({p})"], "parquet": [f"{var}.write.mode('overwrite').parquet({p})"],
            "json": [f"{w}.json({p})"], "jsonl": [f"{w}.json({p})"],
            "excel": [f"{var}.toPandas().to_excel({p}, index=False)   # Excel: collected to the driver"]}[fmt]


def missing_engine_for(fmt: str) -> str | None:
    """A package pandas needs for this format that is not installed (None = fine)."""
    import importlib.util
    need = {"parquet": ["pyarrow", "fastparquet"], "feather": ["pyarrow"], "excel": ["openpyxl"]}.get(fmt)
    if not need:
        return None
    return None if any(importlib.util.find_spec(n) for n in need) else need[0]
