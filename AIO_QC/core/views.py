# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Saved views — named, explained file queries with live counts (Qt-free).

A view is a list of conditions joined by ALL (and) or ANY (or); each condition can be negated. Conditions are
evaluated on what AIO QC already keeps in memory — the label index, review marks, issues, file names / splits /
folders — so a view over a million files is counted in about a second and never re-reads the files. That is also
why counts stay live: they follow every edit, mark and check result.

Condition types (dict "t"):
    label        objects of a label: op ≥ / ≤ / = / between (n, n2)        "at least 3 'plate'"
    total        objects in the file: same ops                               "no objects at all" (= 0)
    labels_n     number of DIFFERENT labels in the file                     "more than 4 different labels"
    review       review state is one of: none / ok / flag / fixed
    note         review note contains text
    split        split is …               folder   folder path starts with …
    name         file name contains text (or matches a regular expression)
    issue        has findings from a tool (or any tool); optional minimum risk (Issues Inbox, 0–100)
    error        the annotation file could not be read
Views are stored in <root>/.aio_qc/views.json and can be exported / imported (JSON) for similar datasets: labels
are matched by name, and labels a dataset does not have are reported (the view still runs; such a condition simply
matches no objects).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .io import atomic_write_text

OPS = {"ge": "at least", "le": "at most", "eq": "exactly", "between": "between"}
REVIEW = {"none": "not reviewed", "ok": "OK", "flag": "flagged", "fixed": "fixed"}
TYPES = {
    "label": "Objects of a label", "total": "Objects in the file", "labels_n": "Different labels in the file",
    "review": "Review state", "note": "Review note contains", "split": "Split is", "folder": "Folder starts with",
    "name": "File name contains", "issue": "Has findings", "error": "Annotation unreadable",
    "keys": "Fixed list of files",
}


def snapshot_cond(session, keys) -> dict:
    """A condition that is exactly these files (root-relative, so the dataset can move)."""
    rv = session.review
    return {"t": "keys", "rels": sorted({rv._rel(k) for k in keys}), "when": time.strftime("%Y-%m-%d %H:%M")}


def new_view(name: str = "New view") -> dict:
    return {"id": uuid.uuid4().hex[:10], "name": name, "match": "all", "conds": [], "desc": "",
            "created": time.strftime("%Y-%m-%d %H:%M")}


def _cmp(v: int, c: dict) -> bool:
    op, n = c.get("op", "ge"), int(c.get("n", 0))
    if op == "ge":
        return v >= n
    if op == "le":
        return v <= n
    if op == "eq":
        return v == n
    return n <= v <= int(c.get("n2", n))


def _num_text(c: dict) -> str:
    op, n = c.get("op", "ge"), c.get("n", 0)
    return f"between {n} and {c.get('n2', n)}" if op == "between" else f"{OPS[op]} {n}"


def explain_cond(c: dict) -> str:
    t = c.get("t")
    neg = c.get("not", False)
    if t == "label" and neg and c.get("op") == "eq" and int(c.get("n", 0)) == 0:
        return f"at least one '{c.get('label')}'"                    # NOT none -> some
    if t == "review" and c.get("states") == ["none"]:
        return "already reviewed" if neg else "not reviewed yet"
    if t == "label":
        n, op = int(c.get("n", 0)), c.get("op", "ge")
        if op == "eq" and n == 0:
            s = f"no '{c.get('label')}'"
        elif op == "ge" and n == 1:
            s = f"at least one '{c.get('label')}'"
        else:
            s = f"{_num_text(c)} '{c.get('label')}'"
    elif t == "total":
        s = "no objects" if c.get("op") == "eq" and int(c.get("n", 0)) == 0 else f"{_num_text(c)} object(s) in total"
    elif t == "labels_n":
        s = f"{_num_text(c)} different label(s)"
    elif t == "review":
        s = "review state " + " or ".join(REVIEW.get(x, x) for x in c.get("states", [])) if c.get("states") \
            else "any review state"
    elif t == "note":
        s = f"review note contains “{c.get('text', '')}”"
    elif t == "split":
        s = f"in split '{c.get('value', '')}'"
    elif t == "folder":
        s = f"in folder '{c.get('value', '')}'"
    elif t == "name":
        s = (f"name matches /{c.get('text', '')}/" if c.get("regex") else f"name contains “{c.get('text', '')}”")
    elif t == "issue":
        tool = c.get("tool") or "any tool"
        s = f"findings from {tool}" + (f" with risk ≥ {c.get('risk')}" if c.get("risk") else "")
    elif t == "error":
        s = "an unreadable annotation file"
    elif t == "keys":
        s = f"one of the {len(c.get('rels') or []):,} files saved on {c.get('when', '?')} (fixed list)"
    else:
        s = f"unknown condition {t!r}"
    return ("NOT " + s) if neg else s


def explain(view: dict) -> str:
    conds = view.get("conds") or []
    if not conds:
        return "Every file (no conditions yet)."
    join = " AND " if view.get("match", "all") == "all" else " OR "
    return "Files with " + join.join(explain_cond(c) for c in conds) + "."


def validate(view: dict, labels: set, splits: set) -> list[str]:
    """Plain-language problems for this dataset (unknown labels / splits, bad regex)."""
    out = []
    for c in view.get("conds") or []:
        if c.get("t") == "label" and c.get("label") not in labels:
            out.append(f"label '{c.get('label')}' does not occur in this dataset")
        if c.get("t") == "split" and c.get("value") not in splits:
            out.append(f"split '{c.get('value')}' does not exist here")
        if c.get("t") == "name" and c.get("regex"):
            try:
                re.compile(c.get("text", ""))
            except re.error as exc:
                out.append(f"regular expression: {exc}")
    return out


def snapshot(session, samples=None) -> dict:
    """Everything a view needs, copied on the GUI thread (cheap) so evaluation can run in the background."""
    rv_status, rv_note = {}, {}
    rv, by_key = session.review, session.by_key
    for rel, e in list(rv.data.items()):
        for pre in rv._prefixes:
            k = pre + (rel if pre[-1] == "/" else rel.replace("/", pre[-1]))
            if k in by_key:
                rv_status[k] = e.get("status")
                if e.get("note"):
                    rv_note[k] = e["note"]
                break
    issues = getattr(session, "issues", None)
    tools: dict[str, set] = {}
    risk = {}
    if issues is not None:
        for t, d in list(issues.by_tool.items()):
            title = issues.titles.get(t, t) if hasattr(issues, "titles") else t
            for k in list(d):
                tools.setdefault(k, set()).update({t, title})
        try:
            risk = {k: issues.risk(k) for k in tools}
        except Exception:
            risk = {}
    return {"prefixes": tuple(rv._prefixes),
            "samples": samples if samples is not None else [(s.key, s.name, s.split, s.folder) for s in session.samples],
            "index": dict(session.index), "errors": set(session.index_errors), "status": rv_status,
            "note": rv_note, "tools": tools, "risk": risk}


def _match_cond(c, key, name, split, folder, counts, snap, rx) -> bool:
    t = c.get("t")
    if t == "label":
        r = _cmp(int((counts or {}).get(c.get("label"), 0)), c)
    elif t == "total":
        r = _cmp(sum((counts or {}).values()), c)
    elif t == "labels_n":
        r = _cmp(sum(1 for v in (counts or {}).values() if v), c)
    elif t == "review":
        st = snap["status"].get(key) or "none"
        r = st in (c.get("states") or [])
    elif t == "note":
        r = c.get("text", "").lower() in snap["note"].get(key, "").lower()
    elif t == "split":
        r = split == c.get("value", "")
    elif t == "folder":
        r = (folder or "").replace("\\", "/").startswith(c.get("value", "").replace("\\", "/"))
    elif t == "name":
        r = bool(rx.search(name)) if rx is not None else c.get("text", "").lower() in name.lower()
    elif t == "issue":
        ts = snap["tools"].get(key)
        r = bool(ts) and (not c.get("tool") or c.get("tool") in ts) and \
            snap["risk"].get(key, 0.0) >= float(c.get("risk") or 0)
    elif t == "error":
        r = key in snap["errors"]
    elif t == "keys":
        r = _rel(key, snap) in c.get("_set", ())
    else:
        r = False
    return (not r) if c.get("not") else r


def _rel(key: str, snap: dict) -> str:
    for pre in snap.get("prefixes", ()):
        if key.startswith(pre):
            return key[len(pre):].replace("\\", "/")
    return key


def evaluate(view: dict, snap: dict) -> list[str]:
    """Keys of the matching files, in dataset order."""
    conds = [dict(c, _set=set(c.get("rels") or [])) if c.get("t") == "keys" else c for c in view.get("conds") or []]
    rxs = []
    for c in conds:
        rx = None
        if c.get("t") == "name" and c.get("regex"):
            try:
                rx = re.compile(c.get("text", ""), re.I)
            except re.error:
                rx = re.compile(r"(?!)")                      # invalid pattern matches nothing
        rxs.append(rx)
    every = view.get("match", "all") == "all"
    idx = snap["index"]
    out = []
    for key, name, split, folder in snap["samples"]:
        counts = idx.get(key)
        if not conds:
            out.append(key)
            continue
        if every:
            ok = all(_match_cond(c, key, name, split, folder, counts, snap, rx) for c, rx in zip(conds, rxs))
        else:
            ok = any(_match_cond(c, key, name, split, folder, counts, snap, rx) for c, rx in zip(conds, rxs))
        if ok:
            out.append(key)
    return out


class ViewStore:
    def __init__(self, root):
        self.path = Path(root) / ".aio_qc" / "views.json"
        self.views: list[dict] = []
        try:
            self.views = json.loads(self.path.read_text(encoding="utf-8")).get("views", [])
        except (OSError, ValueError):
            pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps({"version": 1, "views": self.views}, ensure_ascii=False, indent=1))

    def get(self, vid):
        return next((v for v in self.views if v["id"] == vid), None)

    def export(self, ids, path):
        Path(path).write_text(json.dumps({"aioqc_views": 1, "views": [v for v in self.views if v["id"] in ids]},
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    def import_file(self, path) -> list[dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        added = []
        for v in data.get("views", []):
            v = dict(v)
            v["id"] = uuid.uuid4().hex[:10]
            v.setdefault("match", "all")
            v.setdefault("conds", [])
            self.views.append(v)
            added.append(v)
        self.save()
        return added
