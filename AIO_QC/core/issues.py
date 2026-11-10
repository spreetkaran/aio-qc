# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Issues Inbox: every QC tool publishes its findings here; files get a risk score and one review queue, worst first.

    store.publish("validation", "Validation", scanned_keys, {key: [Issue(...), …]})
    store.risk(key) -> 0..100     store.ranked(keys) -> [(key, risk, issues)], worst first

Risk: each finding weighs error 1.0 · warning 0.4 · info 0.08; one tool contributes at most 1.5 per file (a noisy tool
cannot dominate); risk = 100·(1 − e^(−total)). A rescan replaces the tool's findings for the files it scanned only.
Saved per dataset (.aio_qc/issues.json, keys relative to the root like review marks).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_text

WEIGHT = {"error": 1.0, "warn": 0.4, "info": 0.08}
SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}
TOOL_CAP = 1.5


@dataclass(slots=True)             # slots: ~2-3x less memory per row (scans of millions of files)
class Issue:
    tool: str
    severity: str          # error | warn | info
    text: str
    shape: int | None = None

    def to_json(self):
        return [self.severity, self.text, self.shape]


class IssueStore:
    def __init__(self, root: Path, state_dir: Path):
        self.root = Path(root)
        self.path = Path(state_dir) / "issues.json"
        self.by_tool: dict[str, dict[str, list[Issue]]] = {}
        self.titles: dict[str, str] = {}
        self.when: dict[str, str] = {}
        roots = {str(self.root), str(self.root.resolve())}
        self._prefixes = tuple(sorted({r.rstrip("/\\") + s for r in roots for s in ("/", "\\")}, key=len, reverse=True))
        self._risk: dict[str, float] | None = None
        self._load()

    # ------------------------------------------------------------------ keys (relative, portable)
    def _rel(self, key: str) -> str:
        for pre in self._prefixes:
            if key.startswith(pre):
                return key[len(pre):].replace("\\", "/")
        return key.replace("\\", "/")

    def _abs(self, rel: str) -> str:
        return str(self.root / rel) if not Path(rel).is_absolute() else rel

    # ------------------------------------------------------------------ publish / query
    def publish(self, tool: str, title: str, scanned: set | None, issues: dict[str, list[Issue]]):
        cur = self.by_tool.setdefault(tool, {})
        if scanned is None:
            cur.clear()
        else:
            for k in scanned:
                cur.pop(k, None)
        for k, lst in issues.items():
            if lst:
                cur[k] = list(lst)
        self.titles[tool] = title
        self.when[tool] = time.strftime("%Y-%m-%d %H:%M")
        self._risk = None

    def clear_tool(self, tool: str):
        self.by_tool.pop(tool, None)
        self._risk = None

    def forget(self, keys):
        for d in self.by_tool.values():
            for k in keys:
                d.pop(k, None)
        self._risk = None

    def rename(self, old: str, new: str):
        for d in self.by_tool.values():
            if old in d:
                d[new] = d.pop(old)
        self._risk = None

    def file_issues(self, key: str) -> list[Issue]:
        out = [i for d in self.by_tool.values() for i in d.get(key, ())]
        out.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 3))
        return out

    def _compute(self):
        per: dict[str, float] = defaultdict(float)
        for d in self.by_tool.values():
            for k, lst in d.items():
                per[k] += min(TOOL_CAP, sum(WEIGHT.get(i.severity, 0.0) for i in lst))
        self._risk = {k: 100.0 * (1.0 - math.exp(-v)) for k, v in per.items()}

    def risk(self, key: str) -> float:
        if self._risk is None:
            self._compute()
        return self._risk.get(key, 0.0)

    def ranked(self, alive: set | None = None) -> list[tuple[str, float]]:
        if self._risk is None:
            self._compute()
        items = [(k, r) for k, r in self._risk.items() if alive is None or k in alive]
        items.sort(key=lambda kr: -kr[1])
        return items

    def summary(self, alive: set | None = None) -> dict:
        sev, tools, files = Counter(), Counter(), set()
        for t, d in self.by_tool.items():
            for k, lst in d.items():
                if alive is not None and k not in alive:
                    continue
                files.add(k)
                tools[t] += len(lst)
                sev.update(i.severity for i in lst)
        return {"files": len(files), "severity": sev, "tools": tools}

    # ------------------------------------------------------------------ persistence
    def save(self):
        data = {"version": 1, "titles": self.titles, "when": self.when,
                "tools": {t: {self._rel(k): [i.to_json() for i in lst] for k, lst in d.items()}
                          for t, d in self.by_tool.items()}}
        atomic_write_text(self.path, json.dumps(data, ensure_ascii=False))

    def _load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.titles, self.when = d.get("titles", {}), d.get("when", {})
        for t, files in d.get("tools", {}).items():
            self.by_tool[t] = {self._abs(rel): [Issue(t, s, x, sh) for s, x, sh in lst] for rel, lst in files.items()}
