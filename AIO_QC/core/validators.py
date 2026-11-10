# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Text validators for OCR Check.

A validator is a Python file with

    NAME = "Container ID (ISO 6346)"          # optional, shown in the UI
    DESCRIPTION = "4 letters + 7 digits …"    # optional
    def validate(text: str) -> bool:          # or -> (bool, "reason")
        ...

Validators receive text AFTER the class's normalisation (e.g. spaces / dashes removed, upper case), so they stay
simple. Three places, most specific wins on name clashes:
    built-in   AIO_QC/validators/*.py               (ISO 6346, Indian plate, VIN, EAN/UPC, ISBN, IBAN, dates …)
    user       <user data>/aio_qc/validators/*.py   (shared by all datasets)
    dataset    <dataset>/.aio_qc/validators/*.py

Ways to create one without writing code: pattern templates ("AA 00 AA 0000") and learning a pattern from examples.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from .editors import editors_home

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "validators"


def user_dir() -> Path:
    return editors_home().parent / "validators"


def dataset_dir(root: Path) -> Path:
    return Path(root) / ".aio_qc" / "validators"


# =============================================================================
# Loading & running
# =============================================================================
@dataclass
class Validator:
    name: str                 # file stem = id
    title: str
    description: str
    path: Path
    origin: str               # built-in | user | dataset
    fn: object = None
    error: str = ""

    def __call__(self, text: str) -> tuple[bool | None, str]:
        """(passed, reason). passed=None: the validator itself failed (never counted as a bad read)."""
        if self.fn is None:
            return None, self.error or "not loaded"
        try:
            r = self.fn(text)
        except Exception as exc:
            return None, f"validator error: {type(exc).__name__}: {exc}"
        if isinstance(r, tuple):
            return bool(r[0]), str(r[1]) if len(r) > 1 and r[1] else ""
        return bool(r), ""


def load_file(path: Path, origin: str) -> Validator:
    path = Path(path)
    v = Validator(path.stem, path.stem.replace("_", " "), "", path, origin)
    try:
        spec = importlib.util.spec_from_file_location(f"aioqc_validator_{origin}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "validate", None)
        if not callable(fn):
            raise ValueError("no validate(text) function")
        v.fn = fn
        v.title = str(getattr(mod, "NAME", v.title))
        v.description = str(getattr(mod, "DESCRIPTION", "") or (fn.__doc__ or "").strip())
    except Exception as exc:
        v.error = f"{type(exc).__name__}: {exc}"
    return v


def load_code(code: str, name: str = "draft") -> Validator:
    """Compile validator source from the editor (live testing) without saving it."""
    v = Validator(name, name, "", Path(f"<{name}>"), "draft")
    ns: dict = {"__name__": f"aioqc_validator_draft_{name}"}
    try:
        exec(compile(code, f"<validator {name}>", "exec"), ns)
        fn = ns.get("validate")
        if not callable(fn):
            raise ValueError("define a function validate(text) that returns True / False")
        v.fn, v.title = fn, str(ns.get("NAME", name))
        v.description = str(ns.get("DESCRIPTION", ""))
    except Exception as exc:
        v.error = f"{type(exc).__name__}: {exc}"
        tb = traceback.extract_tb(sys.exc_info()[2])
        line = next((f.lineno for f in reversed(tb) if f.filename.startswith("<validator")), None)
        if line:
            v.error += f" (line {line})"
    return v


def discover(root: Path | None = None) -> dict[str, Validator]:
    """name -> Validator; dataset overrides user overrides built-in."""
    out: dict[str, Validator] = {}
    places = [(BUILTIN_DIR, "built-in"), (user_dir(), "user")]
    if root is not None:
        places.append((dataset_dir(root), "dataset"))
    for d, origin in places:
        if d.is_dir():
            for p in sorted(d.glob("*.py")):
                if not p.name.startswith("_"):
                    out[p.stem] = load_file(p, origin)
    return out


def safe_name(title: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_")[:48] or "validator"


def save(name: str, code: str, where: str, root: Path | None = None) -> Path:
    d = dataset_dir(root) if where == "dataset" else user_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{safe_name(name)}.py"
    tmp = p.with_suffix(".py.tmp")
    tmp.write_text(code, encoding="utf-8")
    tmp.replace(p)
    return p


# =============================================================================
# Normalisation (applied before validators; per text class)
# =============================================================================
DEFAULT_NORMALIZE = {"upper": True, "remove": " -._", "strip": True}


def normalize(text: str, opts: dict | None = None) -> str:
    o = DEFAULT_NORMALIZE if opts is None else opts
    t = text or ""
    if o.get("strip", True):
        t = t.strip()
    if o.get("upper", True):
        t = t.upper()
    rm = o.get("remove", "")
    if rm:
        t = t.translate({ord(c): None for c in rm})
    return t


# look-alike characters for identity matching (leakage) — "0" and "O" are the same plate
LOOKALIKE = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})


def identity_key(text: str) -> str:
    """Letters + digits of any script, upper-cased, Latin look-alikes merged (O/0, I/1 …) — for leakage matching."""
    return re.sub(r"[\W_]+", "", (text or "").upper()).translate(LOOKALIKE)


# =============================================================================
# Pattern templates:  A letter · 0 digit · X letter-or-digit · * any · [..] set · ? optional · | alternatives
# =============================================================================
TEMPLATE_HELP = ("A = letter · 0 = digit · X = letter or digit · * = any character · [ABC] = one of · "
                 "? = previous is optional · A{1,2} = 1–2 letters · | = alternatives · anything else is literal. "
                 "Spaces are ignored (text is normalised first).")


def template_to_regex(template: str) -> str:
    alts = []
    for alt in template.split("|"):
        out, i, s = [], 0, alt.replace(" ", "")
        while i < len(s):
            ch = s[i]
            if ch == "[":
                j = s.find("]", i)
                if j < 0:
                    raise ValueError("unclosed [")
                out.append(s[i:j + 1])
                i = j + 1
                continue
            if ch == "{":
                j = s.find("}", i)
                if j < 0 or not out:
                    raise ValueError("{n,m} must follow a character")
                out[-1] += s[i:j + 1]
                i = j + 1
                continue
            if ch == "?":
                if not out:
                    raise ValueError("? must follow a character")
                out[-1] = f"(?:{out[-1]})?"
            elif ch == "A":
                out.append("[A-Z]")
            elif ch == "0":
                out.append(r"\d")
            elif ch == "X":
                out.append("[A-Z0-9]")
            elif ch == "*":
                out.append(".")
            else:
                out.append(re.escape(ch))
            i += 1
        alts.append("".join(out))
    rx = "^(?:" + "|".join(alts) + ")$"
    re.compile(rx)
    return rx


def _mask(ch: str) -> str:
    return "A" if ch.isalpha() else ("0" if ch.isdigit() else ch)


def _runs(text: str) -> list[tuple[str, int]]:
    runs: list[list] = []
    for ch in text:
        m = _mask(ch)
        if runs and runs[-1][0] == m and m in "A0":
            runs[-1][1] += 1
        else:
            runs.append([m, 1])
    return [(m, n) for m, n in runs]


def learn_template(examples: list[str], normalize_opts: dict | None = None) -> tuple[str, list[str]]:
    """
    Correct examples -> a pattern template. Examples with the same run structure (letters / digits / symbols in the
    same order) are merged with length ranges; different structures become alternatives (max 6).
    Returns (template, notes).
    """
    ex = [normalize(e, normalize_opts) for e in examples if e and e.strip()]
    if not ex:
        raise ValueError("enter at least one example")
    groups: dict[tuple, list[list[tuple[str, int]]]] = {}
    for e in ex:
        r = _runs(e)
        groups.setdefault(tuple(m for m, _n in r), []).append(r)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    notes = []
    if len(ordered) > 6:
        notes.append(f"{len(ordered)} different shapes — kept the 6 most common")
        ordered = ordered[:6]
    alts = []
    for struct, members in ordered:
        parts = []
        for k, m in enumerate(struct):
            lo = min(r[k][1] for r in members)
            hi = max(r[k][1] for r in members)
            if m in "A0":
                parts.append(m * lo if lo == hi and lo <= 4 else (f"{m}{{{lo}}}" if lo == hi else f"{m}{{{lo},{hi}}}"))
            else:
                parts.append(m)
        alts.append("".join(parts))
        if len(members) == 1 and len(ex) > 1:
            notes.append(f"'{''.join(p for p in parts)}' comes from a single example")
    return " | ".join(alts), notes


def template_code(name: str, template: str, description: str = "") -> str:
    rx = template_to_regex(template)
    return (f'NAME = {name!r}\nDESCRIPTION = {(description or "Pattern: " + template)!r}\n\nimport re\n\n'
            f"PATTERN = re.compile({rx!r})\n\n\n"
            "def validate(text: str) -> bool:\n"
            "    return bool(PATTERN.match(text))\n")


BLANK_TEMPLATE = '''NAME = "My validator"
DESCRIPTION = "What a valid text looks like"


def validate(text: str) -> bool:
    """text is already normalised (see the class settings). Return True if it is valid.
    You may also return (False, "reason") to explain a failure."""
    return len(text) > 0
'''
