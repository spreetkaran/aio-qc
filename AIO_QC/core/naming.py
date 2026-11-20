# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One renaming engine for every place where file names can clash (Qt-free): Name Clashes (rename in the dataset),
Convert & Split flat, Combine & Flatten, Clean Export, hand-off batches, the training copy of Extend Model.

    items = [Item(key, rel="camA/day1/img001.jpg"), …]            one per image (its label files follow its name)
    sc = Scheme(preset="folder")                                  or "path" | "suffix" | "sequence" | "hash" | "custom"
    groups = clashes(items, flat=True)                            [[keys…]] that would end up with the same name
    P = plan(items, sc, flat=True, existing={dest: {names…}}, root_len=len(out))
    P.new[key]          the new stem                              (P.renamed: how many change)
    P.problems[key]     ["clashes with camA__img001 (2 files)", "too long for Windows (271 > 260)", …]
    P = resolve(P)      gives the still-clashing ones _1, _2 … and verifies again (until nothing clashes)
    write_csv(path, P)  original path → new name, one line per file

Names are compared the way Windows does — letter case ignored — and by STEM, not by stem + extension: a.jpg and
a.png in one folder would share a.json / a.txt. A plan is verified against every other planned name, against files
already in the destination that are not renamed, and against what Windows refuses (\\ / : * ? " < > |, control
characters, CON / PRN / AUX / NUL / COM1-9 / LPT1-9, a trailing dot or space) and the full path length (260 unless
long paths are on). One pass over plain dicts: millions of names in seconds; files are only read for {hash8}.

Pattern tokens: {stem} {ext} {folder} (the parent folder) {folder2} (grandparent + parent) {path} (every folder
below the root) {n} / {n:04d} (1, 2 … over the files renamed, in path order, from `start`) {hash8} (content) {date} (file date
YYYYMMDD) {split} {sep} (the separator).
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import string
import time
from dataclasses import dataclass, field

PRESETS = {
    "folder": ("Folder prefix", "{folder}{sep}{stem}", "camA__img001"),
    "path": ("Full path prefix", "{path}{sep}{stem}", "site1_camA_day3__img001"),
    "suffix": ("Number suffix (the first keeps its name)", "{stem}", "img001, img001_1, img001_2"),
    "sequence": ("Renumber in sequence", "{n:06d}", "000001, 000002"),
    "hash": ("Content hash", "{stem}_{hash8}", "img001_3f9a1c2e"),
    "custom": ("Custom pattern", "", ""),
}
TOKENS = ("stem", "ext", "folder", "folder2", "path", "n", "hash8", "date", "split", "sep")
WIN_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WIN_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
WIN_MAX_PATH = 260


class Item:
    """One file to name (an image — its label files follow its name). Its path is parsed once."""
    __slots__ = ("key", "rel", "split", "path", "extra_exts", "stem", "suffix", "folders", "parent_l", "nk")

    def __init__(self, key, rel, split="", path="", extra_exts=()):
        self.key, self.rel, self.split, self.path, self.extra_exts = key, rel.replace("\\", "/"), split, path, \
            tuple(extra_exts)
        head, _, name = self.rel.rpartition("/")
        dot = name.rfind(".")
        self.stem, self.suffix = (name[:dot], name[dot:]) if dot > 0 else (name, "")
        self.folders = [x for x in head.split("/") if x not in ("", ".")]
        self.parent_l = head.lower()
        self.nk = name_key(self.stem)


@dataclass
class Scheme:
    preset: str = "folder"
    pattern: str = ""            # custom
    sep: str = "__"
    case: str = "keep"           # keep | lower
    spaces: bool = True          # spaces → _
    max_len: int = 150           # of the new stem
    start: int = 1               # {n}
    only_clashes: bool = True    # leave names that do not clash alone
    windows: bool = True         # check what Windows refuses (always worth it: datasets travel)
    max_path: int = WIN_MAX_PATH

    def template(self) -> str:
        if self.preset == "custom":
            return self.pattern or "{stem}"
        return PRESETS.get(self.preset, PRESETS["folder"])[1]

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Scheme":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


@dataclass
class NamePlan:
    scheme: Scheme
    flat: bool
    items: list
    new: dict = field(default_factory=dict)          # key -> new stem (every item, changed or not)
    problems: dict = field(default_factory=dict)     # key -> [plain-word problems]
    renamed: int = 0
    took: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems

    def clash_count(self) -> int:
        return sum(1 for v in self.problems.values() if any(p.startswith("clashes") for p in v))

    def summary(self) -> str:
        n = len(self.items)
        parts = [f"{self.renamed:,} of {n:,} renamed"]
        if self.ok:
            parts.append("no clashes — ready")
        else:
            kinds = {}
            for v in self.problems.values():
                for p in v:
                    k = p.split(" (")[0].split(" with ")[0]
                    kinds[k] = kinds.get(k, 0) + 1
            parts += [f"{c:,} {k}" for k, c in sorted(kinds.items(), key=lambda kv: -kv[1])]
        return " · ".join(parts)

    def new_rel(self, key) -> str:
        """Where the file goes: <new stem><ext> (flat) or in its own folder (sub-folders kept)."""
        it = self._by_key()[key]
        name = self.new[key] + it.suffix
        return name if self.flat or not it.folders else "/".join(it.folders) + "/" + name

    def _by_key(self):
        if not hasattr(self, "_bk"):
            self._bk = {it.key: it for it in self.items}
        return self._bk


# ------------------------------------------------------------------ helpers
def dest_of(it, flat: bool) -> str:
    """The folder a name must be unique in: flat = one folder per split (all in one when `split` is "")."""
    return it.split.lower() if flat else it.parent_l


def name_key(stem: str) -> str:
    """How names compare: case ignored, as Windows does (and NFC: one spelling of é)."""
    if stem.isascii():
        return stem.lower()
    import unicodedata
    return unicodedata.normalize("NFC", stem).casefold()


def clashes(items, flat: bool = True) -> list[list[str]]:
    """Groups of keys whose files end up with the same name (stem, case ignored) in the same folder — today."""
    by = {}
    for it in items:
        by.setdefault((dest_of(it, flat), it.nk), []).append(it.key)
    return [v for v in by.values() if len(v) > 1]


_WS = re.compile(r"\s+")


def _clean(s: str, sc: Scheme) -> str:
    if not s.isalnum():
        s = WIN_BAD.sub("_", s)
        if sc.spaces:
            s = _WS.sub("_", s)
    if sc.case == "lower":
        s = s.lower()
    return s


class _Fmt(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str) and key not in kwargs:
            raise KeyError(key)
        return super().get_value(key, args, kwargs)


def bad_tokens(pattern: str) -> list[str]:
    """Tokens in a pattern that do not exist (shown as you type)."""
    out = []
    try:
        for _lit, name, _spec, _conv in string.Formatter().parse(pattern):
            if name is not None and name.split(".")[0].split("[")[0] not in TOKENS:
                out.append(name or "{}")
    except ValueError as exc:
        out.append(str(exc))
    return out


def file_hash8(path: str) -> str:
    h = hashlib.blake2b(digest_size=4)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fields(it, n: int, sc: Scheme, hash_fn) -> dict:
    folders = it.folders
    tmpl = sc.template()
    d = {"stem": it.stem, "ext": it.suffix.lstrip("."), "folder": folders[-1] if folders else "",
         "folder2": sc.sep.join(folders[-2:]) if folders else "", "path": "_".join(folders), "n": n,
         "split": it.split or "", "sep": sc.sep, "hash8": "", "date": ""}
    if "{hash8" in tmpl:
        d["hash8"] = hash_fn(it) if hash_fn else (file_hash8(it.path) if it.path else "")
    if "{date" in tmpl and it.path:
        try:
            d["date"] = time.strftime("%Y%m%d", time.localtime(os.path.getmtime(it.path)))
        except OSError:
            d["date"] = ""
    return d


_FAST = {"{folder}{sep}{stem}": lambda it, sc: (it.folders[-1] + sc.sep + it.stem) if it.folders else it.stem,
         "{path}{sep}{stem}": lambda it, sc: ("_".join(it.folders) + sc.sep + it.stem) if it.folders else it.stem}


def render(it, n: int, sc: Scheme, hash_fn=None) -> str:
    tmpl = sc.template()
    fast = _FAST.get(tmpl)
    if fast is not None:
        stem = fast(it, sc)
    elif tmpl == "{n:06d}":
        stem = f"{n:06d}"
    else:
        d = _fields(it, n, sc, hash_fn)
        stem = _Fmt().format(tmpl, **d)
        if not d["folder"] and sc.preset in ("folder", "path"):
            stem = d["stem"]
    if sc.preset in ("folder", "path"):
        stem = stem.strip(sc.sep) if sc.sep else stem
    stem = _clean(stem, sc)
    if sc.max_len and len(stem) > sc.max_len:          # keep the end (the original name) readable
        stem = stem[-sc.max_len:].lstrip("_")
    return stem


def _win_problems(stem: str, new_len: int, root_len: int, sc: Scheme, it) -> list[str]:
    if not stem:
        return ["empty name"]
    if not sc.windows:
        return []
    out = []
    if not stem.isalnum():
        if WIN_BAD.search(stem):
            out.append("characters Windows does not allow")
        if stem[-1] in ". ":
            out.append("ends in a dot or space (Windows drops it)")
    if len(stem) <= 4 and stem.split(".")[0].lower() in WIN_RESERVED:
        out.append(f"a name Windows reserves ({stem.split('.')[0].upper()})")
    if sc.max_path:
        longest = max([len(it.suffix)] + [len(e) for e in it.extra_exts])
        full = root_len + 1 + new_len + longest + (7 if it.extra_exts else 0)        # (+ "labels/")
        if full > sc.max_path:
            out.append(f"too long for Windows ({full} > {sc.max_path} characters)")
    return out


def plan(items, sc: Scheme, flat: bool = True, existing=None, root_len: int = 0, hash_fn=None,
         cancel=None, touch=None) -> NamePlan:
    """New names for `items` under `sc`, verified. existing: {dest folder (lower-case; flat: the split, "" without
    splits): {name keys of files already there that are NOT renamed}}. touch: the keys to rename when only clashing
    ones are (default: those clashes() finds) — every other item keeps its name but still counts in the check."""
    t0 = time.monotonic()
    items = sorted(items, key=lambda it: it.rel.lower())
    groups = clashes(items, flat)
    P = NamePlan(sc, flat, items)
    clash_keys = {k for g in groups for k in g} if touch is None else set(touch)
    seen_n = {}
    first_of = set()
    if sc.preset == "suffix":                          # the first of each clash keeps its name
        for g in groups:
            first_of.add(g[0])
    n = sc.start - 1                                   # {n} counts the files renamed, in path order
    for i, it in enumerate(items):
        if cancel is not None and i % 5000 == 0 and cancel():
            break
        rename = (not sc.only_clashes) or it.key in clash_keys
        if not rename or it.key in first_of:
            P.new[it.key] = it.stem
            continue
        n += 1
        if sc.preset == "suffix":
            base = it.nk, dest_of(it, flat)
            seen_n[base] = seen_n.get(base, 0) + 1
            P.new[it.key] = _clean(f"{it.stem}_{seen_n[base]}", sc)
        else:
            P.new[it.key] = render(it, n, sc, hash_fn)
    P.renamed = _count(P)
    verify(P, existing, root_len)
    P.took = time.monotonic() - t0
    return P


def _count(P) -> int:
    return sum(1 for it in P.items if P.new.get(it.key, it.stem) != it.stem or (P.flat and it.folders))


def verify(P: NamePlan, existing=None, root_len: int = 0) -> NamePlan:
    """Fill P.problems: clashes among the plan and with files already there, and what Windows refuses."""
    P.problems = {}
    existing = existing or {}
    groups = {}
    for it in P.items:
        stem = P.new.get(it.key, it.stem)
        groups.setdefault((dest_of(it, P.flat), name_key(stem)), []).append(it.key)
    for (dest, nk), keys in groups.items():
        taken = nk in existing.get(dest, ())
        if len(keys) > 1 or taken:
            what = f"clashes with {P.new[keys[0]]}" + (f" ({len(keys)} files)" if len(keys) > 1 else "") + \
                (" — a file already there" if taken else "")
            for k in keys:
                P.problems.setdefault(k, []).append(what)
    by = P._by_key()
    for k, stem in P.new.items():
        it = by[k]
        new_len = len(stem) + (0 if P.flat else len(it.parent_l) + 1 if it.folders else 0)
        msgs = _win_problems(stem, new_len, root_len, P.scheme, it)
        if msgs:
            P.problems.setdefault(k, []).extend(msgs)
    return P


def resolve(P: NamePlan, existing=None, root_len: int = 0, max_rounds: int = 6) -> NamePlan:
    """Give every still-clashing name a number (_1, _2 … the first in path order keeps its name, unless the name
    is taken by a file already there), shorten names that are too long, then verify again."""
    existing = existing or {}
    for _ in range(max_rounds):
        if P.ok:
            break
        groups = {}
        for it in P.items:
            groups.setdefault((dest_of(it, P.flat), name_key(P.new[it.key])), []).append(it.key)
        used = {g for g in groups}
        for (dest, nk), keys in groups.items():
            taken = nk in existing.get(dest, ())
            if len(keys) < 2 and not taken:
                continue
            start = 0 if taken else 1
            for k in keys[start:]:
                base, j = P.new[k], 1
                while (dest, name_key(f"{base}_{j}")) in used or name_key(f"{base}_{j}") in existing.get(dest, ()):
                    j += 1
                P.new[k] = f"{base}_{j}"
                used.add((dest, name_key(P.new[k])))
        for k, v in list(P.problems.items()):
            st = P.new[k]
            if any(p.startswith(("characters Windows", "ends in a dot")) for p in v):
                st = _clean(st, P.scheme).rstrip(". ") or "file"
            if any(p.startswith("a name Windows reserves") for p in v):
                st = f"{st}_file"
            if any(p.startswith("empty name") for p in v):
                st = "file"
            P.new[k] = st
            if any(p.startswith("too long") for p in v):
                over = max(int(re.search(r"\((\d+) >", p).group(1)) - P.scheme.max_path for p in v
                           if p.startswith("too long"))
                P.new[k] = P.new[k][over + 1:] or P.new[k]
        P.renamed = _count(P)
        verify(P, existing, root_len)
    return P


def write_csv(path, P: NamePlan, root: str = "") -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["original", "new"])
        for it in P.items:
            new = P.new_rel(it.key)
            if new != it.rel:
                w.writerow([os.path.join(root, it.rel) if root else it.rel, new])


def read_csv(path) -> dict:
    """renamed.csv → {new rel: original} (to trace a file back, or to merge a batch back)."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["new"]] = row["original"]
    return out
