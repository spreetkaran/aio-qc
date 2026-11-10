# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Text Labels (Qt-free): check the text STORED in your annotations (a LabelMe shape field such as `description`) —
no OCR model needed — and suggest corrections. Nothing here knows about any particular kind of text: the rules come
from the validators YOU choose (optional) and from what the open dataset itself shows.

Learned from the dataset, per class (Model):
    formats      the run structures most texts share ("4 letters · 7 digits", "3 digits · “-” · 2 letters" …) —
                 only with ≥ 20 texts, ≥ 10 different ones, and ≥ 80 % of them fitting a few common structures
                 (free text gets none)
    known texts  texts that occur several times (the same object in many images / frames)
    alphabet     the characters the class actually uses (any script)
    confusions   character pairs the dataset itself gets wrong: where the frames of one object disagree by a
                 substitution or two, the pair is learned (seen ≥ 2 times) and used like a look-alike

Per text shape -> findings:
    empty        a text class without any text
    rule         fails every chosen validator
    unusual      fits none of the formats learned for the class
    variant      a rare text that is a look-alike of a frequent one ("O" where the other 40 have "0")
    track        differs from what the other frames of the same object say (video / burst sequences)
    format       stored differently from its normalised form (spaces, lower case …) — optional

Corrections, cheapest first (a correction must pass the validators AND fit a learned format):
    1. what the other frames of the same object agree on
    2. a frequent known text that is a look-alike of this one
    3. one look-alike character (letter↔digit: O↔0, I↔1, B↔8, S↔5 … plus the learned confusions)
    4. a known text one or two characters away
    5. two look-alike characters
    6. any one character (from the class's alphabet) replaced, removed or added
"Sure" (applied in one click) = the frames' majority, the normalised form, or the ONLY candidate of levels 2–3;
everything else is listed for a person to pick. Results are cached per text.
"""
from __future__ import annotations

import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from .validators import normalize

TEXT_FIELDS = ("description", "text", "ocr", "value", "transcription")
ALPHABET = string.ascii_uppercase + string.digits

# letter <-> digit look-alikes (Latin script): a letter in a digit position, or the reverse, is the classic typing /
# OCR slip. Letter-for-letter swaps (M/N) are not included: they would "fix" a check digit by luck.
_GROUPS = ["0OQD", "1IL", "2Z", "5S", "6G", "8B", "7T", "4A"]
LOOKALIKE: dict[str, str] = defaultdict(str)
for _g in _GROUPS:
    for _c in _g:
        LOOKALIKE[_c] += "".join(x for x in _g if x.isdigit() != _c.isdigit() and x not in LOOKALIKE[_c])
CANON = str.maketrans({c: g[0] for g in _GROUPS for c in g[1:]})       # look-alikes folded onto one character

KINDS = {  # kind -> (title, tone)
    "empty": ("no text", "error"),
    "rule": ("rule failed", "error"),
    "unusual": ("unusual format", "warning"),
    "variant": ("look-alike of a frequent text", "warning"),
    "track": ("differs from other frames", "warning"),
    "format": ("not normalised", "muted"),
}
MAX_CANDIDATES = 6


@dataclass
class TextFinding:
    key: str
    idx: int                      # shape index in the file
    label: str
    field: str                    # the shape field that holds (or should hold) the text
    raw: str | None               # the stored value, exactly
    text: str                     # normalised
    kind: str
    detail: str
    suggestion: str | None = None
    sure: bool = False
    why: str = ""                 # how the suggestion was found
    candidates: list = field(default_factory=list)
    fp: str = ""                  # shape fingerprint (re-identified at apply time)


# ------------------------------------------------------------------------------------------------- reading
def read_text(shape: dict, fld: str = "") -> tuple[str, str | None]:
    """(field, value). fld="" = the first text-like field that holds a string."""
    if fld:
        v = shape.get(fld)
        return fld, v if isinstance(v, str) else None
    for k in TEXT_FIELDS:
        v = shape.get(k)
        if isinstance(v, str) and v.strip():
            return k, v
    return "", None


def passes(text: str, validators: list) -> tuple[bool, str]:
    """Any validator accepts -> (True, ""). No validators -> (True, ""). A crashing validator never counts."""
    if not validators:
        return True, ""
    reasons = []
    for v in validators:
        ok, why = v(text)
        if ok:
            return True, ""
        if ok is False:
            reasons.append(f"{v.title}: {why}" if why else v.title)
    return False, "; ".join(reasons) or "no validator could run"


# ------------------------------------------------------------------------------------------------- learning
def signature(t: str) -> tuple:
    """Run structure: letters / digits / other characters in order with their lengths — (('L', 4), ('D', 7)),
    (('D', 3), ('-', 1), ('L', 2)). Any script: str.isalpha / str.isdigit decide; other characters stay literal."""
    runs: list[list] = []
    for ch in t:
        k = "L" if ch.isalpha() else ("D" if ch.isdigit() else ch)
        if runs and runs[-1][0] == k and k in ("L", "D"):
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    return tuple((k, n) for k, n in runs)


def describe(sig: tuple) -> str:
    out = []
    for k, n in sig:
        if k in ("L", "D"):
            out.append(f"{n} {'letter' if k == 'L' else 'digit'}{'s' if n != 1 else ''}")
        else:
            out.append({" ": "space"}.get(k, f"“{k}”"))
    return " · ".join(out)


class Model:
    """What one class's texts look like in this dataset (see the module doc)."""

    def __init__(self, texts, min_n: int = 20, min_share: float = 0.05, coverage: float = 0.8,
                 known_min: int = 3):
        cnt = Counter(t for t in texts if t)
        self.counts, self.n = cnt, sum(cnt.values())
        sigs = Counter()
        for t, c in cnt.items():
            sigs[signature(t)] += c
        common = [(s, c) for s, c in sigs.most_common(6) if c >= 3 and c / max(1, self.n) >= min_share]
        cov = sum(c for _s, c in common) / max(1, self.n)
        # a format needs evidence: enough texts, enough DIFFERENT texts (40 copies of one text prove nothing),
        # and a few structures covering most of them (free text never gets one)
        self.formats = dict(common) if (self.n >= min_n and len(cnt) >= 10 and common and cov >= coverage) else {}
        self.coverage = cov if self.formats else 0.0
        self.alphabet = "".join(sorted({ch for t in cnt for ch in t}))
        self.known = {t: c for t, c in cnt.items() if c >= known_min}
        self._by_canon: dict[str, str] = {}
        for t, c in sorted(self.known.items(), key=lambda kv: -kv[1]):
            self._by_canon.setdefault(t.translate(CANON), t)
        self.pairs: dict[str, set] = defaultdict(set)
        self.n_pairs = 0

    def fits(self, t: str) -> bool:
        return not self.formats or signature(t) in self.formats

    def format_text(self) -> str:
        if not self.formats:
            return ""
        return " | ".join(describe(s) for s in self.formats) + f" ({self.coverage:.0%} of {self.n:,})"

    def frequent_lookalike(self, t: str) -> str | None:
        """A known text that differs from t only by look-alike characters (and is much more frequent)."""
        k = self._by_canon.get(t.translate(CANON))
        if k is None or k == t:
            return None
        return k if self.counts.get(k, 0) >= max(3, 5 * self.counts.get(t, 0)) else None

    def known_near(self, t: str, max_d: int) -> list[str]:
        """Known texts within max_d edits (length-filtered), most frequent first."""
        out = []
        for k, c in self.known.items():
            if k != t and abs(len(k) - len(t)) <= max_d and edit_distance(k, t, max_d) <= max_d:
                out.append((c, k))
        return [k for _c, k in sorted(out, reverse=True)]

    def learn_pairs(self, pairs) -> int:
        """(wrong text, right text) of the same object -> character confusions seen ≥ 2 times."""
        seen = Counter()
        for a, b in pairs:
            if len(a) == len(b):
                diff = [(x, y) for x, y in zip(a, b) if x != y]
                if 0 < len(diff) <= 2:
                    seen.update(diff)
        for (x, y), n in seen.items():
            if n >= 2 and y not in LOOKALIKE.get(x, ""):
                self.pairs[x].add(y)
        self.n_pairs = sum(len(v) for v in self.pairs.values())
        return self.n_pairs


# ------------------------------------------------------------------------------------------------- suggestions
class Suggester:
    """Cheapest corrections that make a text valid: passes the validators and fits the learned formats."""

    def __init__(self, validators: list, model: Model | None = None, max_len: int = 32):
        self.validators = validators
        self.model = model
        self.max_len = max_len
        self.alphabet = (model.alphabet if model is not None and model.alphabet else "") or ALPHABET
        self._cache: dict[str, tuple[list, str, bool]] = {}
        self._ok: dict[str, bool] = {}

    @property
    def has_rules(self) -> bool:
        return bool(self.validators) or bool(self.model is not None and self.model.formats)

    def ok(self, t: str) -> bool:
        r = self._ok.get(t)
        if r is None:
            r = self._ok[t] = passes(t, self.validators)[0] and (self.model is None or self.model.fits(t))
        return r

    def _alts(self, c: str) -> str:
        extra = "".join(sorted(self.model.pairs.get(c, ()))) if self.model is not None else ""
        return LOOKALIKE.get(c, "") + "".join(x for x in extra if x not in LOOKALIKE.get(c, ""))

    def suggest(self, text: str) -> tuple[list[str], str, bool]:
        """([candidates at the cheapest level], how, sure). [] when nothing cheap enough is valid."""
        if text in self._cache:
            return self._cache[text]
        res = self._find(text) if text and len(text) <= self.max_len else ([], "", False)
        self._cache[text] = res
        return res

    def _find(self, t: str) -> tuple[list[str], str, bool]:
        m = self.model
        if m is not None:                                                 # a frequent look-alike of this text
            k = m.frequent_lookalike(t)
            if k is not None and self.ok(k):
                return [k], f"look-alike of “{k}”, seen {m.counts[k]:,} times", True
        found = []
        for i, c in enumerate(t):                                         # one look-alike
            for a in self._alts(c):
                cand = t[:i] + a + t[i + 1:]
                if self.ok(cand):
                    found.append(cand)
        if found:
            found = _uniq(found)
            return found, "one look-alike character", len(found) == 1
        if m is not None:                                                 # known texts one or two edits away
            near = [k for k in m.known_near(t, 1 if len(t) < 10 else 2) if self.ok(k)]
            if near:
                return near, f"known text(s) of this class (“{near[0]}” seen {m.counts[near[0]]:,} times)", False
        spots = [i for i, c in enumerate(t) if self._alts(c)]
        if len(spots) <= 16:                                              # two look-alikes
            for i, j in combinations(spots, 2):
                for a in self._alts(t[i]):
                    for b in self._alts(t[j]):
                        cand = t[:i] + a + t[i + 1:j] + b + t[j + 1:]
                        if self.ok(cand):
                            found.append(cand)
                if len(found) > 50:
                    break
            if found:
                return _uniq(found), "two look-alike characters", False
        n = len(t)
        for i in range(n):                                                # any one edit
            for a in self.alphabet:
                if a != t[i]:
                    cand = t[:i] + a + t[i + 1:]
                    if self.ok(cand):
                        found.append(cand)
            cand = t[:i] + t[i + 1:]
            if cand and self.ok(cand):
                found.append(cand)
        if n < self.max_len:
            for i in range(n + 1):
                for a in self.alphabet:
                    cand = t[:i] + a + t[i:]
                    if self.ok(cand):
                        found.append(cand)
        return (_uniq(found), "one character replaced / removed / added", False) if found else ([], "", False)


def _uniq(xs):
    return list(dict.fromkeys(xs))


def edit_distance(a: str, b: str, cap: int | None = None) -> int:
    """Levenshtein distance; with `cap`, stops early once every path exceeds it (returns cap + 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------------------------------------- per shape
def check_text(key, idx, label, fld, raw, cls_cfg: dict, validators: list, sugg: Suggester, flag_format: bool,
               fp: str = "") -> TextFinding | None:
    """One stored text -> its finding (None = fine)."""
    if raw is None or not raw.strip():
        return TextFinding(key, idx, label, fld, raw, "", "empty", "the shape holds no text", fp=fp)
    t = normalize(raw, cls_cfg.get("normalize"))
    ok, why = passes(t, validators)
    m = sugg.model
    kind = None
    if not ok:
        kind, detail = "rule", why
    elif m is not None and not m.fits(t):
        kind = "unusual"
        detail = f"“{describe(signature(t))}” — this class: {m.format_text()}"
    elif m is not None and m.frequent_lookalike(t) is not None:
        k = m.frequent_lookalike(t)
        kind = "variant"
        detail = f"stored {m.counts.get(t, 0):,}×, “{k}” {m.counts[k]:,}×"
    if kind is not None:
        f = TextFinding(key, idx, label, fld, raw, t, kind, detail, fp=fp)
        cands, how, sure = sugg.suggest(t)
        if cands:
            f.suggestion, f.why, f.candidates, f.sure = cands[0], how, cands[:MAX_CANDIDATES], sure
        return f
    if flag_format and raw != t:
        return TextFinding(key, idx, label, fld, raw, t, "format", f"stored as “{raw}”", suggestion=t, sure=True,
                           why="normalised form", fp=fp)
    return None


def check_shape(key, idx, shape, cls_cfg, validators, sugg, flag_format, fp="") -> TextFinding | None:
    fld, raw = read_text(shape, cls_cfg.get("text_field", ""))
    return check_text(key, idx, str(shape.get("label", "")), fld or cls_cfg.get("text_field") or "description", raw,
                      cls_cfg, validators, sugg, flag_format, fp)


# ------------------------------------------------------------------------------------------------- across frames
def consensus(tracks: list[list[tuple]], ok_fn, min_votes: int = 2, min_share: float = 0.5) -> dict:
    """tracks: [[(key, idx, normalised text or ""), …] one entry per frame]; ok_fn(text, key, idx) -> valid?
    -> {(key, idx): (majority, votes, frames)} for the shapes that disagree with a clear, valid majority."""
    out = {}
    for tr in tracks:
        texts = [t for _k, _i, t in tr if t]
        if len(texts) < min_votes:
            continue
        cnt = Counter(texts)
        (major, votes), *rest = cnt.most_common(2)
        if votes < min_votes or votes / len(tr) < min_share or (rest and rest[0][1] == votes):
            continue
        k0, i0, _t = next(e for e in tr if e[2] == major)
        if not ok_fn(major, k0, i0):
            continue
        for k, i, t in tr:
            if t != major:
                out[(k, i)] = (major, votes, len(tr))
    return out


def merge_track(f: TextFinding, cons: tuple) -> TextFinding:
    """A shape that disagrees with its track: the majority becomes the (strongest) suggestion."""
    major, votes, n = cons
    d = edit_distance(f.text, major) if f.text else len(major)
    f.detail = (f"{votes} of {n} frames read “{major}”" + (f" ({d} character(s) apart)" if f.text else "")
                + (f" · {f.detail}" if f.detail and f.kind not in ("track", "format") else ""))
    if f.kind == "format":
        f.kind = "track"
    f.suggestion, f.sure, f.why = major, True, f"what {votes} of {n} frames say"
    f.candidates = [major] + [c for c in f.candidates if c != major][:MAX_CANDIDATES - 1]
    return f
