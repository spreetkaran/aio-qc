# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Per-dataset QC state, stored in <root>/.aio_qc/ (hidden -> ignored by discovery):

    review.json      file review markers (ok / flag / fixed + note)
    classes_confirmed.json   the user confirmed this classes file for this YOLO dataset
    edit_sessions/<ts>/      external-editor sessions (copies the editor works on; see core.edit_session)
    label_map.json   view-time label mapping
    cache.sqlite     labels / image hashes / model results (core.cache)
    backups.sqlite   originals of every file AIO QC rewrote (core.backups; Tools ▸ Storage…)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .io import atomic_write_text

STATE_DIR = ".aio_qc"
REVIEW_OK, REVIEW_FLAG, REVIEW_FIXED = "ok", "flag", "fixed"
REVIEW_STATES = (REVIEW_OK, REVIEW_FLAG, REVIEW_FIXED)
# flag -> fixed (corrected in an editor, needs a look) -> ok (verified)


def state_dir(root: Path) -> Path:
    return Path(root) / STATE_DIR


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class ReviewStore:
    """Relative-path keyed so the dataset can be moved/shared."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = state_dir(root) / "review.json"
        self.data: dict[str, dict] = _read(self.path).get("files", {})
        self.dirty = False
        # append-only journal: a mark costs a few bytes, not a rewrite of the whole file (100 MB at 1M files)
        self.journal = self.path.with_suffix(".journal")
        self._pending: list[str] = []
        self._journal_lines = self._replay()
        # keys are absolute paths under the root (as discovered): the relative key is a string slice. The root is
        # resolved ONCE here — Path.resolve() per lookup cost ~70 µs (file-system calls) and ran for every row of the
        # file list, every badge repaint and every count.
        roots = {str(self.root), str(self.root.resolve())}
        self._prefixes = tuple(sorted({r.rstrip("/\\") + sep for r in roots for sep in ("/", "\\")}, key=len,
                                      reverse=True))

    def _rel(self, key: str) -> str:
        for pre in self._prefixes:
            if key.startswith(pre):
                return key[len(pre):].replace("\\", "/")
        try:                                              # outside the root (rare): the old, exact way
            return Path(key).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(key).as_posix()

    def get(self, key: str) -> dict | None:
        return self.data.get(self._rel(key))

    def status(self, key: str) -> str | None:
        e = self.get(key)
        return e["status"] if e else None

    def set(self, key: str, status: str | None, note: str | None = None):
        rel = self._rel(key)
        if status is None:
            self.data.pop(rel, None)
        else:
            entry = self.data.get(rel, {})
            entry.update(status=status, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
            if note is not None:
                if note:
                    entry["note"] = note
                else:
                    entry.pop("note", None)
            self.data[rel] = entry
        self.dirty = True
        self._pending.append(json.dumps({"k": rel, "e": self.data.get(rel)}, ensure_ascii=False))

    def counts(self, keys) -> dict[str, int]:
        out = {REVIEW_OK: 0, REVIEW_FLAG: 0, REVIEW_FIXED: 0}
        for k in keys:
            s = self.status(k)
            if s in out:
                out[s] += 1
        return out

    COMPACT_AFTER = 20000           # journal lines before the full file is rewritten

    def _replay(self) -> int:
        """Apply journal lines written after the last full save (crash-safe: a torn last line is skipped)."""
        n = 0
        try:
            with open(self.journal, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("e") is None:
                        self.data.pop(d["k"], None)
                    else:
                        self.data[d["k"]] = d["e"]
                    n += 1
        except OSError:
            pass
        return n

    def save(self, compact: bool = False):
        """Append the marks since the last save (O(changes)); rewrite the full file only now and then / on close."""
        if not self.dirty:
            return
        if not compact and self._journal_lines + len(self._pending) < self.COMPACT_AFTER:
            if self._pending:
                self.journal.parent.mkdir(parents=True, exist_ok=True)
                with open(self.journal, "a", encoding="utf-8") as f:
                    f.write("\n".join(self._pending) + "\n")
                    f.flush()
                self._journal_lines += len(self._pending)
                self._pending = []
            return
        atomic_write_text(self.path, json.dumps({"version": 1, "files": self.data}, ensure_ascii=False,
                                                separators=(",", ":")))
        try:
            self.journal.unlink()
        except OSError:
            pass
        self._pending, self._journal_lines = [], 0
        self.dirty = False


def load_label_map(root: Path) -> dict[str, str]:
    return {str(k): str(v) for k, v in _read(state_dir(root) / "label_map.json").get("mapping", {}).items()}


def save_label_map(root: Path, mapping: dict[str, str]):
    p = state_dir(root) / "label_map.json"
    if not mapping:
        if p.exists():
            p.unlink()
        return
    atomic_write_text(p, json.dumps({"version": 1, "mapping": mapping}, indent=2, ensure_ascii=False))


def samples_dir(root: Path) -> Path:
    return state_dir(root) / "samples"


def save_sample(root: Path, plan_json: dict) -> Path:
    import re
    name = re.sub(r"[^\w.-]+", "_", plan_json["name"]).strip("_") or "sample"
    p = samples_dir(root) / f"{name}.json"
    atomic_write_text(p, json.dumps(plan_json, indent=1, ensure_ascii=False))
    return p


def list_samples(root: Path) -> list[dict]:
    d = samples_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = _read(p)
        if data.get("keys"):
            data["_path"] = str(p)
            out.append(data)
    return out


def load_models_config(root: Path) -> dict:
    return _read(state_dir(root) / "models.json")


def save_models_config(root: Path, cfg: dict):
    atomic_write_text(state_dir(root) / "models.json", json.dumps(cfg, indent=2, ensure_ascii=False))


def audits_dir(root: Path) -> Path:
    return state_dir(root) / "audits"


def save_audit(root: Path, audit_json: dict) -> Path:
    import re
    name = re.sub(r"[^\w.-]+", "_", audit_json["name"]).strip("_") or "audit"
    p = audits_dir(root) / f"{name}.json"
    atomic_write_text(p, json.dumps(audit_json, indent=1, ensure_ascii=False))
    return p


def list_audits(root: Path) -> list[dict]:
    d = audits_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = _read(p)
        if data.get("items") is not None:
            data["_path"] = str(p)
            out.append(data)
    return out


# ---------------------------------------------------------------------------
# "Is this the final classes file?" — remembered per dataset, invalidated when the file changes
# ---------------------------------------------------------------------------
def _file_sig(path: Path) -> str:
    try:
        st = Path(path).stat()
        return f"{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return "missing"


def classes_confirmed(root: Path, source: Path | None, names: list[str]) -> bool:
    if source is None:
        return False
    d = _read(state_dir(root) / "classes_confirmed.json")
    return (d.get("source") == str(Path(source).resolve()) and d.get("sig") == _file_sig(source)
            and d.get("names") == list(names))


def confirm_classes(root: Path, source: Path, names: list[str]):
    atomic_write_text(state_dir(root) / "classes_confirmed.json", json.dumps(
        {"source": str(Path(source).resolve()), "sig": _file_sig(source), "names": list(names),
         "confirmed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
# OCR Check settings (per dataset): environment, engine, text classes + validators, thresholds
# ---------------------------------------------------------------------------
def load_ocr_config(root: Path) -> dict:
    return _read(state_dir(root) / "ocr.json")


def save_ocr_config(root: Path, cfg: dict):
    atomic_write_text(state_dir(root) / "ocr.json", json.dumps(cfg, indent=1, ensure_ascii=False))
