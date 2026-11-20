# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Provenance — where every annotation, dataset, image and model the app writes came from (Qt-free).

No person is ever recorded: only what made a thing (a manual edit, a model, an import, a conversion, a rule …), with
which feature, model and app version, and when. Settings that look like secrets (keys, tokens, passwords) are never
written; folders are named, never given as full paths (those carry user names).

Annotations (LabelMe JSON) — per shape and per file, under the key "provenance" (LabelMe keeps unknown keys):

    shape["provenance"] = {"id": "3f9a0c1e7b2d",
                           "created": {"at": "2026-11-20T04:15:02Z", "by": "model", "tool": "Model Assist",
                                       "model": "yolo11n.pt", "model_sha256": "ab12…", "score": 0.91,
                                       "software": "AIO QC 3.0.0"},
                           "changes": [{"at": …, "by": "manual", "tool": "Image QC", "what": ["geometry"]}]}
    data["provenance"]  = {"format": "aio-qc-provenance/1", "created": {…}, "modified": {… "added": 1,
                           "changed": 2, "removed": 0}, "source": {… e.g. the video and frame}}

    by: manual · model · import · conversion · rule · augmentation · unknown (made before provenance was recorded)

  stamp_labelme(data, path) does it all before a write: it reads what the file held, keeps the record of every
  untouched shape, adds a change to every changed one (what changed: label, geometry, type, group, description,
  flags), a "created" record to every new one, and the file's created / modified. The origin comes from the
  feature: `with origin("model", "Auto-Annotate", model_path=…):` (a context variable — threads get it passed).
  A shape that already has a record when it arrives (a model detection with its score, a copy from another file)
  keeps it.

  YOLO label files cannot hold more than geometry: each save of a YOLO dataset adds a line to
  <dataset>/.aio_qc_label_edits.jsonl (file, origin, objects added / changed / removed). COCO annotations carry the
  shape record as "provenance".

Datasets — <folder>/.aio_qc_provenance.json, a list of records (each run that wrote or changed the folder adds one):
  {"id", "at", "software", "action", "tool", "by", "inputs": [{"name", "fingerprint", "provenance"}],
   "settings", "counts", "output": fingerprint}. A fingerprint is the file count, total size and a SHA-256 of the
  sorted (relative path, size) listing — the same dataset copied elsewhere has the same one. "provenance" of an
  input is the id of its own last record: the chain from raw data to a trained model can be followed.

Images the app makes (frames, crops, masks, augmented images, screenshots, re-encoded images) — embedded in the
file without re-encoding: XMP (xmp:CreatorTool, xmp:CreateDate and the record as aioqc:Record) in a JPEG APP1
segment / PNG iTXt chunk, plus EXIF Software in JPEGs and a PNG Software text chunk. read_image(path) reads it back.

Models — the training run folder gets .aio_qc_provenance.json (training data fingerprint and its provenance id, classes,
settings, the model it started from and its SHA-256, the weights' SHA-256); ONNX files carry it as metadata
"provenance", AIO QC classifier checkpoints as the key "provenance".

AIOQC_NO_PROVENANCE=1 (or set_enabled(False), Settings ▸ General) writes none of it.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "aio-qc-provenance/1"
KEY = "provenance"
SIDECAR = ".aio_qc_provenance.json"           # hidden: no dataset reader takes it for a label file
YOLO_LOG = ".aio_qc_label_edits.jsonl"
NS = "urn:aio-qc:provenance:1"
MAX_CHANGES = 20
BY = ("manual", "model", "import", "conversion", "rule", "augmentation", "unknown")

_enabled = os.environ.get("AIOQC_NO_PROVENANCE", "").strip() not in ("1", "true", "yes")


def enabled() -> bool:
    return _enabled


def set_enabled(on: bool):
    """On / off for this process and the worker processes it starts from now on (they inherit the environment)."""
    global _enabled
    _enabled = bool(on)
    if _enabled:
        os.environ.pop("AIOQC_NO_PROVENANCE", None)
    else:
        os.environ["AIOQC_NO_PROVENANCE"] = "1"


def software() -> str:
    from AIO_QC import __version__
    return f"AIO QC {__version__}"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ====================================================================== origin
_SHA_CACHE: dict = {}
_SHA_LOCK = threading.Lock()


def file_sha256(path, short: int = 16) -> str:
    """SHA-256 of a file (weights), remembered by path, size and modification time."""
    try:
        p = Path(path)
        st = p.stat()
    except (OSError, TypeError):
        return ""
    k = (str(p), st.st_size, st.st_mtime_ns)
    with _SHA_LOCK:
        if k in _SHA_CACHE:
            return _SHA_CACHE[k][:short] if short else _SHA_CACHE[k]
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    with _SHA_LOCK:
        _SHA_CACHE[k] = h.hexdigest()
    return h.hexdigest()[:short] if short else h.hexdigest()


@dataclass(frozen=True)
class Origin:
    by: str = "manual"
    tool: str = "AIO QC"
    model: str = ""                       # weights file name or model id — never a folder
    model_sha256: str = ""
    detail: str = ""
    extra: dict = field(default_factory=dict, hash=False, compare=False)
    timeless: bool = False                # no time: outputs that must come out byte for byte the same (augmentation)

    def entry(self, **more) -> dict:
        e = {} if self.timeless else {"at": now()}
        e.update({"by": self.by, "tool": self.tool})
        if self.model:
            e["model"] = self.model
        if self.model_sha256:
            e["model_sha256"] = self.model_sha256
        if self.detail:
            e["detail"] = self.detail
        e.update(self.extra)
        e.update({k: v for k, v in more.items() if v is not None})
        e["software"] = software()
        return e


HOSTED = "hosted vision-language model"


def model_label(name) -> str:
    """A model as it is written down: the weights file name (no folders), or its id."""
    s = str(name or "").strip()
    if not s:
        return ""
    if s.startswith(("/", "~", "\\")) or (len(s) > 2 and s[1] == ":") or "\\" in s or os.path.isfile(s):
        return Path(s.replace("\\", "/")).name                  # a file: its name (folders carry user names)
    return s                                                       # a model id (e.g. org/name on a hub)


def make_origin(by="manual", tool="AIO QC", model="", model_path=None, detail="", timeless=False,
                **extra) -> Origin:
    if model_path and not model:
        model = Path(str(model_path)).name
    model = model_label(model)
    sha = file_sha256(model_path) if model_path and Path(str(model_path)).is_file() else ""
    return Origin(by if by in BY else "unknown", tool, str(model or ""), sha, detail, dict(extra), bool(timeless))


_ctx: contextvars.ContextVar = contextvars.ContextVar("aioqc_provenance_origin", default=None)


@contextlib.contextmanager
def origin(by="manual", tool="Image QC", model="", model_path=None, detail="", **extra):
    """Everything written inside is attributed to this origin (in this thread; pass current() to workers)."""
    o = by if isinstance(by, Origin) else make_origin(by, tool, model, model_path, detail, **extra)
    tok = _ctx.set(o)
    try:
        yield o
    finally:
        _ctx.reset(tok)


def current() -> Origin:
    return _ctx.get() or Origin()


def active() -> bool:
    """True inside an origin() block (a feature said what is writing)."""
    return _ctx.get() is not None


def unless_active(by="manual", tool="AIO QC", **kw):
    """origin(…) — or nothing, when a feature already set one (it knows better: a model, an import …)."""
    return contextlib.nullcontext() if active() else origin(by, tool, **kw)


def shape_record(o: Origin | None = None, score=None, **more) -> dict:
    """The record of a shape made now (give a detection its score)."""
    o = o or current()
    return {"id": new_id(), "created": o.entry(score=None if score is None else round(float(score), 4), **more)}


# ====================================================================== annotations
def _pts(sh):
    out = []
    for p in sh.get("points") or []:
        try:
            out.append((round(float(p[0]), 2), round(float(p[1]), 2)))
        except (TypeError, ValueError, IndexError):
            out.append(str(p))
    return tuple(out)


def _changed(old: dict, new: dict) -> list[str]:
    what = []
    if str(old.get("label", "")) != str(new.get("label", "")):
        what.append("label")
    if _pts(old) != _pts(new):
        what.append("geometry")
    if (old.get("shape_type") or "polygon") != (new.get("shape_type") or "polygon"):
        what.append("type")
    if old.get("group_id") != new.get("group_id"):
        what.append("group")
    if (old.get("description") or "") != (new.get("description") or ""):
        what.append("description")
    if (old.get("flags") or {}) != (new.get("flags") or {}):
        what.append("flags")
    return what


def _sig(sh) -> tuple:
    return (str(sh.get("label", "")), sh.get("shape_type") or "polygon", sh.get("group_id"), _pts(sh),
            sh.get("description") or "", json.dumps(sh.get("flags") or {}, sort_keys=True, default=str))


def bump(rec: dict | None, what: list, o: Origin | None = None) -> dict:
    """A record with one more change (a record made for a shape that had none)."""
    o = o or current()
    r = dict(rec) if isinstance(rec, dict) and rec.get("id") else \
        {"id": new_id(), "created": {"by": "unknown", "note": "made before provenance was recorded"}}
    ch = list(r.get("changes") or [])
    ch.append(o.entry(what=list(what)))
    if len(ch) > MAX_CHANGES:
        r["changes_dropped"] = int(r.get("changes_dropped", 0)) + len(ch) - MAX_CHANGES
        ch = ch[-MAX_CHANGES:]
    r["changes"] = ch
    return r


def _rec(sh) -> dict | None:
    p = sh.get(KEY) if isinstance(sh, dict) else None
    return p if isinstance(p, dict) and p.get("id") else None


def stamp_shapes(shapes: list, old_shapes=(), o: Origin | None = None) -> dict:
    """Give every new / changed shape its record (in place); returns {"added", "changed", "removed"}."""
    o = o or current()
    old_shapes = [s for s in (old_shapes or []) if isinstance(s, dict)]
    old_by_id = {r["id"]: s for s in old_shapes if (r := _rec(s))}
    plain_old = [s for s in old_shapes if not _rec(s)]                  # written before provenance / elsewhere
    plain_sigs: dict = {}
    for i, s in enumerate(plain_old):
        plain_sigs.setdefault(_sig(s), []).append(i)
    used_plain = set()
    seen, kept_ids = set(), set()
    added = changed = 0
    pending = []
    for idx, sh in enumerate(shapes):
        if not isinstance(sh, dict):
            continue
        r = _rec(sh)
        if r is not None and r["id"] not in seen:
            seen.add(r["id"])
            prev = old_by_id.get(r["id"])
            if prev is None:                          # arrives with its record (a detection, a copy): kept
                added += 1
                continue
            kept_ids.add(r["id"])
            what = _changed(prev, sh)
            if what:
                sh[KEY] = bump(r, what, o)
                changed += 1
            continue
        if r is not None:                             # a second shape with the same record: a copy
            sh[KEY] = {"id": new_id(), "created": o.entry(), "copied_from": r["id"]}
            added += 1
            continue
        hit = plain_sigs.get(_sig(sh))
        while hit and hit[0] in used_plain:
            hit.pop(0)
        if hit:                                       # untouched, made before provenance: left as it is
            used_plain.add(hit.pop(0))
            continue
        pending.append((idx, sh))
    # shapes without a record that match nothing: an edit of an old shape in its place, or new
    free = [i for i in range(len(plain_old)) if i not in used_plain]
    for idx, sh in pending:
        j = idx if idx in free else None
        if j is None and len(free) == 1 and len(pending) == 1:
            j = free[0]
        if j is not None and j < len(plain_old):
            free.remove(j)
            sh[KEY] = {"id": new_id(), "created": {"by": "unknown", "note": "made before provenance was recorded"},
                       "changes": [o.entry(what=_changed(plain_old[j], sh))]}
            changed += 1
        else:
            sh[KEY] = shape_record(o)
            added += 1
    removed = len([i for i in old_by_id if i not in kept_ids]) + len(free)
    return {"added": added, "changed": changed, "removed": removed}


def _read_old(path):
    try:
        from .io import load_json_light
        d = load_json_light(path)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError, TypeError):
        return None
    except Exception:                                  # noqa: BLE001 — an unreadable old file: treated as new
        return None


def stamp_labelme(data: dict, path=None, o: Origin | None = None, old: dict | None = None,
                  source: dict | None = None) -> dict:
    """Stamp a LabelMe dict before it is written to `path` (what the file held is read from there)."""
    if not _enabled or not isinstance(data, dict):
        return data
    o = o or current()
    if old is None and path is not None and Path(path).is_file():
        old = _read_old(path)
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return data
    counts = stamp_shapes(shapes, (old or {}).get("shapes") or [], o)
    prev = (old or {}).get(KEY) if isinstance((old or {}).get(KEY), dict) else None
    rec = dict(data.get(KEY) if isinstance(data.get(KEY), dict) else (prev or {}))
    rec["format"] = FORMAT
    if old is None and "created" not in rec:
        rec["created"] = o.entry(objects=len(shapes))
    elif old is not None:
        other = {k: v for k, v in data.items() if k not in ("shapes", KEY, "imageData")} != \
            {k: v for k, v in old.items() if k not in ("shapes", KEY, "imageData")}
        if any(counts.values()) or other:
            rec["modified"] = o.entry(**{k: v for k, v in counts.items() if v})
            rec["edits"] = int(rec.get("edits", 0)) + 1
    if source:
        rec["source"] = source
    data[KEY] = rec
    return data


def derive_labelme(data: dict, source_rec: dict | None, o: Origin | None = None, **source) -> dict:
    """A LabelMe file written from another one (a conversion, a copy for a batch): the shapes keep their records,
    the file keeps the source's and gets "derived": what wrote it, from what."""
    if not _enabled or not isinstance(data, dict):
        return data
    o = o or current()
    rec = dict(source_rec) if isinstance(source_rec, dict) else {}
    rec["format"] = FORMAT
    rec["derived"] = o.entry(**{k: v for k, v in source.items() if v})
    data[KEY] = rec
    return data


def strip(data: dict) -> dict:
    """A LabelMe dict without any provenance (for outputs where it is not wanted)."""
    data = dict(data)
    data.pop(KEY, None)
    data["shapes"] = [{k: v for k, v in s.items() if k != KEY} if isinstance(s, dict) else s
                      for s in data.get("shapes") or []]
    return data


def describe_shape(sh: dict) -> str:
    """One line for a tooltip: who made it, with what, when — and how often it changed."""
    r = _rec(sh)
    if r is None:
        return ""
    c = r.get("created") or {}
    by = c.get("by", "unknown")
    head = {"manual": "Drawn by hand", "model": "Made by a model", "import": "Imported",
            "conversion": "Converted", "rule": "Made by a rule", "augmentation": "Augmented",
            "unknown": "Made before provenance was recorded"}.get(by, by)
    bits = [head]
    if c.get("model"):
        bits.append(str(c["model"]) + (f" ({float(c['score']):.2f})" if c.get("score") is not None else ""))
    if c.get("tool") and by != "unknown":
        bits.append(str(c["tool"]))
    if c.get("at"):
        bits.append(str(c["at"])[:10])
    line = " · ".join(bits)
    ch = r.get("changes") or []
    if ch:
        n = len(ch) + int(r.get("changes_dropped", 0))
        last = ch[-1]
        line += f"\nChanged {n}×, last: {', '.join(last.get('what') or [])} ({last.get('by', '')}, " \
                f"{str(last.get('at', ''))[:10]})"
    return line


# ---------------------------------------------------------------------- YOLO / COCO
def log_yolo(root, rel_file: str, counts: dict, o: Origin | None = None):
    """One line per save of a YOLO label file (YOLO text has no room for records)."""
    if not _enabled or not any(counts.values()):
        return
    o = o or current()
    line = json.dumps(dict(o.entry(**{k: v for k, v in counts.items() if v}), file=str(rel_file)),
                      ensure_ascii=False)
    try:
        with open(Path(root) / YOLO_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def yolo_counts(old_rows: list, new_rows: list) -> dict:
    """Objects added / changed / removed between two lists of YOLO rows (token lists)."""
    a = [" ".join(r) for r in old_rows]
    b = [" ".join(r) for r in new_rows]
    common = 0
    pool = {}
    for x in a:
        pool[x] = pool.get(x, 0) + 1
    for x in b:
        if pool.get(x):
            pool[x] -= 1
            common += 1
    left, right = len(a) - common, len(b) - common
    changed = min(left, right)
    return {"added": right - changed, "changed": changed, "removed": left - changed}


# ====================================================================== datasets
_SECRET = ("key", "token", "secret", "password", "passwd", "credential", "auth", "cookie")


def clean_settings(d, depth: int = 0):
    """Settings fit for a record: no secrets, folders by name only, nothing huge."""
    if depth > 4:
        return None
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            ks = str(k)
            if any(s in ks.lower() for s in _SECRET):
                continue
            cv = clean_settings(v, depth + 1)
            if cv is not None:
                out[ks] = cv
        return out
    if isinstance(d, (list, tuple, set)):
        items = [clean_settings(v, depth + 1) for v in list(d)[:200]]
        return [v for v in items if v is not None]
    if isinstance(d, Path) or (isinstance(d, str) and len(d) > 2
                               and (d.startswith(("/", "~", "\\")) or (d[1] == ":" and d[2] in "\\/"))):
        return str(d).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d if not isinstance(d, str) else d[:500]
    return str(d)[:200]


def fingerprint(folder, cancel=None, limit: int = 0) -> dict:
    """{"files", "bytes", "listing_sha256"} of a folder (hidden folders and provenance files left out)."""
    folder = Path(folder)
    if folder.is_file():
        return quick_file(folder)
    rows = []
    total = 0

    def walk(d: str, rel: str):
        nonlocal total
        try:
            it = list(os.scandir(d))
        except OSError:
            return
        for e in it:
            if cancel is not None and cancel.is_set():
                return
            name = e.name
            if name.startswith(".") or name == SIDECAR:
                continue
            r = f"{rel}/{name}" if rel else name
            try:
                if e.is_dir(follow_symlinks=False):
                    walk(e.path, r)
                elif e.is_file():
                    sz = e.stat().st_size
                    rows.append(f"{r}\t{sz}")
                    total += sz
            except OSError:
                continue
    walk(str(folder), "")
    rows.sort()
    h = hashlib.sha256("\n".join(rows).encode("utf-8", "surrogatepass")).hexdigest()
    return {"files": len(rows), "bytes": total, "listing_sha256": h}


def quick_file(p) -> dict:
    """A file's size and a SHA-256 of its size, first and last MiB — instant even for a video of many GB."""
    p = Path(p)
    try:
        st = p.stat()
        h = hashlib.sha256(str(st.st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if st.st_size > 2 << 20:
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read(1 << 20))
        return {"files": 1, "bytes": st.st_size, "quick_sha256": h.hexdigest()}
    except OSError:
        return {"files": 0}


def last_record_id(folder) -> str:
    try:
        d = json.loads((Path(folder) / SIDECAR).read_text(encoding="utf-8"))
        recs = d.get("records") or []
        return str(recs[-1].get("id", "")) if recs else ""
    except (OSError, ValueError, AttributeError):
        return ""


def input_info(p, with_fingerprint: bool = True) -> dict:
    p = Path(p)
    info = {"name": p.name}
    if with_fingerprint and p.exists():
        info["fingerprint"] = fingerprint(p)
    pid = last_record_id(p if p.is_dir() else p.parent)
    if pid:
        info["provenance"] = pid
    return info


def record(folder, action: str, tool: str, inputs=(), settings=None, counts=None, o: Origin | None = None,
           output_fingerprint: bool = True, input_fingerprints: bool = True, **more) -> dict | None:
    """Add a record to <folder>/.aio_qc_provenance.json (made when missing); returns the record."""
    if not _enabled:
        return None
    o = o or current()
    folder = Path(folder)
    rec = {"id": uuid.uuid4().hex, **o.entry(), "action": action, "tool": tool}
    rec["inputs"] = [input_info(p, input_fingerprints) for p in inputs if p]
    if settings:
        rec["settings"] = clean_settings(settings)
    if counts:
        rec["counts"] = clean_settings(counts)
    rec.update(clean_settings(more) or {})
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if output_fingerprint:
            rec["output"] = fingerprint(folder)
    except OSError:
        return None
    with _REC_LOCK:
        return _append(folder, rec)


_REC_LOCK = threading.Lock()


def _append(folder: Path, rec: dict) -> dict | None:
    try:
        path = folder / SIDECAR
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
                doc = {"format": FORMAT, "records": []}
        except (OSError, ValueError):
            doc = {"format": FORMAT, "records": []}
        doc["records"].append(rec)
        from .io import atomic_write_text
        atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    except OSError:
        return None
    return rec


def read_records(folder) -> list[dict]:
    try:
        d = json.loads((Path(folder) / SIDECAR).read_text(encoding="utf-8"))
        return list(d.get("records") or [])
    except (OSError, ValueError, AttributeError):
        return []


# ====================================================================== images
def image_record(o: Origin | None = None, **more) -> dict:
    o = o or current()
    return o.entry(**{k: v for k, v in more.items() if v is not None})


def _xmp(info: dict) -> bytes:
    from xml.sax.saxutils import quoteattr
    rec = json.dumps(info, ensure_ascii=False, separators=(",", ":"), default=str)
    body = (f'<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:aioqc="{NS}" '
            f'xmp:CreatorTool={quoteattr(str(info.get("software", software())))} '
            + (f'xmp:CreateDate={quoteattr(str(info["at"]))} ' if info.get("at") else "")
            + f'aioqc:Record={quoteattr(rec)}/>'
            f'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
    return body.encode("utf-8")


def _exif_software(text: str) -> bytes:
    """A minimal little-endian EXIF (TIFF) block with one tag: Software (0x0131)."""
    val = text.encode("ascii", "replace")[:250] + b"\0"
    ifd_off = 8
    data_off = ifd_off + 2 + 12 + 4
    tiff = b"II*\0" + struct.pack("<I", ifd_off) + struct.pack("<H", 1)
    if len(val) <= 4:
        entry = struct.pack("<HHI", 0x0131, 2, len(val)) + val.ljust(4, b"\0")
        tiff += entry + struct.pack("<I", 0)
    else:
        entry = struct.pack("<HHII", 0x0131, 2, len(val), data_off)
        tiff += entry + struct.pack("<I", 0) + val
    return b"Exif\0\0" + tiff


def _jpeg_segments(data: bytes):
    """(marker, start, end) of the segments before the image data."""
    i = 2
    n = len(data)
    while i + 4 <= n and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA or m == 0xD9:                    # start of scan / end
            break
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 2
            continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        yield m, i, i + 2 + ln
        i += 2 + ln


def embed_jpeg(data: bytes, info: dict) -> bytes:
    if not data.startswith(b"\xff\xd8"):
        return data
    has_exif = has_xmp = False
    insert_at = 2
    for m, s, e in _jpeg_segments(data):
        if m == 0xE1:
            body = data[s + 4:e]
            if body.startswith(b"Exif\0\0"):
                has_exif = True
            elif body.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
                has_xmp = True
        if m in (0xE0,) and s == 2:                   # keep JFIF first
            insert_at = e
    segs = b""
    if not has_exif:
        ex = _exif_software(str(info.get("software", software())))
        segs += b"\xff\xe1" + struct.pack(">H", len(ex) + 2) + ex
    if not has_xmp:
        xmp = b"http://ns.adobe.com/xap/1.0/\0" + _xmp(info)
        if len(xmp) + 2 <= 0xFFFF:
            segs += b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
    return data[:insert_at] + segs + data[insert_at:] if segs else data


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def embed_png(data: bytes, info: dict) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(sig):
        return data
    i = 8
    after_ihdr = None
    while i + 8 <= len(data):
        ln = struct.unpack(">I", data[i:i + 4])[0]
        kind = data[i + 4:i + 8]
        end = i + 12 + ln
        if kind == b"IHDR":
            after_ihdr = end
        elif kind == b"iTXt" and data[i + 8:i + 8 + 17] == b"XML:com.adobe.xmp":
            return data                               # already has XMP
        elif kind == b"IDAT":
            break
        i = end
    if after_ihdr is None:
        return data
    itxt = b"XML:com.adobe.xmp\0" + b"\0\0" + b"\0" + b"\0" + _xmp(info)
    text = b"Software\0" + str(info.get("software", software())).encode("latin-1", "replace")
    return data[:after_ihdr] + _png_chunk(b"tEXt", text) + _png_chunk(b"iTXt", itxt) + data[after_ihdr:]


def embed_bytes(data: bytes, info: dict) -> bytes:
    """Image bytes with the record embedded (JPEG / PNG; anything else unchanged)."""
    if not _enabled or not data:
        return data
    if data.startswith(b"\xff\xd8"):
        return embed_jpeg(data, info)
    if data.startswith(b"\x89PNG"):
        return embed_png(data, info)
    return data


def embed_file(path, info: dict | None = None, **more) -> bool:
    """Embed the record into an image file the app just wrote (in place, atomically). Never call it on a hard link
    of the user's original (Convert & Split links images: those stay byte for byte as they were)."""
    if not _enabled:
        return False
    p = Path(path)
    try:
        if p.stat().st_nlink > 1:                     # a hard link of someone's original: left alone
            return False
        data = p.read_bytes()
    except OSError:
        return False
    out = embed_bytes(data, info or image_record(**more))
    if out is data or out == data:
        return False
    tmp = p.with_name(f".{p.name}.{os.getpid()}.prov")
    try:
        tmp.write_bytes(out)
        os.replace(tmp, p)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False


def read_image(path) -> dict | None:
    """The record embedded in an image (None: none)."""
    try:
        with open(path, "rb") as f:
            head = f.read(4 << 20)
    except OSError:
        return None
    k = head.find(b"aioqc:Record=")
    if k < 0:
        return None
    q = head[k + 13:k + 14]
    end = head.find(q, k + 14)
    if end < 0:
        return None
    from xml.sax.saxutils import unescape
    raw = unescape(head[k + 14:end].decode("utf-8", "replace"), {"&quot;": '"', "&apos;": "'"})
    try:
        return json.loads(raw)
    except ValueError:
        return None


# ====================================================================== models
def model_record(dataset=None, classes=None, settings=None, parent=None, weights=None, task: str = "",
                 o: Origin | None = None, **more) -> dict:
    """What a trained model came from (for its run folder, its ONNX metadata, a classifier checkpoint)."""
    o = o or make_origin("model", "Train Model")
    rec = o.entry(action="train", task=task or None)
    rec.pop("by", None)
    if dataset:
        ds = Path(dataset)
        folder = ds.parent if ds.is_file() else ds
        rec["dataset"] = {"name": folder.name, "fingerprint": fingerprint(folder)}
        pid = last_record_id(folder)
        if pid:
            rec["dataset"]["provenance"] = pid
    if classes:
        rec["classes"] = [str(c) for c in (classes.values() if isinstance(classes, dict) else classes)]
    if settings:
        rec["settings"] = clean_settings(settings)
    if parent:
        pp = Path(str(parent))
        rec["parent_model"] = {"name": pp.name}
        if pp.is_file():
            rec["parent_model"]["sha256"] = file_sha256(pp, short=0)
    if weights and Path(str(weights)).is_file():
        rec["weights"] = {"name": Path(str(weights)).name, "sha256": file_sha256(weights, short=0)}
    rec.update(clean_settings(more) or {})
    return rec
