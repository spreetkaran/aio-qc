# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Low-level, format-agnostic file helpers."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

try:  # optional speed-up (3-10x faster JSON parsing)
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tif", ".tiff", ".gif"}
TRASH_DIR_NAME = ".aio_qc_trash"


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def walk_files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    """
    Fast recursive file walk (os.scandir) that prunes hidden dirs such as
    `.aio_qc_trash`, `.git`, `.ipynb_checkpoints`.
    """
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        subdirs = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
                elif os.path.splitext(entry.name)[1].lower() in suffixes:
                    yield Path(entry.path)
            except OSError:
                continue
        stack.extend(reversed(subdirs))


def any_file(root: Path, suffixes: set[str], skip_dirs: set[str] = frozenset()) -> bool:
    """True as soon as one matching file is found (early exit)."""
    for p in walk_files(root, suffixes):
        if not skip_dirs.intersection(p.relative_to(root).parts[:-1]):
            return True
    return False


# ----------------------------------------------------------------------------
# Reading / writing
# ----------------------------------------------------------------------------
def load_json(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


_IMGDATA_KEY = b'"imageData"'
_WS = b" \t\r\n"


def load_json_light(path: str | Path) -> Any:
    """
    LabelMe JSON without its embedded image: the base64 `imageData` string (often several MB) is cut out by byte
    search before parsing (~90x faster). Base64 has no quote characters, so the string end is unambiguous.
    Read-only use (indexing, scans, crops) — never for files that will be saved.
    """
    raw = Path(path).read_bytes()
    start = 0
    while True:
        i = raw.find(_IMGDATA_KEY, start)
        if i < 0:
            break
        k = i + len(_IMGDATA_KEY)
        while k < len(raw) and raw[k] in _WS:
            k += 1
        if k < len(raw) and raw[k] == 0x3A:                   # ':' -> this is the key, not a label containing it
            k += 1
            while k < len(raw) and raw[k] in _WS:
                k += 1
            if k < len(raw) and raw[k] == 0x22:               # '"' -> string value
                end = raw.find(b'"', k + 1)
                if end > 0:
                    raw = raw[:k] + b"null" + raw[end + 1:]
            break
        start = k
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via temp file + os.replace so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        if path.exists():
            try:
                shutil.copystat(path, tmp)
            except OSError:
                pass
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def dump_labelme(path: str | Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class Trash:
    """
    Recoverable delete: files are *moved* to <root>/.aio_qc_trash/<timestamp>/<relpath>.
    Hidden dirs are skipped by discovery, so trashed files never reappear.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.session_dir: Path | None = None

    def move(self, path: str | Path) -> Path:
        path = Path(path)
        if self.session_dir is None:
            self.session_dir = self.root / TRASH_DIR_NAME / time.strftime("%Y%m%d-%H%M%S")
        try:
            rel = path.resolve().relative_to(self.root.resolve())
        except ValueError:
            rel = Path(path.name)
        dest = self.session_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}_{time.time_ns()}{dest.suffix}")
        shutil.move(str(path), str(dest))
        return dest


# ----------------------------------------------------------------------------
# Images (Qt-free; the GUI uses QImageReader instead)
# ----------------------------------------------------------------------------
_EXIF_ORIENTATION = 0x0112


def image_size(path: str | Path) -> tuple[int, int] | None:
    """(width, height) as displayed, i.e. after EXIF orientation. Header-only when PIL exists."""
    if Image is not None:
        try:
            with Image.open(path) as im:
                w, h = im.size
                try:
                    if im.getexif().get(_EXIF_ORIENTATION) in (5, 6, 7, 8):
                        w, h = h, w
                except Exception:
                    pass
                return int(w), int(h)
        except Exception:
            pass
    img = decode_image(path)
    if img is None:
        return None
    return int(img.shape[1]), int(img.shape[0])


def decode_image(path: str | Path):
    """Full decode with OpenCV; unicode-path safe (cv2.imread is not on Windows)."""
    try:
        import cv2
        import numpy as np
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def verify_image(path: str | Path) -> tuple[bool, tuple[int, int] | None, str]:
    """Deep check: fully decodes the image. Returns (ok, (w, h), message)."""
    if Image is not None:
        try:
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                try:
                    if im.getexif().get(_EXIF_ORIENTATION) in (5, 6, 7, 8):
                        w, h = h, w
                except Exception:
                    pass
                return True, (int(w), int(h)), ""
        except Exception as exc:
            pil_error = str(exc)
    else:
        pil_error = "PIL unavailable"
    img = decode_image(path)
    if img is None:
        return False, None, pil_error
    return True, (int(img.shape[1]), int(img.shape[0])), ""
