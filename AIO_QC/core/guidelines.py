# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Labelling guidelines with picture examples (Qt-free): <root>/.aio_qc/guidelines/

    g = GuideStore(root)
    g.set_text("car", "Whole vehicle incl. mirrors; parked cars too.")
    g.add_example("car", "good", image_path, (x1, y1, x2, y2), note="tight box", src=key)   -> example id
    g.add_example("tool:dim_area", "bad", …)       # a rule / finding type instead of a class
    g.get("car") -> {"text", "examples": [{"id", "kind", "note", "file", "src", "box", "ts"}]}
    g.export_html(path, scopes=None)                # one page for annotators (images embedded)

Scopes: a class name, "rule:<rule id>" for one Relationship Rule (.aio_qc/rules.json), or "tool:<tool key>" for
the check a QC tool makes. Example crops are small JPEGs
(≤ 320 px, the object plus some context, the object outlined) — a guide of 100 examples is a few MB at most.
guidelines.json is written atomically; crops are content-named so nothing is ever overwritten.
"""
from __future__ import annotations

import base64
import html
import json
import time
import uuid
from pathlib import Path

from .io import atomic_write_text

KINDS = ("good", "bad")
CROP_SIDE = 320


class GuideStore:
    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / ".aio_qc" / "guidelines"
        self.path = self.dir / "guidelines.json"
        self.data = {"version": 1, "scopes": {}}
        self.mtime = None
        self.reload()

    def reload(self):
        try:
            st = self.path.stat()
            if st.st_mtime == self.mtime:
                return
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("scopes"), dict):
                self.data = d
            self.mtime = st.st_mtime
        except (OSError, ValueError):
            pass

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=1))
        try:
            self.mtime = self.path.stat().st_mtime
        except OSError:
            pass

    # ------------------------------------------------------------------ queries
    def scopes(self) -> list[str]:
        return sorted(k for k, v in self.data["scopes"].items() if v.get("text") or v.get("examples"))

    def get(self, scope: str) -> dict | None:
        v = self.data["scopes"].get(scope)
        return v if v and (v.get("text") or v.get("examples")) else None

    def has(self, scope: str) -> bool:
        return self.get(scope) is not None

    def image(self, ex: dict) -> Path:
        return self.dir / ex["file"]

    def title(self, scope: str) -> str:
        """Readable name of a scope: the class, "Rule · <what the rule checks>", or "Check · <tool key>"."""
        if scope.startswith("rule:"):
            from . import rules as RL
            rid = scope[5:]
            r = next((r for r in RL.load_rules(self.root / ".aio_qc" / "rules.json") if r.get("id") == rid), None)
            return f"Rule · {RL.describe(r)}" if r is not None else "Rule · (deleted rule)"
        if scope.startswith("tool:"):
            return f"Check · {scope[5:]}"
        return scope

    # ------------------------------------------------------------------ edits
    def _scope(self, scope: str) -> dict:
        return self.data["scopes"].setdefault(scope, {"text": "", "examples": []})

    def set_text(self, scope: str, text: str):
        self._scope(scope)["text"] = text.strip()
        self.save()

    def add_example(self, scope: str, kind: str, image_path, box, note: str = "", src: str = "",
                    outline=None, poly=None) -> str | None:
        """poly: the shape's own outline (image px) — drawn instead of its bounding box."""
        if kind not in KINDS:
            raise ValueError(kind)
        jpeg = crop(image_path, box, outline=outline, poly=poly)
        if jpeg is None:
            return None
        eid = uuid.uuid4().hex[:12]
        rel = f"ex/{eid}.jpg"
        (self.dir / "ex").mkdir(parents=True, exist_ok=True)
        (self.dir / rel).write_bytes(jpeg)
        self._scope(scope)["examples"].append({"id": eid, "kind": kind, "note": note.strip(), "file": rel,
                                               "src": src, "box": [round(float(v), 2) for v in box],
                                               "ts": time.time()})
        self.save()
        return eid

    def update_example(self, scope: str, eid: str, **fields):
        for ex in self._scope(scope)["examples"]:
            if ex["id"] == eid:
                ex.update({k: v for k, v in fields.items() if k in ("kind", "note")})
        self.save()

    def remove_example(self, scope: str, eid: str):
        sc = self._scope(scope)
        keep = []
        for ex in sc["examples"]:
            if ex["id"] == eid:
                try:
                    (self.dir / ex["file"]).unlink()
                except OSError:
                    pass
            else:
                keep.append(ex)
        sc["examples"] = keep
        self.save()

    def rename_scope(self, old: str, new: str):
        sc = self.data["scopes"].pop(old, None)
        if sc is None:
            return
        tgt = self._scope(new)
        tgt["text"] = (tgt["text"] + "\n" + sc.get("text", "")).strip()
        tgt["examples"] += sc.get("examples", [])
        self.save()

    # ------------------------------------------------------------------ hand-out
    def export_html(self, path, scopes=None, title: str = "Labelling guidelines", titles: dict | None = None) -> int:
        titles = titles or {}
        scopes = [s for s in (scopes or self.scopes()) if self.has(s)]
        o = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
             "body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;margin:28px;color:#222;max-width:1100px}"
             "h1{color:#0b4f55} h2{color:#0b4f55;border-bottom:1px solid #cfd8dc;padding-bottom:3px;margin-top:28px}"
             ".ex{display:inline-block;margin:6px;vertical-align:top;width:200px;font-size:9pt}"
             ".ex img{max-width:200px;max-height:200px;border:3px solid #ccc;border-radius:4px}"
             ".good img{border-color:#2e7d32} .bad img{border-color:#c62828} .k{font-weight:600}"
             ".good .k{color:#2e7d32} .bad .k{color:#c62828} .t{white-space:pre-wrap}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><p style='color:#777'>{time.strftime('%Y-%m-%d')} · "
             f"{len(scopes)} topic(s)</p>"]
        for sc in scopes:
            g = self.get(sc)
            name = titles.get(sc) or self.title(sc)
            o.append(f"<h2>{html.escape(name)}</h2>")
            if g.get("text"):
                o.append(f"<div class=t>{html.escape(g['text'])}</div>")
            for kind in KINDS:
                for ex in [e for e in g["examples"] if e["kind"] == kind]:
                    try:
                        b64 = base64.b64encode((self.dir / ex["file"]).read_bytes()).decode()
                    except OSError:
                        continue
                    o.append(f"<div class='ex {kind}'><img src='data:image/jpeg;base64,{b64}'><br>"
                             f"<span class=k>{'✓ Do' if kind == 'good' else '✗ Don’t'}</span> "
                             f"{html.escape(ex.get('note', ''))}</div>")
        o.append("</body></html>")
        Path(path).write_text("".join(o), encoding="utf-8")
        return len(scopes)


def crop(image_path, box, side: int = CROP_SIDE, context: float = 0.35, outline=None, poly=None) -> bytes | None:
    """The object with `context` × its size around it, outlined (its own outline when `poly` is given, else its
    box), at most `side` px. Box in pixels or 0..1."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        img = None
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if all(0 <= v <= 1.0001 for v in (x1, y1, x2, y2)):
        x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    pad = max(bw, bh) * context + 6
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(W, x2 + pad)), int(min(H, y2 + pad))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    c = img[cy1:cy2, cx1:cx2].copy()
    s = min(1.0, side / max(c.shape[:2])) if max(c.shape[:2]) > side else min(4.0, side / max(c.shape[:2]))
    if s != 1.0:
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    col = outline or (0, 200, 255)
    p1 = (int((x1 - cx1) * s), int((y1 - cy1) * s))
    p2 = (int((x2 - cx1) * s), int((y2 - cy1) * s))
    bgr, thick = (int(col[2]), int(col[1]), int(col[0])), max(1, int(max(c.shape[:2]) / 160))
    if poly is not None and len(poly) >= 2:
        pts = np.array([[(float(x) - cx1) * s, (float(y) - cy1) * s] for x, y in poly], np.float32)
        cv2.polylines(c, [np.round(pts).astype(np.int32)], len(poly) >= 3, bgr, thick, cv2.LINE_AA)
    else:
        cv2.rectangle(c, p1, p2, bgr, thick, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return buf.tobytes() if ok else None
