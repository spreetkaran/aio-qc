# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Claude Review (Qt-free) — Claude looks at an image with its labels drawn and numbered, together with the class list,
your class descriptions and labelling guidelines, and reports what is wrong as structured findings.

    request(...)        the Messages API request for one image (the picture with numbered outlines, downscaled to
                        `max_side`; optional clean copy; the labels as text; a forced tool call so the answer is
                        always the same JSON shape)
    parse(message, …)   -> [Finding] in the ORIGINAL image's pixels
    Client              POST /v1/messages (live: retries on 429 / 5xx / overloaded, honouring retry-after) and
                        the Message Batches API (create / status / results) — half the price, results within 24 h,
                        usually much sooner
    ResultStore         <dataset>/.aio_qc/claude_review.sqlite: one row per image (signature = image file + labels +
                        model + prompt version — an unchanged image is never sent twice), dismissed findings,
                        accepted / dismissed counts per finding type, batches still running (survive a restart)

Security: the API key is read from an environment variable whose NAME is stored — never the key itself; it is not
written to any file, log or report. Images go to the API only for datasets you allowed (asked once per dataset).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROMPT_VERSION = 2
API_VERSION = "2023-06-01"
DEFAULT_BASE = "https://api.anthropic.com"
DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"
MODELS = [("claude-sonnet-5-5", "Claude Sonnet 5.5 — balanced (recommended)"),
          ("claude-opus-5-5", "Claude Opus 5.5 — most careful, slower"),
          ("claude-haiku-5-5", "Claude Haiku 5.5 — fastest, cheapest")]
DEFAULT_MODEL = MODELS[0][0]
MAX_SIDE = 1568                     # longer side sent; larger images are scaled down by the API anyway
TYPES = {
    "missing_object": "Missing object",
    "wrong_class": "Wrong class",
    "box_too_loose": "Box too loose",
    "box_cut_off": "Box cuts the object off",
    "duplicate_label": "Labelled twice",
    "wrong_text": "Text does not match",
    "guideline": "Guideline not followed",
    "not_an_object": "Nothing there",
    "other": "Other",
}
TONES = {"missing_object": "error", "wrong_class": "error", "wrong_text": "error", "not_an_object": "error",
         "duplicate_label": "warning", "box_cut_off": "warning", "box_too_loose": "warning", "guideline": "warning",
         "other": "muted"}

TOOL = {
    "name": "report_findings",
    "description": "Report every real labelling problem in this image (none when the labels are right).",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(TYPES)},
                    "shape": {"type": ["integer", "null"],
                              "description": "the number drawn on the label it is about; null for a missing object "
                                             "or a whole-image problem"},
                    "suggested_class": {"type": "string", "description": "from the class list: the right class "
                                                                         "for wrong_class, the class of the object "
                                                                         "for missing_object; '' otherwise"},
                    "box": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                            "description": "missing object / better box: [x1, y1, x2, y2] in pixels of THIS image"},
                    "text": {"type": "string", "description": "for wrong_text: the text as it reads in the image"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "description": "one short sentence an annotator understands"},
                },
                "required": ["type", "shape", "confidence", "reason"]}},
            "image_ok": {"type": "boolean", "description": "true when the labels look right"},
        },
        "required": ["findings", "image_ok"],
    },
}

SYSTEM = (
    "You review image annotations for a machine-learning dataset, like a careful senior annotator. You see the image "
    "with every label drawn as a coloured outline with its number, and the list of labels. Report only real "
    "problems you can see: an object of a listed class with no label, a label with the wrong class, a box much "
    "looser than the object or cutting part of it off, the same object labelled twice, text that does not match "
    "what is written, a label on nothing, or a guideline that is not followed. Use the class list exactly; never "
    "invent classes. Do not report style preferences, tiny box differences (a few pixels) or objects too small or "
    "blurred to judge. If unsure, give a lower confidence rather than leaving a clear problem out. Always answer "
    "with the report_findings tool.")


@dataclass
class Finding:
    type: str
    shape: int | None              # 0-based index into the file's shapes (None: missing object / whole image)
    label: str = ""
    suggested_class: str = ""
    box: list | None = None        # original image pixels
    text: str = ""
    confidence: float = 0.0
    reason: str = ""

    @property
    def fid(self) -> str:
        """Stable id for dismissing: type + shape/label + rounded box."""
        b = ",".join(str(int(round(v / 8))) for v in (self.box or []))
        return hashlib.sha1(f"{self.type}|{self.shape}|{self.label}|{self.suggested_class}|{b}".encode()).hexdigest()[:16]


@dataclass
class Prepared:
    """One image ready to send: the request params + what is needed to read the answer back."""
    key: str
    rel: str
    sig: str
    params: dict
    scale: float                  # sent pixels per original pixel
    labels: list = field(default_factory=list)    # label per shape number (1-based in the prompt)
    est_tokens: int = 0
    text: str = ""                # the label list / classes / guidelines part of the prompt
    pictures: list = field(default_factory=list)  # [(file name, what it is, jpeg bytes)] — for Claude Code


# ---------------------------------------------------------------------------------------------- the request
def _palette(i: int) -> tuple[int, int, int]:
    rnd = random.Random(i * 7919 + 13)
    return tuple(int(60 + rnd.random() * 195) for _ in range(3))


def overlay(img, shapes: list[dict], scale: float):
    """BGR image (already scaled by `scale`) with each shape outlined and numbered (1-based)."""
    import cv2
    import numpy as np
    out = img.copy()
    names = sorted({str(s.get("label", "")) for s in shapes})
    col = {n: _palette(i) for i, n in enumerate(names)}
    th = max(1, int(round(max(out.shape[:2]) / 700)))
    fs = max(0.45, max(out.shape[:2]) / 1600)
    for n, s in enumerate(shapes, 1):
        pts = np.asarray([[float(p[0]) * scale, float(p[1]) * scale] for p in s.get("points") or []], np.float32)
        if not len(pts):
            continue
        c = col[str(s.get("label", ""))]
        st = s.get("shape_type") or "polygon"
        if st == "rectangle" and len(pts) >= 2:
            p1, p2 = pts[0], pts[1]
            cv2.rectangle(out, (int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))),
                          (int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))), c, th, cv2.LINE_AA)
            anchor = (min(p1[0], p2[0]), min(p1[1], p2[1]))
        elif st == "circle" and len(pts) >= 2:
            r = float(np.hypot(*(pts[1] - pts[0])))
            cv2.circle(out, (int(pts[0][0]), int(pts[0][1])), int(r), c, th, cv2.LINE_AA)
            anchor = (pts[0][0] - r, pts[0][1] - r)
        elif st in ("point", "points") or len(pts) == 1:
            for p in pts:
                cv2.circle(out, (int(p[0]), int(p[1])), th * 3, c, -1, cv2.LINE_AA)
            anchor = tuple(pts[0])
        else:
            cv2.polylines(out, [pts.astype(np.int32)], st not in ("linestrip", "line"), c, th, cv2.LINE_AA)
            anchor = (float(pts[:, 0].min()), float(pts[:, 1].min()))
        tag = str(n)
        (tw, tht), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y = int(max(0, anchor[0])), int(max(tht + 4, anchor[1]))
        cv2.rectangle(out, (x, y - tht - 4), (x + tw + 4, y), c, -1)
        cv2.putText(out, tag, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0) if sum(c) > 380 else
                    (255, 255, 255), th, cv2.LINE_AA)
    return out


def _jpeg_b64(img, quality=88) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def label_lines(shapes: list[dict], scale: float) -> list[str]:
    out = []
    for n, s in enumerate(shapes, 1):
        pts = s.get("points") or []
        st = s.get("shape_type") or "polygon"
        try:
            xs, ys = [float(p[0]) * scale for p in pts], [float(p[1]) * scale for p in pts]
            box = f"[{min(xs):.0f}, {min(ys):.0f}, {max(xs):.0f}, {max(ys):.0f}]" if pts else "[]"
        except (TypeError, ValueError, IndexError):
            box = "[?]"
        extra = []
        if s.get("description"):
            extra.append(f"text {json.dumps(str(s['description']), ensure_ascii=False)}")
        fl = [k for k, v in (s.get("flags") or {}).items() if v]
        if fl:
            extra.append("flags " + ", ".join(fl))
        if s.get("group_id") is not None:
            extra.append(f"group {s['group_id']}")
        out.append(f"#{n} {s.get('label', '')} — {st}, box {box}" + (" · " + " · ".join(extra) if extra else ""))
    return out


def signature(image_path, shapes: list[dict], tags: list[str], model: str, extra: str = "") -> str:
    try:
        st = os.stat(image_path)
        isig = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        isig = "?"
    lab = json.dumps([[s.get("label"), s.get("shape_type"), s.get("points"), s.get("description") or ""]
                      for s in shapes] + [sorted(tags)], sort_keys=True, default=str)
    return hashlib.sha1(f"v{PROMPT_VERSION}|{model}|{isig}|{lab}|{extra}".encode()).hexdigest()


def prepare(key: str, rel: str, image_path, shapes: list[dict], tags: list[str], classes: list[str],
            descriptions: dict, guidelines: str, model: str, max_side: int = MAX_SIDE, clean_copy: bool = False,
            classification: bool = False, max_tokens: int = 2048) -> Prepared:
    from .io import decode_image
    img = decode_image(image_path)
    if img is None:
        raise ValueError("the image could not be read")
    H, W = img.shape[:2]
    scale = min(1.0, float(max_side) / max(W, H))
    if scale < 1.0:
        import cv2
        img = cv2.resize(img, (max(1, round(W * scale)), max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
    sw, sh = img.shape[1], img.shape[0]
    content, pictures = [], []
    if clean_copy:
        b = _jpeg_b64(img)
        pictures.append(("clean.jpg", "the image without any drawing", base64.b64decode(b)))
        content += [{"type": "text", "text": "The image without any drawing:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}]
    b = _jpeg_b64(overlay(img, shapes, scale) if shapes else img)
    pictures.append(("labels.jpg", "the image with the labels drawn and numbered" if shapes else "the image",
                     base64.b64decode(b)))
    content += [{"type": "text", "text": "The image with the labels drawn and numbered:" if shapes else "The image:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}]
    desc = "\n".join(f"- {c}: {descriptions[c]}" if descriptions.get(c) else f"- {c}" for c in classes)
    parts = [f"Image size: {sw} × {sh} pixels (all coordinates are in these pixels).",
             f"Classes (use exactly these names):\n{desc or '(none given)'}"]
    if classification:
        parts.append("This is an image-classification dataset: the image's class(es): " +
                     (", ".join(tags) if tags else "(none)") + ". Report wrong_class with shape null and the right "
                     "class in suggested_class when the class is wrong; missing_object is not used.")
    else:
        parts.append("Labels:\n" + ("\n".join(label_lines(shapes, scale)) if shapes else "(no labels in this image)"))
        if tags:
            parts.append("Whole-image tags: " + ", ".join(tags))
    if guidelines.strip():
        parts.append("Labelling guidelines of this dataset:\n" + guidelines.strip())
    content.append({"type": "text", "text": "\n\n".join(parts)})
    params = {"model": model, "max_tokens": int(max_tokens), "system": SYSTEM, "tools": [TOOL],
              "tool_choice": {"type": "tool", "name": TOOL["name"]},
              "messages": [{"role": "user", "content": content}]}
    est = int(sw * sh / 750) * (2 if clean_copy else 1) + (len(SYSTEM) + len(json.dumps(TOOL)) +
                                                          sum(len(p) for p in parts)) // 4 + 300
    return Prepared(key, rel, "", params, scale, [str(s.get("label", "")) for s in shapes], est,
                    "\n\n".join(parts), pictures)


def guidelines_text(store, classes: list[str]) -> str:
    """The written guidelines (text only — pictures are not sent) for these classes and the general rules."""
    if store is None:
        return ""
    try:
        store.reload()
    except Exception:
        pass
    out = []
    for scope in store.scopes():
        g = store.get(scope) or {}
        t = (g.get("text") or "").strip()
        if not t:
            continue
        if scope in classes:
            out.append(f"- {scope}: {t}")
        elif scope.startswith("rule:"):
            out.append(f"- {store.title(scope)}: {t}")
    return "\n".join(out)


def parse(message: dict, prep: Prepared) -> list[Finding]:
    """A Messages API response -> findings in original pixels (shape numbers -> 0-based indices)."""
    data = None
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == TOOL["name"]:
            data = block.get("input") or {}
            break
    if data is None:                                    # fall back: JSON in a text block
        for block in message.get("content") or []:
            if block.get("type") == "text":
                try:
                    data = json.loads(block.get("text", "").strip().strip("`").removeprefix("json"))
                    break
                except ValueError:
                    continue
    if not isinstance(data, dict):
        raise ValueError("the answer had no findings")
    out = []
    inv = 1.0 / prep.scale if prep.scale else 1.0
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        t = str(f.get("type") or "other")
        t = t if t in TYPES else "other"
        n = f.get("shape")
        idx = int(n) - 1 if isinstance(n, (int, float)) and 1 <= int(n) <= len(prep.labels) else None
        box = f.get("box")
        if isinstance(box, list) and len(box) == 4:
            try:
                x1, y1, x2, y2 = (float(v) * inv for v in box)
                box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            except (TypeError, ValueError):
                box = None
        else:
            box = None
        try:
            conf = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out.append(Finding(t, idx, prep.labels[idx] if idx is not None else "", str(f.get("suggested_class") or ""),
                           box, str(f.get("text") or ""), conf, str(f.get("reason") or "").strip()))
    return out


def usage_of(message: dict) -> tuple[int, int]:
    u = message.get("usage") or {}
    return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)


# ---------------------------------------------------------------------------------------------- the API
class ApiError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"{status}: {message}" if status else message)
        self.status, self.retry_after = status, retry_after


def api_key(env_name: str) -> str:
    name = (env_name or DEFAULT_KEY_ENV).strip()
    key = os.environ.get(name, "").strip()
    if not key:
        raise ApiError(0, f"no API key: set the environment variable {name} (then restart AIO QC)")
    return key


class Client:
    def __init__(self, key_env: str = DEFAULT_KEY_ENV, base_url: str = DEFAULT_BASE, timeout: float = 180.0,
                 retries: int = 4, sleep=time.sleep):
        self.key_env, self.base = key_env or DEFAULT_KEY_ENV, (base_url or DEFAULT_BASE).rstrip("/")
        self.timeout, self.retries, self.sleep = timeout, retries, sleep

    def _req(self, method: str, path: str, body=None, raw: bool = False):
        url = self.base + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"x-api-key": api_key(self.key_env), "anthropic-version": API_VERSION,
                   "content-type": "application/json", "user-agent": "AIO-QC"}
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = r.read()
                return payload if raw else json.loads(payload.decode("utf-8"))
            except urllib.error.HTTPError as e:
                try:
                    msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
                except Exception:
                    msg = e.reason or ""
                ra = e.headers.get("retry-after") if e.headers else None
                try:
                    ra = float(ra) if ra else None
                except ValueError:
                    ra = None
                last = ApiError(e.code, msg or str(e.reason), ra)
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < self.retries:
                    self.sleep(ra if ra is not None else min(60.0, 2.0 * 2 ** attempt))
                    continue
                raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = ApiError(0, f"cannot reach {self.base}: {getattr(e, 'reason', e)}")
                if attempt < self.retries:
                    self.sleep(min(30.0, 2.0 * 2 ** attempt))
                    continue
                raise last from None
        raise last or ApiError(0, "request failed")

    def message(self, params: dict) -> dict:
        return self._req("POST", "/v1/messages", params)

    def test(self, model: str) -> str:
        r = self.message({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Reply OK."}]})
        return "".join(b.get("text", "") for b in r.get("content") or [] if b.get("type") == "text").strip() or "OK"

    # batches
    def batch_create(self, requests: list[dict]) -> dict:
        return self._req("POST", "/v1/messages/batches", {"requests": requests})

    def batch_get(self, batch_id: str) -> dict:
        return self._req("GET", f"/v1/messages/batches/{batch_id}")

    def batch_results(self, batch: dict) -> list[dict]:
        url = batch.get("results_url") or f"{self.base}/v1/messages/batches/{batch['id']}/results"
        path = url[len(self.base):] if url.startswith(self.base) else "/v1/messages/batches/" + batch["id"] + \
            "/results"
        raw = self._req("GET", path, raw=True)
        return [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]

    def batch_cancel(self, batch_id: str) -> dict:
        return self._req("POST", f"/v1/messages/batches/{batch_id}/cancel", {})


# ---------------------------------------------------------------------------------------------- Claude Code
class PlanError(ApiError):
    """Claude Code could not answer: not signed in, the plan's limit reached, not installed — stop the run."""


CC_SYSTEM = SYSTEM.replace("Always answer with the report_findings tool.",
                           "Read the image file(s) named in the request with the Read tool, then give your answer "
                           "as the structured output — nothing else. Do not run commands or change any file.")
CC_PLAN_LIMIT = 25                      # images per run on a Claude plan: interactive use, not whole datasets


def find_claude(hint: str = "") -> str | None:
    """The Claude Code program: the path you chose, else PATH, else where the installers put it."""
    cands = [hint] if hint else []
    w = shutil.which("claude")
    if w:
        cands.append(w)
    home = Path.home()
    if sys.platform == "win32":
        cands += [str(home / ".local" / "bin" / "claude.exe"),
                  str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "claude.exe"),
                  str(Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd")]
    else:
        cands += [str(home / ".local" / "bin" / "claude"), "/opt/homebrew/bin/claude", "/usr/local/bin/claude",
                  "/usr/bin/claude", str(home / ".npm-global" / "bin" / "claude")]
    for c in sorted(cands, key=lambda c: str(c).lower().endswith((".cmd", ".bat"))):   # claude.exe before .cmd
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def plan_env() -> dict:
    """The environment for Claude Code: without API-key variables, so it uses YOUR sign-in (your plan), never a key
    that happens to be set. AIO QC does not read, copy or store the sign-in itself."""
    env = dict(os.environ)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    return env


class ClaudeCode:
    """Runs the unmodified Claude Code program you installed and signed in to (`claude -p`, structured output)."""

    def __init__(self, path: str = "", timeout: float = 300.0):
        self.path = find_claude(path)
        self.timeout = timeout

    def _need(self):
        if not self.path:
            raise PlanError(0, "Claude Code is not installed (or not found) — Connection… shows how to install it")
        if self.path.lower().endswith((".cmd", ".bat")):          # cmd.exe would mangle the arguments
            raise PlanError(0, "found the npm launcher (claude.cmd) — install Claude Code with the native installer "
                               "or WinGet (claude.exe), or choose claude.exe in Connection…")

    def version(self) -> str:
        self._need()
        r = subprocess.run([self.path, "--version"], capture_output=True, text=True, timeout=60, env=plan_env())
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else "?"

    def _run(self, args: list[str], stdin: str, cwd: str, cancel=None) -> dict:
        self._need()
        p = subprocess.Popen([self.path, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=cwd,
                             env=plan_env())
        t0, first = time.monotonic(), True
        while True:
            try:
                out, err = p.communicate(stdin if first else None, timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                first = False
                if (cancel is not None and cancel.is_set()) or time.monotonic() - t0 > self.timeout:
                    p.kill()
                    p.communicate()
                    raise ApiError(0, "stopped" if cancel is not None and cancel.is_set() else
                                   f"Claude Code took longer than {self.timeout:.0f} s") from None
        data = {}
        text = (out or "").strip()
        for cand in (text, text.splitlines()[-1] if text else ""):
            try:
                d = json.loads(cand)
                if isinstance(d, dict):
                    data = d
                    break
            except ValueError:
                continue
        if not data:
            raise PlanError(0, (err or out or f"Claude Code exited with {p.returncode}").strip()[:300])
        if data.get("is_error") or data.get("subtype", "success") != "success":
            msg = str(data.get("result") or data.get("subtype") or "failed").strip()
            low = msg.lower()
            if any(w in low for w in ("login", "log in", "logged in", "limit", "credit", "auth", "billing",
                                      "subscription", "/login")):
                raise PlanError(0, msg[:300])
            raise ApiError(0, msg[:300])
        return data

    def test(self, model: str) -> str:
        with tempfile.TemporaryDirectory(prefix="aioqc_cc_") as d:
            data = self._run(["-p", "Reply with the single word OK.", "--output-format", "json", "--model", model,
                              "--tools", "", "--strict-mcp-config", "--no-session-persistence"], "", d)
        return str(data.get("result") or "OK").strip()[:60]

    def review(self, prep: "Prepared", cancel=None) -> dict:
        """One image -> a Messages-API-shaped message (so parse() reads it)."""
        with tempfile.TemporaryDirectory(prefix="aioqc_cc_") as d:
            names = []
            for fname, what, data in prep.pictures:
                (Path(d) / fname).write_bytes(data)
                names.append(f"- {Path(d) / fname} — {what}")
            prompt = ("Review the annotations of one image for a machine-learning dataset.\n"
                      "Read these image files first:\n" + "\n".join(names) + "\n\n" + prep.text)
            args = ["-p", "Review the image annotations described in the input. Answer with the structured output.",
                    "--output-format", "json", "--json-schema", json.dumps(TOOL["input_schema"]),
                    "--model", prep.params.get("model") or DEFAULT_MODEL, "--system-prompt", CC_SYSTEM,
                    "--tools", "Read", "--allowedTools", "Read", "--permission-mode", "dontAsk",
                    "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"]
            # only the Read tool, our own short system prompt, no MCP servers, nothing added to your session
            # history: each review costs your plan little more than the image itself
            data = self._run(args, prompt, d, cancel)
        so = data.get("structured_output")
        if not isinstance(so, dict):
            raise ApiError(0, "Claude Code gave no structured answer: " + str(data.get("result") or "")[:200])
        u = data.get("usage") or {}
        tin = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + \
            int(u.get("cache_creation_input_tokens") or 0)
        return {"content": [{"type": "tool_use", "name": TOOL["name"], "input": so}],
                "usage": {"input_tokens": tin, "output_tokens": int(u.get("output_tokens") or 0)}}


def batch_chunks(preps: list[Prepared], max_bytes: int = 200 << 20, max_n: int = 50_000):
    """Split into batches under the API's size / count limits."""
    cur, size = [], 0
    for p in preps:
        b = len(json.dumps(p.params))
        if cur and (size + b > max_bytes or len(cur) >= max_n):
            yield cur
            cur, size = [], 0
        cur.append(p)
        size += b
    if cur:
        yield cur


# ---------------------------------------------------------------------------------------------- the store
def _locked(fn):
    def wrap(self, *a, **k):
        with self.lock:
            return fn(self, *a, **k)
    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


class ResultStore:
    """One connection shared by the window and its worker threads — every call holds the lock."""

    def __init__(self, root):
        self.lock = threading.RLock()
        self.path = Path(root) / ".aio_qc" / "claude_review.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        try:
            from .sqlite_util import configure
            configure(self.db, self.path)
        except Exception:
            pass
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS results (rel TEXT PRIMARY KEY, sig TEXT, model TEXT, findings TEXT,
                                                tin INTEGER, tout INTEGER, at REAL);
            CREATE TABLE IF NOT EXISTS dismissed (rel TEXT, fid TEXT, type TEXT, at REAL, PRIMARY KEY (rel, fid));
            CREATE TABLE IF NOT EXISTS verdicts (type TEXT PRIMARY KEY, accepted INTEGER DEFAULT 0,
                                                 dismissed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, created REAL, model TEXT, items TEXT,
                                                status TEXT);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        """)
        self.db.commit()

    @_locked
    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    # results
    @_locked
    def get(self, rel: str, sig: str) -> list[Finding] | None:
        r = self.db.execute("SELECT findings FROM results WHERE rel=? AND sig=?", (rel, sig)).fetchone()
        if r is None:
            return None
        return [Finding(**d) for d in json.loads(r[0])]

    @_locked
    def put(self, rel: str, sig: str, model: str, findings: list[Finding], tin: int = 0, tout: int = 0):
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?)",
                        (rel, sig, model, json.dumps([asdict(f) for f in findings]), tin, tout, time.time()))
        self.db.commit()

    @_locked
    def usage(self) -> tuple[int, int, int]:
        r = self.db.execute("SELECT COUNT(*), COALESCE(SUM(tin),0), COALESCE(SUM(tout),0) FROM results").fetchone()
        return int(r[0]), int(r[1]), int(r[2])

    # dismissed / accepted
    @_locked
    def dismissed(self, rel: str) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT fid FROM dismissed WHERE rel=?", (rel,))}

    @_locked
    def dismiss(self, rel: str, f: Finding):
        self.db.execute("INSERT OR REPLACE INTO dismissed VALUES (?,?,?,?)", (rel, f.fid, f.type, time.time()))
        self.verdict(f.type, False)

    @_locked
    def verdict(self, type_: str, accepted: bool):
        self.db.execute("INSERT OR IGNORE INTO verdicts (type) VALUES (?)", (type_,))
        col = "accepted" if accepted else "dismissed"
        self.db.execute(f"UPDATE verdicts SET {col} = {col} + 1 WHERE type=?", (type_,))
        self.db.commit()

    @_locked
    def verdicts(self) -> dict:
        return {t: (a, d) for t, a, d in self.db.execute("SELECT type, accepted, dismissed FROM verdicts")}

    # batches
    @_locked
    def add_batch(self, batch_id: str, model: str, items: dict):
        self.db.execute("INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)",
                        (batch_id, time.time(), model, json.dumps(items), "in_progress"))
        self.db.commit()

    @_locked
    def open_batches(self) -> list[tuple[str, str, dict]]:
        return [(i, m, json.loads(it)) for i, m, it in
                self.db.execute("SELECT id, model, items FROM batches WHERE status != 'done' ORDER BY created")]

    @_locked
    def close_batch(self, batch_id: str, status: str = "done"):
        self.db.execute("UPDATE batches SET status=? WHERE id=?", (status, batch_id))
        self.db.commit()

    # per-dataset permission to send images
    @_locked
    def allowed(self) -> bool | None:
        r = self.db.execute("SELECT v FROM meta WHERE k='allowed'").fetchone()
        return None if r is None else r[0] == "1"

    @_locked
    def set_allowed(self, on: bool):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('allowed', ?)", ("1" if on else "0",))
        self.db.commit()


def collect_batch(client: Client, store: ResultStore, batch_id: str, items: dict, model: str) -> dict:
    """Check one batch; when it has ended, store every result. -> {"status", "done", "errors", "counts"}"""
    b = client.batch_get(batch_id)
    st = b.get("processing_status", "")
    out = {"status": st, "done": 0, "errors": [], "counts": b.get("request_counts") or {}}
    if st != "ended":
        return out
    for line in client.batch_results(b):
        cid = line.get("custom_id")
        it = items.get(cid)
        if it is None:
            continue
        res = line.get("result") or {}
        if res.get("type") != "succeeded":
            err = (res.get("error") or {}).get("error", {}).get("message") or res.get("type", "failed")
            out["errors"].append(f"{it['rel']}: {err}")
            continue
        msg = res.get("message") or {}
        prep = Prepared(it["key"], it["rel"], it["sig"], {}, it["scale"], it["labels"])
        try:
            fs = parse(msg, prep)
        except ValueError as exc:
            out["errors"].append(f"{it['rel']}: {exc}")
            continue
        tin, tout = usage_of(msg)
        store.put(it["rel"], it["sig"], model, fs, tin, tout)
        out["done"] += 1
    store.close_batch(batch_id)
    return out
