# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
An AI annotate project: everything the AI needs to annotate one project's images, in one config.yaml (Qt-free).

The app edits it (Auto-Annotate ▸ Annotate ▸ AI annotate); the YAML is what it saves — for sharing a project and
for the command line. Unknown keys are kept as they are (a newer app's settings survive an older one), missing keys
take their defaults, and `check()` explains every problem in plain words with the YAML line it is on.

The API key is never part of the config: only the NAME of the environment variable that holds it.
"""
from __future__ import annotations

import copy
import os
import re

MODES = {
    "vlm_detect": "Find and label the objects (vision model)",
    "detect_vlm": "A detector finds boxes, the vision model checks them",
    "detect": "A detector only (no API)",
    "classify": "Answer questions about each image (no boxes)",
}
BOX_FORMATS = {
    "yxyx_1000": "[ymin, xmin, ymax, xmax], 0–1000",
    "xyxy_1000": "[xmin, ymin, xmax, ymax], 0–1000",
    "xyxy_px": "[xmin, ymin, xmax, ymax] in pixels of the image sent",
    "xyxy_norm": "[xmin, ymin, xmax, ymax], 0–1",
    "yxyx_norm": "[ymin, xmin, ymax, xmax], 0–1",
}
DETECTORS = {
    "grounding_dino": "Grounding DINO (text prompts)",
    "owlv2": "OWLv2 (text prompts)",
    "florence2": "Florence-2 (text prompts)",
    "yolo_world": "YOLO-World (text prompts)",
    "yolo": "Your own YOLO model (.pt / .onnx)",
}
TEXT_CHECKS = {"none": "No check", "iso6346": "Container ID check digit (ISO 6346)", "luhn": "Luhn check digit"}
FORMATS = {"yolo": "YOLO txt", "labelme": "LabelMe", "both": "YOLO txt + LabelMe"}
MERGE = {"add": "Keep its shapes, add the AI's", "replace": "Replace it", "skip": "Leave the image alone"}
RULE_RE = re.compile(r"^\s*(len\(\s*\w+\s*\)|count\(\s*[\w\- .]+\s*\)|boxes|\w+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def default() -> dict:
    """A new project: every setting with its default (the app fills in the starred ones with you)."""
    return {
        "version": 1,
        "project": "my-project",
        "description": "",
        "connection": {"provider": "litellm", "base_url": "http://localhost:4000", "api_key_env": "LITELLM_API_KEY",
                       "api_version": "", "region": "", "timeout_s": 120, "workers": 8, "retries": 5},
        "classes": [],
        "flags": [],
        "references": {"max_side": 1024, "draw_boxes": True, "views": {"default": {"match": "", "images": [],
                                                                                   "note": ""}}},
        "mode": "vlm_detect",
        "roi": {"class": "", "padding": 0.10, "max_crops_per_image": 4},
        "vlm": {"model": "", "params": {"temperature": 0, "max_tokens": 3000}, "json_mode": True,
                "image_max_side": 1536, "box_format": "auto", "instructions": ""},
        "detector": {"type": "grounding_dino", "model": "", "prompts": {}, "box_threshold": 0.30,
                     "text_threshold": 0.25, "iou": 0.5, "device": "auto"},
        "segment": {"type": "none", "model": ""},
        "checks": {"fields": {}, "accept": [], "require_boxes": False, "keep_rejected": True,
                   "on_text_fail": "review"},
        "limits": {"targets": {}, "max_images": 0, "max_attempts_factor": 2.0, "budget_usd": 0, "seed": 42},
        "pricing": {"input": 0.0, "output": 0.0},
        "output": {"format": "yolo", "labelme_version": "6.3.1", "image_data": False, "write_confidence": True,
                   "needs_review_below": 0.6, "merge_with_existing": "add", "audit": True, "data_yaml": True},
    }


def new_class(name="", description="") -> dict:
    return {"name": name, "description": description, "min_confidence": 0.0, "max_per_image": 0, "min_size_px": 0,
            "inside": "", "text": False, "text_pattern": "", "text_check": "none", "parts": []}


def _merge(base, over):
    """Defaults deep-merged with a loaded file (unknown keys kept)."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return copy.deepcopy(over)
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if k in base and isinstance(base[k], dict) and isinstance(v, dict) \
            else copy.deepcopy(v)
    return out


def normalize(cfg: dict) -> dict:
    """Fill defaults; turn loose spellings into the canonical form (a list of class names, a missing view …)."""
    out = _merge(default(), cfg or {})
    classes = []
    for c in out.get("classes") or []:
        if isinstance(c, str):
            c = {"name": c}
        if isinstance(c, dict):
            cc = _merge(new_class(), c)
            if not isinstance(cc.get("parts"), list):
                cc["parts"] = [p for p in re.split(r"[,\s]+", str(cc.get("parts") or "")) if p]
            classes.append(cc)
    out["classes"] = classes
    flags = []
    for f in out.get("flags") or []:
        if isinstance(f, str):
            f = {"name": f, "description": ""}
        if isinstance(f, dict) and f.get("name"):
            flags.append({"name": str(f["name"]), "description": str(f.get("description") or "")})
    out["flags"] = flags
    views = (out.get("references") or {}).get("views") or {}
    if not isinstance(views, dict):
        views = {}
    if isinstance(views.get("default"), list):
        views["default"] = {"images": views["default"]}
    for k, v in list(views.items()):
        if isinstance(v, list):
            v = {"images": v}
        v = dict(v or {})
        v.setdefault("match", "")
        v.setdefault("note", "")
        imgs = v.get("images") or []
        v["images"] = [imgs] if isinstance(imgs, str) else list(imgs)
        views[k] = v
    views.setdefault("default", {"match": "", "images": [], "note": ""})
    out["references"]["views"] = views
    if not isinstance(out["checks"].get("fields"), dict):
        out["checks"]["fields"] = {}
    if isinstance(out["checks"].get("accept"), str):
        out["checks"]["accept"] = [out["checks"]["accept"]]
    return out


def class_names(cfg) -> list[str]:
    """The labels in YOLO order (class 0, 1, 2 …)."""
    return [c["name"] for c in cfg.get("classes") or [] if c.get("name")]


def needs_api(cfg) -> bool:
    return cfg.get("mode") in ("vlm_detect", "detect_vlm", "classify")


def needs_env(cfg) -> bool:
    return cfg.get("mode") in ("detect", "detect_vlm") or cfg["connection"].get("provider") == "local" or \
        (cfg.get("segment") or {}).get("type", "none") != "none"


# ------------------------------------------------------------------------------------------------ yaml io
def _yaml():
    try:
        import yaml
        return yaml
    except ImportError as exc:
        raise RuntimeError("AI annotate needs PyYAML to read and write config files:  pip install PyYAML") from exc


class ConfigError(Exception):
    def __init__(self, message, line=None):
        super().__init__(message)
        self.line = line


def loads(text: str) -> dict:
    yaml = _yaml()
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        raise ConfigError(f"{getattr(exc, 'problem', None) or exc}", line) from exc
    if not isinstance(data, dict):
        raise ConfigError("The file must hold settings (key: value), not a list or a single value.", 1)
    return normalize(data)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = loads(f.read())
    cfg["_path"] = os.path.abspath(path)
    return cfg


HEADER = """# AIO QC · AI annotate project — made in the app (Auto-Annotate ▸ Annotate ▸ AI annotate).
# Every setting is in the app; edit there, or here if you prefer (the app checks this file as you type).
# The API key is NOT in this file: only the name of the environment variable that holds it.
"""


def dumps(cfg: dict) -> str:
    yaml = _yaml()
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}

    class _D(yaml.SafeDumper):
        pass

    def _str(dumper, s):
        style = "|" if "\n" in s else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)
    _D.add_representer(str, _str)
    return HEADER + yaml.dump(data, Dumper=_D, sort_keys=False, allow_unicode=True, width=110,
                              default_flow_style=False)


def save(cfg: dict, path: str):
    text = dumps(cfg)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def line_of(text: str, dotted: str) -> int | None:
    """Best-effort YAML line of a setting such as 'vlm.model' or 'classes.2.name' (for error messages)."""
    lines = text.splitlines()
    i, req = 0, 0
    for p in dotted.split("."):
        if p.isdigit():
            n, k = int(p), -1
            for j in range(i + 1, len(lines)):
                st = lines[j]
                ind = len(st) - len(st.lstrip())
                if st.strip() and ind < req:
                    return None
                if st.lstrip().startswith("- ") and ind >= req:
                    k += 1
                    if k == n:
                        i, req = j, ind
                        break
            else:
                return None
            continue
        pat = re.compile(rf"^(\s*)(- )?{re.escape(p)}\s*:")
        for j in range(i, len(lines)):
            m = pat.match(lines[j])
            if m and len(m.group(1)) >= req:
                i, req = j, len(m.group(1)) + (2 if m.group(2) else 0)
                break
        else:
            return None
    return i + 1


# ------------------------------------------------------------------------------------------------ checks
def check(cfg: dict, base_dir: str = "") -> list[tuple[str, str]]:
    """[(setting, problem)] in plain words — empty when the project can run."""
    out = []
    add = lambda k, m: out.append((k, m))
    con = cfg["connection"]
    mode = cfg.get("mode")
    if mode not in MODES:
        add("mode", f"Choose what the AI does ({', '.join(MODES)}).")
    names = class_names(cfg)
    if mode != "classify" and not names:
        add("classes", "Add at least one class (what to annotate).")
    if mode == "classify" and not cfg["flags"] and not cfg["checks"]["fields"]:
        add("flags", "Add at least one question (a flag) for the AI to answer about each image.")
    seen = set()
    for i, c in enumerate(cfg["classes"]):
        n = c.get("name", "")
        if not n:
            add(f"classes.{i}.name", f"Class {i + 1} has no name.")
        elif not re.fullmatch(r"[^\s/\\]+", n):
            add(f"classes.{i}.name", f"Class “{n}”: no spaces or slashes in a class name (use _).")
        elif n in seen:
            add(f"classes.{i}.name", f"Class “{n}” is listed twice.")
        seen.add(n)
        if c.get("inside") and c["inside"] not in names:
            add(f"classes.{i}.inside", f"“{n}” must lie inside “{c['inside']}”, which is not a class.")
        if c.get("inside") == n:
            add(f"classes.{i}.inside", f"“{n}” cannot lie inside itself.")
        for p in c.get("parts") or []:
            if p not in names:
                add(f"classes.{i}.parts", f"Part “{p}” of “{n}” must also be a class (add it to the list).")
        if c.get("text_pattern"):
            try:
                re.compile(c["text_pattern"])
            except re.error as exc:
                add(f"classes.{i}.text_pattern", f"“{n}”: the text pattern is not valid ({exc}).")
        if c.get("text_check", "none") not in TEXT_CHECKS:
            add(f"classes.{i}.text_check", f"“{n}”: unknown text check “{c['text_check']}”.")
        if not 0 <= float(c.get("min_confidence") or 0) <= 1:
            add(f"classes.{i}.min_confidence", f"“{n}”: the minimum confidence is between 0 and 1.")
    if needs_api(cfg):
        if con.get("provider") != "local" and not con.get("base_url") and con.get("provider") != "bedrock":
            add("connection.base_url", "Type the address of the AI service.")
        if not cfg["vlm"].get("model"):
            add("vlm.model", "Choose the vision model.")
        bf = cfg["vlm"].get("box_format", "auto")
        if bf != "auto" and bf not in BOX_FORMATS:
            add("vlm.box_format", f"Unknown box format “{bf}”.")
    if mode in ("detect", "detect_vlm"):
        d = cfg["detector"]
        if d.get("type") not in DETECTORS:
            add("detector.type", "Choose a detector.")
        if not d.get("model"):
            add("detector.model", "Choose the detector model (a folder or a file).")
    if cfg["roi"].get("class") and cfg["roi"]["class"] not in names:
        add("roi.class", f"“Only inside” uses “{cfg['roi']['class']}”, which is not a class.")
    for i, rule in enumerate(cfg["checks"].get("accept") or []):
        if not RULE_RE.match(str(rule)):
            add(f"checks.accept.{i}", f"Rule “{rule}” is not understood — write it like  photorealistic >= 4  "
                                      f"or  count(class) >= 1  or  len(artifacts) <= 1.")
    for n, t in (cfg["limits"].get("targets") or {}).items():
        if n not in names:
            add("limits.targets", f"Target for “{n}”, which is not a class.")
    out_ = cfg["output"]
    if out_.get("format") not in FORMATS:
        add("output.format", "Choose YOLO txt, LabelMe or both.")
    if out_.get("merge_with_existing") not in MERGE:
        add("output.merge_with_existing", "Choose what to do with images that already have annotations.")
    for k, v in (cfg["references"].get("views") or {}).items():
        for p in v.get("images") or []:
            full = p if os.path.isabs(p) else os.path.join(base_dir or ".", p)
            if not os.path.exists(full):
                add(f"references.views.{k}.images", f"Example image not found: {p}")
    return out


# ------------------------------------------------------------------------------------------------ templates
TEMPLATES = {
    "find": ("Find objects",
             "The AI finds and boxes the objects you describe.",
             {"mode": "vlm_detect"}),
    "find_check": ("Find objects, then check each one",
                   "A detector on your GPU proposes boxes; the vision model keeps the right ones and fixes labels. "
                   "Tightest boxes.",
                   {"mode": "detect_vlm"}),
    "read_text": ("Read text (IDs, plates) with its parts",
                  "Boxes around text, the text itself written with each box, a pattern / check digit to validate it, "
                  "and optional boxes for its parts.",
                  {"mode": "vlm_detect", "classes": [dict(new_class("text_box", "the full text, with a small margin"),
                                                          text=True)]}),
    "inside": ("Find small things inside a bigger object",
               "Boxes for a big object first (e.g. from your own model), then the AI looks inside each one for the "
               "small things — better for tiny objects.",
               {"mode": "vlm_detect", "roi": {"class": "", "padding": 0.1, "max_crops_per_image": 4}}),
    "classify": ("Yes / no questions per image",
                 "No boxes: the AI answers your questions about each image (LabelMe flags), e.g. to keep or set "
                 "aside images.",
                 {"mode": "classify", "flags": [{"name": "question_1", "description": "true when …"}]}),
}


def from_template(key: str, project: str) -> dict:
    cfg = normalize(TEMPLATES[key][2])
    cfg["project"] = project or "my-project"
    return cfg


# ------------------------------------------------------------------------------------------------ hashing
def run_key(cfg: dict) -> str:
    """What decides the AI's answers: changes here re-annotate; connection / limits / output do not."""
    import hashlib
    import json
    keep = {k: cfg.get(k) for k in ("classes", "flags", "mode", "roi", "vlm", "detector", "segment", "checks")}
    keep["references"] = cfg.get("references")
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]
