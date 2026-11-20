# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
An Auto-Annotate setup: every choice of a run in one small JSON file, reused batch after batch (Qt-free).

Nothing in it is about one project: the classes, the rule, the folder grouping and the part size are all the
project's own. Unknown keys are ignored and missing ones take their defaults, so setups saved by older or newer
versions still load.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

STAGES = ("frames", "similar", "annotate", "rules", "package")
STAGE_TITLES = {
    "frames": "Frames",
    "similar": "Similar frames",
    "annotate": "Annotate",
    "rules": "Keep rule",
    "package": "Package",
}
STAGE_HINTS = {
    "frames": "Extract frames from the videos (or take an image folder)",
    "similar": "Set aside frames that look like one already kept",
    "annotate": "Your trained model or the AI draws the boxes: a LabelMe / YOLO file per frame",
    "rules": "Keep only frames with the classes you need",
    "package": "Folders, parts of N files, zip files to upload",
}
# quick choices: which stages run
PRESETS = {
    "Frames only": ("frames",),
    "Frames + similar removed": ("frames", "similar"),
    "Frames + annotations": ("frames", "similar", "annotate"),
    "Everything": STAGES,
}
SIMILAR_PRESETS = {      # name: (pHash Hamming distance, global %, local %) — the thumbnail check must agree too
    "strict": (4, 1.5, 6.0),
    "balanced": (6, 2.5, 9.0),
    "loose": (10, 4.0, 14.0),
}
SETUP_SUFFIX = ".aasetup.json"
FORMAT = 1


@dataclass
class Recipe:
    name: str = ""
    stages: list = field(default_factory=lambda: list(STAGES[:4]))
    # ---------------------------------------------------------------- frames
    source: str = "videos"            # videos | images
    recursive: bool = True
    include: str = ""                 # file-name patterns, e.g. "*cam1*; *cam2*" (empty = all)
    exclude: str = ""
    mode: str = "step"                # step | sharpest
    step: int = 3                     # every Nth frame / the sharpest of every N
    min_sharpness: float = 0.0        # 0 = blur filter off
    skip_repeats: str = "near"        # off | exact | near   (within one video, as it is decoded)
    across_videos: bool = False       # also skip frames already extracted from another video / an earlier run
    ext: str = ".jpg"
    quality: int = 95
    name_template: str = "{video}_frame_{frame}"
    workers: int = 0                  # 0 = automatic
    import_mode: str = "link"         # images: link (hard link, no extra space) | copy
    # ---------------------------------------------------------------- similar frames
    similar: str = "balanced"         # strict | balanced | loose
    prefer_sharper: bool = True       # of two similar frames, keep the sharper one
    # ---------------------------------------------------------------- annotate
    annotator: str = "model"          # model (your trained model) | ai (AI annotate, a project config.yaml)
    ai_config: str = ""               # the AI annotate project file
    output_format: str = "labelme"    # model annotator: labelme | yolo | both
    model: str = ""
    imgsz: int = 0                    # 0 = the model's own (read from the file)
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    device: str = "auto"              # auto | cpu | 0 | 1 ... (CUDA device)
    half: bool = False
    batch: int = 8
    classes: dict = field(default_factory=dict)   # model class id (str) -> {"keep": bool, "name": str, "conf": float}
    min_box: int = 0                  # boxes smaller than this (px, either side) are left out
    labelme_version: str = "6.3.1"
    image_data: bool = False          # embed the image in the JSON (imageData)
    description: str = "none"         # none | conf  (what goes in each box's description)
    also_yolo: bool = False           # also write YOLO .txt + classes.txt next to the frames
    existing: str = "update"          # update: redo files this run wrote with other settings; keep: never touch
                                      # files that exist; replace: rewrite every file
    # ---------------------------------------------------------------- keep rule
    keep_when: str = "detections"     # always | detections | any_of | all_of
    keep_classes: list = field(default_factory=list)
    min_boxes: int = 1                # with "detections": at least this many boxes
    others: str = "move"              # move (to extra_images/) | keep
    # ---------------------------------------------------------------- package
    group_by: str = "none"            # none | video | pattern
    group_pattern: str = ""           # regular expression on the file name; group 1 if it has one
    split_size: int = 200             # 0 = no parts
    part_names: str = "number"        # number: part_001 · start: part_0, part_200 (index of the first file)
    zip_parts: bool = True
    keep_folders: bool = True         # keep the part folders next to the zip files
    package_images: bool = True       # images go with the JSONs (always, unless imageData is embedded)
    yolo_layout: str = "beside"       # beside: labels next to images · split: images/ and labels/ in each part
    package_files: str = "link"       # link: hard links (no extra disk space; copies when the drive can't) | copy

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = asdict(self)
        d["format"] = FORMAT
        return d

    @classmethod
    def from_dict(cls, d) -> "Recipe":
        names = {f.name for f in fields(cls)}
        r = cls(**{k: v for k, v in (d or {}).items() if k in names})
        r.stages = [s for s in STAGES if s in (r.stages or [])]
        return r

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Recipe":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ meaning
    def runs(self, stage: str) -> bool:
        return stage in self.stages

    def writes_labelme(self) -> bool:
        return self.output_format in ("labelme", "both")

    def writes_yolo(self) -> bool:
        return self.output_format in ("yolo", "both") or bool(self.also_yolo)

    def class_entry(self, cid) -> dict:
        e = self.classes.get(str(int(cid))) or {}
        return {"keep": bool(e.get("keep", True)), "name": str(e.get("name") or ""),
                "conf": float(e["conf"]) if e.get("conf") not in (None, "", 0) else None}

    def output_name(self, cid, model_name: str) -> str | None:
        """The label written for a model class (None = left out)."""
        e = self.class_entry(cid)
        return (e["name"] or model_name) if e["keep"] else None

    def problems(self, model_names: dict | None = None) -> list[str]:
        """What stops this setup from running, in plain words (empty = fine)."""
        out = []
        if not self.stages:
            out.append("Choose at least one step.")
        if self.mode not in ("step", "sharpest"):
            out.append(f"Unknown frame mode {self.mode!r}.")
        if int(self.step) < 1:
            out.append("The frame step must be 1 or more.")
        if self.runs("annotate") and self.annotator == "ai":
            import os
            if not self.ai_config:
                out.append("Annotate: choose or make an AI annotate project.")
            elif not os.path.isfile(self.ai_config):
                out.append(f"Annotate: the AI annotate project file was not found ({self.ai_config}).")
        if self.runs("annotate") and self.annotator != "ai":
            if not self.model:
                out.append("Annotate: choose a model file.")
            if not 0 < float(self.conf) <= 1:
                out.append("Annotate: confidence must be between 0 and 1.")
            if not 0 < float(self.iou) <= 1:
                out.append("Annotate: IoU must be between 0 and 1.")
            from AIO_QC.tools.frame_extractor.core.autoannotate.labelme_io import VERSIONS
            if self.labelme_version not in VERSIONS:
                out.append(f"Annotate: LabelMe version must be {' or '.join(VERSIONS)}.")
            if model_names and self.annotator != "ai":
                labels = [self.output_name(c, n) for c, n in model_names.items()]
                if not any(labels):
                    out.append("Annotate: every class is left out — keep at least one.")
        if self.runs("rules") and self.keep_when in ("any_of", "all_of") and not self.keep_classes:
            out.append("Keep rule: choose the classes a frame must have (or keep every frame).")
        if self.runs("rules") and model_names is not None and self.keep_when in ("any_of", "all_of") and \
                self.annotator != "ai":
            labels = {self.output_name(c, n) for c, n in model_names.items()} - {None}
            missing = [c for c in self.keep_classes if c not in labels]
            if missing and labels:
                out.append(f"Keep rule: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not among the "
                           f"classes written ({', '.join(sorted(labels))}).")
        if self.runs("package"):
            if self.group_by == "pattern":
                import re
                try:
                    re.compile(self.group_pattern or "")
                except re.error as exc:
                    out.append(f"Package: the folder pattern is not a valid regular expression ({exc}).")
                if not self.group_pattern:
                    out.append("Package: type the pattern that names the folders (or choose another grouping).")
            if int(self.split_size) < 0:
                out.append("Package: the part size cannot be negative.")
            if self.package_files not in ("link", "copy"):
                out.append("Package: files are either linked or copied.")
            if not self.package_images and not self.image_data:
                out.append("Package: without the images the JSONs need imageData — turn one of them on.")
        return out
