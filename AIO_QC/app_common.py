# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image QC: the tool catalogue (QC_TOOLS, GROUPS …) and small helpers shared by the main window's modules."""
from __future__ import annotations

import sys
from dataclasses import dataclass


from AIO_QC.widgets import (
    AmbiguityDialog, ClassRemovalDialog, DimAreaDialog, DuplicateDialog, EmptyDialog, ImageDuplicatesDialog, OutlierDialog, SmartFilterDialog, StatsDialog, ValidationDialog,
)
from AIO_QC.widgets.crop_gallery import CropGalleryDialog
from AIO_QC.widgets.class_gallery import ClassGalleryDialog
from AIO_QC.widgets.sequence_dialog import SequenceDialog
from AIO_QC.widgets.similarity_dialog import SimilarityDialog
from AIO_QC.widgets.text_search_dialog import TextSearchDialog
from AIO_QC.widgets.claude_review_dialog import ClaudeReviewDialog
from AIO_QC.widgets.review_grid import ReviewGridDialog
from AIO_QC.widgets.views_dialog import ViewsDialog
from AIO_QC.widgets.handoff_dialog import ExportBatchDialog, ImportBatchDialog
from AIO_QC.widgets.split_health_dialog import SplitHealthDialog
from AIO_QC.widgets.name_clashes_dialog import NameClashesDialog
from AIO_QC.widgets.classification_check_dialog import ClassificationCheckDialog
from AIO_QC.widgets.report_dialog import ReportDialog
from AIO_QC.widgets.guidelines import GuidelinesDialog
from AIO_QC.widgets.model_check_dialog import ModelCheckDialog
from AIO_QC.widgets.sampling_dialog import SamplingDialog
from AIO_QC.widgets.coverage_dialog import CoverageDialog
from AIO_QC.widgets.issues_dialog import IssuesDialog
from AIO_QC.widgets.rules_dialog import RulesDialog
from AIO_QC.widgets.image_quality_dialog import ImageQualityDialog
from AIO_QC.widgets.convert_split_dialog import ConvertSplitDialog  # noqa: E402
from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog
from AIO_QC.widgets.ocr_dialog import OcrCheckDialog
from AIO_QC.widgets.text_labels_dialog import TextLabelsDialog
from AIO_QC.widgets.train_size_dialog import TrainSizeDialog
from AIO_QC.widgets.pairs_dialog import PairsDialog
from AIO_QC.widgets.batch_shots import BatchShotsDialog
from AIO_QC.widgets.qc_presets_dialog import QCPresetsDialog
from AIO_QC.core.diag import ignored as _ignored

# macOS: Cmd+M minimises the window in front (the system convention), so Label Mapping moves to Cmd+Shift+Option+M
LABEL_MAP_KEY = "Ctrl+Shift+Alt+M" if sys.platform == "darwin" else "Ctrl+M"

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    def natsorted(it, key=None):
        return sorted(it, key=key)


@dataclass(frozen=True)
class QCTool:
    key: str
    title: str
    icon: str           # icon file name in AIO_QC/icons
    shortcut: str
    group: str          # a key of GROUPS
    dialog: type | None
    tip: str
    action: str = ""    # tools that are plain actions (e.g. Label Mapping) reuse that QAction


# Tabs = groups, in workflow order: integrity → names → shapes → model-assisted → sign-off.

GROUPS = [   # (key, tab title, question) — tab icon: icons/group-<key>.png
    ("dataset", "Dataset", "Are the files and images sound?"),
    ("labels", "Labels", "Are the label NAMES right?"),
    ("shapes", "Shapes", "Are the annotations themselves right?"),
    ("models", "Models", "Where do models and labels disagree?"),
    ("review", "Review", "Is it done, and is it good enough?"),
]

TOOL_NOT_HERE = {       # "what this tool deliberately does NOT do → which tool does"
    "stats": "Nothing is edited here.",
    "validation": "Label names → Labels tab · empty files → Empty · duplicates → Duplicate Images / Shapes.",
    "ambiguity": "Renaming / merging → Label Mapping.",
    "remove_class": "Renaming → Label Mapping.",
    "crop_gallery": "Whole-label renames → Label Mapping.",
    "class_gallery": "Renaming / merging whole classes → Classes.",
    "outliers": "Ranges you define → Dim / Area.",
    "dim_area": "Statistical outliers → Outliers.",
    "train_size": "Sizes in the original image (any ranges you define) → Dim / Area.",
    "model_setup": "Running the comparison → Model Check.",
    "model_check": "Setup → Models & Environment.",
    "sampling": "Per-file marks → the review keys S / X.",
    "smart_filter": "Shape-level ranges → Dim / Area.",
    "coverage": "Setting marks → the review keys S / X.",
    "fix_editor": "Drawing inside AIO QC — geometry is edited in labelme / labelImg.",
    "inbox": "Finding issues itself → it collects what the other tools found.",
    "export": "Changing annotations → it writes a separate, clean copy.",
    "data_prep": "Checking annotations → the QC tabs (Preparation works on folders, before a dataset is open).",
    "rules": "Per-class size statistics → Dim / Area and Outliers.",
    "image_quality": "Annotation problems → Validation.",
    "ocr_check": "Class / geometry comparison with a detector → Model Check.",
    "text_labels": "Reading the text from the image itself → OCR Check.",
    "sequence": "Single images without neighbours → Validation / Model Check.",
    "similarity": "Detector-based comparison (boxes, missed objects) → Model Check.",
    "review_grid": "Editing shapes → on the image (select a shape) or Fix in Editor.",
    "views": "Shape sizes / positions → Dim / Area (its results can be pinned as a fixed-list view).",
    "name_clashes": "Duplicates (identical images) → Duplicate Images · a label without its image → Image–Label Pairs.",
    "split_health": "Building new splits from scratch → Convert & Split · duplicates inside one split → Duplicate "
                    "Images.",
    "guidelines": "Checking labels against the rules → the Shapes / Labels tools (the rule shows next to them).",
    "report": "Finding new problems → it summarises what the tools already found.",
    "handoff_export": "Editing here → on the image, or Fix in Editor for your own editor.",
    "improve": "Training itself → Train Model · the checking itself → Model Check (it runs it for you).",
    "qc_presets": "Fixing → each check's own window (open a finding from the report).",
    "handoff_import": "Files changed by another program in the dataset itself → picked up live (History shows them).",
}


# Left out of the menus and QC tabs by View ▸ Simple Menus (still in Ctrl+K, shortcuts still work)
IMAGE_ADVANCED = {
    "Guidelines", "Text Labels", "Remove / Re-order Classes", "Outliers", "Sequence Check", "Dim / Area",
    "Training Size", "Relationship Rules", "Models & Environment", "OCR Check", "Visual Similarity", "Model Check",
    "Testing", "QC Sampling", "Fix in Editor", "Coverage Map", "Smart Filter", "Saved Views", "Flagged Screenshots",
    "Export Hand-off Batch", "Import Hand-off Batch", "Split Health", "Name Clashes", "Classification Check", "Recommended Checks", "Storage",
    "Background Tasks", "Logs", "Reset App", "Annotations → Data Studio", "Show Source Video in Frame Extractor",
    "Keep Previous Zoom", "Keep Previous Brightness/Contrast", "Pin QC Panel", "Model Predictions", "Next Finding",
    "Previous Finding", "Next Shape", "Previous Shape", "Add as Good Example", "Add as Bad Example",
    "Open Trash Folder", "Next Panel", "Previous Panel",
    "Export Review Report", "Save Flagged Files",
}

# Single source of truth for tabs, menus and shortcuts.
QC_TOOLS = [
    # ---- DATASET
    # first: get the folders ready (works without a dataset open — its own action, also in Tools)
    QCTool("data_prep", "Preparation", "tool-prep", "Ctrl+Alt+P", "dataset", None,
           "Get data ready before (or after) QC: unzip archives, import other formats, combine & "
           "flatten folders, combine YOLO datasets, add new data or classes to a dataset (matched by image content), "
           "orphans, integrity, duplicates, EXIF, resize, pick images to label — every run previewed, logged and "
           "undoable. No dataset needs to be open", action="data_prep"),
    QCTool("stats", "Dataset Stats", "tool-stats", "Ctrl+Shift+S", "dataset", StatsDialog,
           "Label balance per split, image profile (resolution, aspect, lighting …)"),
    QCTool("validation", "Validation", "tool-validation", "Ctrl+Shift+V", "dataset", ValidationDialog,
           "File & geometry integrity: is every file really LabelMe / YOLO, shape structure, class ids, points, "
           "image size"),
    QCTool("pairs", "Image–Label Pairs", "tool-pairs", "Ctrl+Alt+K", "dataset", PairsDialog,
           "Every image with its label file: JSONs / labels without their image, images without a label, labels in "
           "the wrong folder or case, files that are not LabelMe / YOLO at all — with the likely partner and a fix"),
    QCTool("name_clashes", "Name Clashes", "tool-clashes", "Ctrl+Alt+C", "dataset", NameClashesDialog,
           "File names that would collide when flattened or copied to Windows: the same name in different "
           "sub-folders, upper / lower case only, two images on one label file, names Windows refuses — identical "
           "or different, and Rename… with presets or your own pattern (checked as you type, undoable)"),
    QCTool("empty", "Empty Annotations", "tool-empty", "Ctrl+Shift+E", "dataset", EmptyDialog,
           "Annotation files without shapes (background images)"),
    QCTool("image_dups", "Duplicate Images", "tool-image-dups", "Ctrl+Shift+I", "dataset", ImageDuplicatesDialog,
           "Exact / near-duplicate images and split leakage"),
    # ---- LABELS (names)
    QCTool("image_quality", "Image Quality", "tool-quality", "Ctrl+Shift+Q", "dataset", ImageQualityDialog,
           "Corrupt / truncated images, EXIF rotation and size disagreements; blur and exposure (information)"),
    QCTool("split_health", "Split Health", "tool-splits", "Ctrl+Alt+S", "dataset", SplitHealthDialog,
           "Every class in every split with enough examples, val / test like train, no near-identical images across "
           "splits (leakage) — with suggested file moves (undoable)"),
    QCTool("ambiguity", "Label Ambiguities", "tool-ambiguity", "Ctrl+Shift+A", "labels", AmbiguityDialog,
           "Find label-name variants, misspellings and unknown labels"),
    QCTool("label_map", "Label Mapping", "tool-label-map", LABEL_MAP_KEY, "labels", None,
           "The one place to rename / merge labels (view-only or written)", action="label_map"),
    QCTool("cls_check", "Classification Check", "tool-class", "Ctrl+Alt+D", "labels", ClassificationCheckDialog,
           "Classification datasets: images without a class (and why), several classes on a one-class dataset, flag "
           "values like 1 or \"true\", name variants (Cat / cat), classes with few images or missing from a split "
           "— with fixes"),
    # no key of its own: Ctrl+Shift+G (Crop Gallery) opens it in a classification dataset
    QCTool("class_gallery", "Class Gallery", "tool-class-gallery", "", "labels", ClassGalleryDialog,
           "Classification datasets: one class at a time as a wall of images — most suspicious first (your "
           "classifier's findings, then the least typical by look), move images with 1 … 9 or by dragging, Space "
           "marks the rest OK; confusions, look-alikes in other classes, compare two classes, sample check and "
           "sign-off"),
    QCTool("guidelines", "Guidelines", "tool-guidelines", "Ctrl+Alt+G", "labels", GuidelinesDialog,
           "Write the labelling rule per class / QC rule with ✓ good and ✗ bad example crops; shown next to the "
           "selected shape and in the tools, exported for annotators"),
    QCTool("text_labels", "Text Labels", "tool-text-labels", "Ctrl+Alt+T", "labels", TextLabelsDialog,
           "Check the text stored in your annotations without OCR: formats and known texts learned from the dataset, "
           "your validators (optional), what other frames of the same object say — with the correction applied in "
           "one click"),
    QCTool("remove_class", "Remove / Re-order Classes", "tool-remove-class", "Ctrl+Shift+R", "labels", ClassRemovalDialog,
           "Delete classes from every file; YOLO: renumber ids and rewrite the class map"),
    # ---- SHAPES (annotations)
    QCTool("crop_gallery", "Crop Gallery", "tool-crop-gallery", "Ctrl+Shift+G", "shapes", CropGalleryDialog,
           "Every instance of a label as a grid of crops — spot wrong labels and junk boxes fast"),
    QCTool("duplicates", "Duplicate & Overlapping Shapes", "tool-overlaps", "Ctrl+D", "shapes", DuplicateDialog,
           "Every pair of shapes worth a look, each listed once: exact copies, one object marked twice, boxes "
           "overlapping too much, a shape inside another of its class — fix each on the image (F8 steps through "
           "them)"),
    QCTool("outliers", "Outliers", "tool-outliers", "Ctrl+Shift+O", "shapes", OutlierDialog,
           "Shapes whose size is statistically unusual for their class"),
    QCTool("sequence", "Sequence Check", "tool-sequence", "Ctrl+Shift+J", "shapes", SequenceDialog,
           "Consecutive frames compared with their neighbours: objects missing for a frame, label flips, size / "
           "position spikes, attribute changes, one-frame objects — missing boxes can be added"),
    QCTool("dim_area", "Dim / Area", "tool-dim-area", "Ctrl+Shift+D", "shapes", DimAreaDialog,
           "Shapes inside width / height / area ranges you define"),
    QCTool("train_size", "Training Size", "tool-train-size", "Ctrl+Alt+Z", "shapes", TrainSizeDialog,
           "Every object measured at the training size (longer side = imgsz) and after scale augmentation: objects "
           "too small to learn, and per class the imgsz that keeps 95 % of them big enough"),
    # ---- MODELS
    QCTool("rules", "Relationship Rules", "tool-rules", "Ctrl+Shift+K", "shapes", RulesDialog,
           "How annotations relate, for any classes: parent / part (inside, not larger, parts per parent), must "
           "touch, counts, requires, never together, overlaps, size, shape type, tilt, attributes, groups — or let "
           "AIO QC suggest rules from the data"),
    QCTool("model_setup", "Models & Environment", "tool-model-setup", "Ctrl+Alt+N", "models", None,
           "Python env, models, class maps, device, imgsz, ONNX / TensorRT", action="model_setup"),
    QCTool("ocr_check", "OCR Check", "tool-ocr", "Ctrl+Shift+T", "models", OcrCheckDialog,
           "Read the text in text-class boxes (PaddleOCR / RapidOCR / EasyOCR / Tesseract from your environment) and "
           "check it: validators, readability, box fit, stored text, identity leakage"),
    QCTool("similarity", "Visual Similarity", "tool-similar", "Ctrl+Shift+Y", "models", SimilarityDialog,
           "Find mislabels by appearance with DINOv3 / SigLIP2 / CLIP (folder, zip or download): likely "
           "mislabels, atypical objects, look-alikes with different labels"),
    QCTool("words", "Search in Words", "search", "Ctrl+Alt+W", "models", TextSearchDialog,
           "Search images or objects in words (“truck at night in rain”) and check every label against a one-line "
           "description of its class — local, with SigLIP2 / CLIP"),
    QCTool("claude_review", "Claude Review", "tool-ai-review", "Ctrl+Alt+V", "models", ClaudeReviewDialog,
           "Claude looks at the image with its labels and says what is wrong, in words — missing objects, wrong "
           "classes, loose boxes, text, guidelines (Anthropic API; live or as a half-price batch)"),
    QCTool("model_check", "Model Check", "tool-model-check", "Ctrl+Shift+M", "models", ModelCheckDialog,
           "Findings + analysis: report card, confusion, error types, box quality, slices, label-error estimate"),
    QCTool("train", "Train Model", "tool-train", "Ctrl+Alt+B", "models", None,
           "Train an Ultralytics YOLO model (detection or segmentation) on a YOLO dataset, in your own Python "
           "environment: the dataset checked first, image size suggested, presets, live charts, stop / pause / "
           "resume, per-class results, then Add to Models", action="train"),
    QCTool("improve", "Improve Model", "tool-improve", "Ctrl+Alt+U", "models", None,
           "One guided loop: check a trained model on this dataset (where it is weak, and why), flag the images where "
           "model and labels disagree, pick new images it is unsure about, fine-tune — every round in a history, so "
           "you see whether it got better", action="improve"),
    QCTool("augment", "Image Augmentation", "tool-augment", "Ctrl+Alt+A", "models", None,
           "Build an augmentation recipe on your own images — geometry, colour, weather, blur, noise, occlusion, plus "
           "albumentations / imgaug extras — with labels that follow; then train with it in memory (no files) or "
           "write an augmented copy of the dataset", action="augment"),
    QCTool("testing", "Testing", "tool-testing", "Ctrl+Alt+L", "models", None,
           "Measure and compare detection models (formerly the Frame Extractor's Model Lab): this dataset, a test "
           "set from videos, or videos only — leaderboard, charts, confusion, per class, failures, plots, A/B, "
           "video behaviour, reports; label fixes go through AIO QC", action="testing"),
    # ---- REVIEW
    QCTool("inbox", "Issues Inbox", "tool-inbox", "Ctrl+I", "review", IssuesDialog,
           "Every tool's findings in one place, a risk score per file, and one review queue — worst first"),
    QCTool("export", "Convert & Split", "tool-convert", "Ctrl+Shift+X", "review", ConvertSplitDialog,
           "One window, steps in a side pane: write this dataset as YOLO (detection / segmentation), LabelMe, COCO, "
           "masks, VOC, CVAT, YOLO OBB, DOTA, YOLO pose, crops or class folders — the classes, their order and names, "
           "one set / its splits / a new balanced train / val / test split (duplicates, video frames and OCR "
           "identities kept together, rare cases in train), flat or with sub-folders; replace and undo "
           "(Ctrl+Alt+Y: on YOLO)"),
    QCTool("sampling", "QC Sampling", "tool-sampling", "Ctrl+Shift+P", "review", SamplingDialog,
           "Review a random / stratified sample and estimate the dataset error rate"),
    QCTool("fix_editor", "Fix in Editor", "tool-editor", "Ctrl+Alt+O", "review", EditorSessionDialog,
           "Open flagged files in labelme / labelImg; every save is checked, written back and marked Fixed"),
    QCTool("coverage", "Coverage Map", "tool-coverage", "Ctrl+Shift+C", "review", CoverageDialog,
           "Every file as a square coloured by review state — see what is QC'ed, click to open"),
    QCTool("smart_filter", "Smart Filter", "tool-smart-filter", "Ctrl+Shift+F", "review", SmartFilterDialog,
           "Build a file subset to review by labels and object counts"),
    QCTool("review_grid", "Review Grid", "tool-review-grid", "Ctrl+Shift+W", "review", ReviewGridDialog,
           "Whole images with their annotations, a page at a time: X the bad ones, Space marks the rest OK"),
    QCTool("views", "Saved Views", "tool-saved-filter", "Ctrl+Shift+B", "review", ViewsDialog,
           "Named file queries (labels, counts, review state, findings, names …) with live counts and a "
           "plain-language explanation; show in the file list or review in the grid"),
    QCTool("screenshots", "Flagged Screenshots", "screenshot", "Ctrl+Shift+Alt+E", "review", BatchShotsDialog,
           "A screenshot of every flagged (or selected) file, drawn as Image QC shows it — the same spot as now or "
           "the whole image, with a caption strip (file, mark, flag note) and sketches — in the background"),
    QCTool("qc_presets", "QC Presets", "tool-presets", "Ctrl+Alt+Q", "review", QCPresetsDialog,
           "Named sets of checks with their settings and limits — run one on every new delivery and get a pass / fail "
           "report (HTML + CSV) to hand to the annotators"),
    QCTool("report", "QC Report", "tool-report", "Ctrl+Alt+R", "review", ReportDialog,
           "One click: a self-contained HTML / PDF of class balance, review coverage, findings, the QC sample "
           "estimate, the riskiest files (thumbnails) and what changed since the last report"),
    QCTool("handoff_export", "Export Hand-off Batch", "tool-handoff", "Ctrl+Alt+E", "review", ExportBatchDialog,
           "Send flagged / risky files to an annotator: their own folder with notes, findings and hint images"),
    QCTool("handoff_import", "Import Hand-off Batch", "import", "Ctrl+Alt+I", "review", ImportBatchDialog,
           "Merge a returned batch: each file compared three ways (as sent · as returned · now), conflicts shown, "
           "what changed listed, every write backed up"),
]
MAX_RECENT = 10

PIXMAP_CACHE_BYTES = 512 * 1024 * 1024     # decoded images kept for instant next / previous (ceiling)


def tool_for(tool: QCTool, session) -> QCTool:
    """The tool as it runs on this dataset: on a classification dataset a check written for boxes opens its
    classifier version (whole images and their classes) — widgets/cls_tools.py."""
    if session is None or tool.dialog is None or not getattr(session, "is_classification", False):
        return tool
    sup = tool.dialog.SUPPORTED
    if sup is None or session.type in sup:
        return tool
    from dataclasses import replace
    from AIO_QC.widgets.cls_tools import CLS_VARIANTS
    d = CLS_VARIANTS.get(tool.key)
    if d is None or (d.SUPPORTED is not None and session.type not in d.SUPPORTED):
        return tool
    return replace(tool, dialog=d)


def tool_supported(tool: QCTool, session) -> bool:
    t = tool_for(tool, session)
    return t.dialog is None or t.dialog.SUPPORTED is None or (session is not None and session.type in t.dialog.SUPPORTED)


def _find_ffmpeg(user_path=""):
    """ffmpeg for Testing's annotated-video export (the Frame Extractor's lookup: its setting, PATH, bundled)."""
    try:
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
        return find_ffmpeg(user_path)
    except Exception as _exc:
        _ignored(_exc)
        return None

