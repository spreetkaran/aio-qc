# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Qt-free core of AIO QC: dataset discovery/IO, geometry and QC checks.

Everything in here is importable (and testable) without PyQt6.
"""
from .dataset import (
    DatasetType, Sample, Annotation, AnnotationError,
    LabelMeAdapter, YoloAdapter, detect_dataset_types, open_dataset,
)
from .io import load_json, atomic_write_text, Trash, image_size, IMAGE_EXTENSIONS
