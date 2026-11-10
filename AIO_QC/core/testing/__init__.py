# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Lab (Qt-free): test YOLO models on videos and image folders.

  metrics   - mAP / P / R / F1 / confusion matrix / error analysis (YOLO-val compatible)
  dataset   - ground truth: YOLO and LabelMe folders, class-name mapping
  testset   - reusable test sets built from many videos (unique frames) and folders
  runner    - batched multi-model inference with a persistent prediction cache
  videoeval - label-free video analysis: stability, detections over time, model agreement
  charts    - SVG charts shared by the app and the reports
  report    - HTML reports (summary / detailed / comparison / video) + CSV / JSON
  annotate  - annotated-video and clip export
  search    - frame search by text (CLIP) or example
"""
