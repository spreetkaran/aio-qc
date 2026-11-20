# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Frame Extractor window: module-level helpers shared by its modules (also re-exported by AIO_QC.tools.frame_extractor.app)."""
from PyQt6.QtWidgets import (
    QLabel,
)
from PyQt6.QtCore import Qt


MOUSE_HELP = [
    ("Video", [("Mouse wheel", "Previous / next frame"),
               ("Ctrl/Cmd + wheel, pinch", "Smooth zoom around the cursor"),
               ("Ctrl/Cmd + = / + / numpad +  ·  - / numpad -", "Zoom in / out (animated)"),
               ("Ctrl/Cmd + 0 · Ctrl/Cmd + 1", "Fit to window (button lights up while fitted) · actual pixels 100%"),
               ("Double-click", "Toggle fit ↔ 100% at the cursor"),
               ("Middle-drag, or drag when zoomed", "Pan"), ("Double-click", "Fit to window"),
               ("ROI mode: drag", "Draw a new ROI; drag inside to move, edges/corners to resize"),
               ("Right-click", "Context menu (save, ROI, marks, zoom)"), ("Drop a file", "Load it (several: batch queue)")]),
    ("Seekbar & timeline strip", [("Click / drag seekbar", "Jump / scrub (keyframe previews while dragging with PyAV)"),
               ("Hover", "Thumbnail preview"), ("Blue bars", "Sharpness (red = below the blur threshold)"),
               ("Orange line", "Motion (% change, or encoder kB/frame in keyframe mode)"),
               ("White ticks", "Scene changes"), ("Bottom row", "Model: purple = detections, amber = uncertain/disagree"),
               ("Right-click strip", "Blur threshold, model track, re-analyze")]),
    ("Extraction markers (top lane of the strip)", [("M", "Drop / remove a marker at the playhead (also while playing)"),
               ("Drag a marker", "Move it"), ("Double-click / right-click a marker", "Jump / snap, extract, remove"),
               ("Green / blue / hollow", "Pending / saved / suggested"),
               ("Ctrl+E · Ctrl+Shift+E", "Extract pending · auto-pick sharpest per window → snap → extract")]),
    ("Position display (under the seekbar)", [("Click", "Type a position: 1:23.5 · 00:01:23:12 · 1234f · 42% · "
                                                         "+5s · -30f · in · out · end (↑/↓ step while typing)"),
               ("Scroll on a unit", "Nudge hours / minutes / seconds / frames (Shift ×10)"),
               ("Drag on a unit", "Scrub by that unit"), ("Format chip / Ctrl+Shift+T", "Time · Timecode (SMPTE) · Frames · Seconds · Remaining"),
               ("Right-click", "Formats, copy time / timecode / frame, timecode overlay on the video")]),
    ("Command line", [("aio-qc extract -h", "Batch extraction without the GUI"),
                      ("… extract v.mp4 -o out --frames @marks.txt --model best.pt --env <conda env>",
                       "Save picked frames with pre-annotations from a .pt model"),
                      ("aio-qc probe VIDEO", "Decoder / VFR / rotation info"),
                      ("… testset OUT --add a.mp4 b.mp4 frames/ --every 1s", "Test set of unique frames from many sources"),
                      ("… eval DATA -m old.pt -m new.onnx --env yolo --report comparison",
                       "Score weights: mAP, P/R, confusion matrix, HTML report"),
                      ("… clip VIDEO 1:00 1:30 -o out.mp4 [--exact]", "Cut a clip (lossless, or frame-exact)")]),
    ("Search & clips", [("Ctrl+F", "Search frames by text (CLIP) or by example; click a result to jump"),
                          ("Ctrl+Shift+X", "Export the in → out range as a clip")]),
]
class _ElidedLabel(QLabel):
    """The status bar's left line: a long path is shortened in the middle to fit, the whole text in the tooltip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""

    def set_full(self, text):
        self._full = text
        self._fit()

    def _fit(self):
        w = max(20, self.width() - 12)
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, w))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()
FX_ADVANCED = {"Benchmark decoders for this video", "Copy current time", "Cycle time format", "Go to in point",
               "Go to out point", "Clear in/out", "Next flagged sample", "Previous flagged sample",
               "Playback statistics", "Export annotated video", "Export clip (in → out)", "Reset window layout",
               "Models", "Track ROI", "Detect", "Open log folder", "Previous bookmark", "Next bookmark",
               "Slower", "Faster", "Reopen last video"}

