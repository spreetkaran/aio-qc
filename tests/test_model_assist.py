# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Model Assist in Image QC: the models suggest labels on the shown image (labels already there are not repeated),
click to accept (undoable), right-click to drop, accept all, suggest on every image (cached), and SAM outlines for
drawn boxes and accepted box suggestions — models and SAM answered by stand-ins."""
import os
import re

import pytest
from PyQt6.QtCore import QRectF

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401
from .test_explorer import _fake_models

CALLS = []


def _predict(images, ids, conf=0.05, batch=8, classes=None, cancel=None):
    CALLS.append(list(images))
    out = {}
    for p in images:
        i = int(re.search(r"_(\d+)\.", os.path.basename(p)).group(1))
        x = 40 + i % 50
        boxes = [[x + 1, 41, x + 99, 119, 0.9, 0],                # the labelled car: not suggested again
                 [250, 10, 300, 60, 0.8, 0],                       # a car nobody labelled
                 [200, 150, 240, 190, 0.2, 0]]                     # too unsure
        out[p] = {"w": 320, "h": 240, "by_model": {m: list(boxes) if m == "ma" else [] for m in ids}, "geo": {}}
    return out


@pytest.fixture
def assist(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    _fake_models(w.session.root, tmp_path)
    svc = w.inference
    monkeypatch.setattr(svc, "python", "/fake/python")
    monkeypatch.setattr(svc, "ensure_loaded", lambda specs, *a, **k: None)
    monkeypatch.setattr(svc, "predict", _predict)
    monkeypatch.setattr(svc, "segment", lambda model, image, boxes: [[[250, 10], [300, 10], [300, 60], [255, 58]]])
    CALLS.clear()
    w.settings.setValue("assist/auto", False)
    w.settings.setValue("assist/sam", False)
    w.settings.setValue("assist/conf", 0.35)
    w.file_dock.step(0) if hasattr(w.file_dock, "step") else None
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    return w


def _shapes(w):
    return w.session.load(w.current_key).shapes


def test_suggest_accept_undo_drop(assist, qtbot):
    from AIO_QC.widgets.model_assist import new_suggestions
    w = assist
    a = w.assist
    n0 = len(_shapes(w))
    w.a.assist_suggest.trigger()
    qtbot.waitUntil(lambda: bool(a.items), timeout=20000)
    assert [round(x["box"][0]) for x in a.items] == [250] and a.items[0]["label"] == "car"     # only the new one
    assert len(w.canvas.suggestions) == 1 and w.a.assist_accept_all.isEnabled()
    # a click on it (the canvas finds it under the mouse)
    item = w.canvas.suggestions[0][0]
    pos = w.canvas.mapFromScene(item.sceneBoundingRect().center())
    assert w.canvas.suggestion_at(pos) == 0
    w.canvas.suggestionClicked.emit(0)
    assert len(_shapes(w)) == n0 + 1 and _shapes(w)[-1]["label"] == "car" and not a.items
    assert not w.canvas.suggestions
    w.inline.undo()
    assert len(_shapes(w)) == n0
    # again: from the cache (no new model call), then dropped with a right-click
    calls = len(CALLS)
    a.suggest()
    qtbot.waitUntil(lambda: bool(a.items), timeout=20000)
    assert len(CALLS) == calls
    w.canvas.suggestionDismissed.emit(0)
    assert not a.items and len(_shapes(w)) == n0
    # accept all (only those at or above the confidence)
    a.suggest()
    qtbot.waitUntil(lambda: bool(a.items), timeout=20000)
    assert a.accept_all() == 1 and len(_shapes(w)) == n0 + 1 and not a.items
    a.clear()
    # the pure part: duplicates of a label, and two models finding one object
    preds = [("car", 0.9, (0, 0, 10, 10), None, "a"), ("car", 0.8, (0, 0, 10, 11), None, "b"),
             ("plate", 0.7, (0, 0, 10, 10), None, "a"), ("car", 0.3, (50, 50, 60, 60), None, "a")]
    got = new_suggestions(preds, [("plate", (1, 1, 10, 10))], 0.35)
    assert [(x["label"], x["model"]) for x in got] == [("car", "a")]


def test_every_image_and_sam(assist, qtbot, monkeypatch):
    w = assist
    a = w.assist
    w.a.assist_auto.trigger()                                   # on: this image and every next one
    qtbot.waitUntil(lambda: bool(a.items), timeout=20000)
    first = w.current_key
    w.a.next.trigger()
    qtbot.waitUntil(lambda: w.current_key != first and bool(a.items) and a.key == w.current_key, timeout=20000)
    w.a.assist_auto.trigger()
    # SAM: an accepted box suggestion becomes an outline
    w.a.assist_sam.trigger()
    assert a.sam_on
    n0 = len(_shapes(w))
    a.accept(0)
    qtbot.waitUntil(lambda: len(_shapes(w)) == n0 + 1, timeout=20000)
    assert _shapes(w)[-1]["shape_type"] == "polygon" and len(_shapes(w)[-1]["points"]) == 4
    # … and so does a box drawn by hand
    monkeypatch.setattr(w.inline, "_choose_label", lambda *a_, **k: "car")
    w.inline.on_box(QRectF(240, 5, 70, 60))
    qtbot.waitUntil(lambda: len(_shapes(w)) == n0 + 2, timeout=20000)
    assert _shapes(w)[-1]["shape_type"] == "polygon"
    # SAM fails: the box stays a box, and the reason is said plainly
    def broken(*_a, **_k):
        raise RuntimeError("No module named 'ultralytics'")
    monkeypatch.setattr(w.inference, "segment", broken)
    w.inline.on_box(QRectF(10, 150, 40, 40))
    qtbot.waitUntil(lambda: len(_shapes(w)) == n0 + 3, timeout=20000)
    assert _shapes(w)[-1]["shape_type"] == "rectangle"
    assert "missing the package" in w.statusBar().currentMessage()
    w.a.assist_sam.trigger()
    assert not a.sam_on


def test_no_model_says_what_to_do(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets import errors as ER
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    w.assist.suggest()
    assert ER.LAST[-1].action == "model_setup" and "model" in ER.LAST[-1].what.lower()
