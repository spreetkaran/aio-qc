# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""F8 / Shift+F8: step through a check's findings from the image — each opens with its shape selected and the
keyboard on the image, so arrows / Del / relabel fix it at once."""
import json
import os


from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _with_dups(root, files=(0, 2, 5)):
    FX.labelme(root)
    for i in files:
        p = os.path.join(root, f"cam_frame_{i:04d}.json")
        d = json.load(open(p))
        (x1, y1), (x2, y2) = d["shapes"][0]["points"]
        d["shapes"].append({"label": "car", "points": [[x1 + 2, y1 + 1], [x2 + 2, y2 + 1]],
                            "shape_type": "rectangle", "flags": {}, "group_id": None, "description": ""})
        json.dump(d, open(p, "w"))
    return root


def _run(w, qtbot, key):
    w.qc_actions[key].trigger()
    qtbot.waitUntil(lambda: key in w.open_dialogs, timeout=3000)
    d = w.open_dialogs[key]
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None or not d.worker.isRunning(), timeout=10000)
    qtbot.wait(50)
    return d


def test_f8_steps_through_findings_on_the_image(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, _with_dups(tmp_path / "ds"))
    d = _run(w, qtbot, "duplicates")
    qtbot.waitUntil(lambda: len(d.finding_list()) == 3, timeout=10000)
    items = d.finding_list()
    assert len(items) == 3
    w.step_finding(1, source=d)                      # F8 in the check's window
    qtbot.waitUntil(lambda: w.current_key == items[0][0] and w.canvas.selected_group is not None, timeout=3000)
    assert w.canvas.selected_group.index == items[0][1]
    w.step_finding(1)                                # F8 on the image
    qtbot.waitUntil(lambda: w.current_key == items[1][0], timeout=3000)
    qtbot.waitUntil(lambda: w.canvas.selected_group is not None and w.canvas.selected_group.index == items[1][1],
                    timeout=3000)
    w.step_finding(-1)                               # Shift+F8
    qtbot.waitUntil(lambda: w.current_key == items[0][0], timeout=3000)
    w.step_finding(1)
    w.step_finding(1)
    qtbot.waitUntil(lambda: w.current_key == items[2][0], timeout=3000)
    w.step_finding(1)                                # past the end: stays, says so
    assert w.current_key == items[2][0] and "Last finding" in w.statusBar().currentMessage()


def test_double_click_puts_the_keyboard_on_the_image(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtWidgets import QApplication
    w = _open(shell, qtbot, _with_dups(tmp_path / "ds"))
    d = _run(w, qtbot, "duplicates")
    qtbot.waitUntil(lambda: len(d.finding_list()) == 3, timeout=10000)
    d.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is d, timeout=2000)
    d.on_row_activated(d.visible_rows()[1])          # as a double-click on a result
    qtbot.waitUntil(lambda: w.canvas.selected_group is not None, timeout=3000)
    qtbot.waitUntil(lambda: w.canvas.hasFocus(), timeout=2000)
    assert w._finding_src is d and w._finding_pos == 1


def test_fixing_a_finding_does_not_skip_the_next(shell, qtbot, tmp_path):  # noqa: F811
    """Delete the duplicate on the image: the check rescans that file, the row leaves, F8 goes to the next one."""
    w = _open(shell, qtbot, _with_dups(tmp_path / "ds"))
    d = _run(w, qtbot, "duplicates")
    qtbot.waitUntil(lambda: len(d.finding_list()) == 3, timeout=10000)
    items = d.finding_list()
    w.step_finding(1, source=d)
    qtbot.waitUntil(lambda: w.current_key == items[0][0] and w.canvas.selected_group is not None, timeout=3000)
    w.canvas.shapeDeleteRequested.emit(w.canvas.selected_group.index)        # Del on the image
    qtbot.waitUntil(lambda: len(d.finding_list()) == 2, timeout=10000)
    w.step_finding(1)
    qtbot.waitUntil(lambda: w.current_key == items[1][0], timeout=3000)
