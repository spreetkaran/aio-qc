# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Fine corrections from the keyboard: a click on an edge / corner handle makes the arrows move just that side
(Shift ×10); Esc goes back to moving the whole box. Written to the file like any edit (one write, undoable)."""
import json

from PyQt6.QtCore import QPoint, Qt

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _box(w, i=0):
    data = json.loads(open(w.session.by_key[w.current_key].ann_path, encoding="utf-8").read())
    (x1, y1), (x2, y2) = data["shapes"][i]["points"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def test_arrows_move_just_the_clicked_edge(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None)
    c = w.canvas
    rect_idx = next(i for i, g in c.index_to_group.items() if c._stype(g.shape_data) == "rectangle")
    assert c.select_shape(rect_idx)
    before = c._norm_pts(c.selected_group, c._shape_pts(c.selected_group))
    x1, y1 = before[0]
    x2, y2 = before[1]
    right = next(h for h in c._handles if h.data(0) == ("e", 1))          # the right edge's handle
    vp = c.mapFromScene(right.pos())
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(vp.x(), vp.y()))
    assert c._active_handle == ("e", 1)
    for _ in range(3):
        qtbot.keyClick(c, Qt.Key.Key_Left)                                 # tighten the right side by 3 px
    qtbot.keyClick(c, Qt.Key.Key_Up)                                       # ignored: an edge moves across itself
    qtbot.keyClick(c, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)  # and 10 more
    c._flush_nudge()
    qtbot.waitUntil(lambda: abs(_box(w, rect_idx)[2] - (x2 - 13)) < 0.01, timeout=3000)
    bx = _box(w, rect_idx)
    assert abs(bx[0] - x1) < 0.01 and abs(bx[1] - y1) < 0.01 and abs(bx[3] - y2) < 0.01   # other sides untouched
    qtbot.keyClick(c, Qt.Key.Key_Escape)                                   # back to moving the whole box
    assert c._active_handle is None and c.selected_group is not None
    qtbot.keyClick(c, Qt.Key.Key_Down)
    c._flush_nudge()
    qtbot.waitUntil(lambda: abs(_box(w, rect_idx)[1] - (y1 + 1)) < 0.01, timeout=3000)
    assert abs(_box(w, rect_idx)[3] - (y2 + 1)) < 0.01
