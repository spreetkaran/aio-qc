# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image QC: several shapes selected (Ctrl / Shift + click, Ctrl+A), moved / deleted / relabelled together, copied,
cut and pasted — undo one by one."""
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core import inline_edit as IE

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _shapes(w, key=None):
    return w.session.load(key or w.current_key).shapes


def _click(c, idx, mods=Qt.KeyboardModifier.NoModifier):
    r = c.index_to_group[idx].shape_item.sceneBoundingRect()
    pts = c.shapes[idx]["points"]
    p = c.mapFromScene(*((pts[0][0], pts[0][1] + 6) if c.shapes[idx].get("shape_type") == "circle" else
                         (r.center().x(), r.top() + 4)))            # (the wheel: its part below the car box)
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, mods, QPoint(int(p.x()), int(p.y())))


@pytest.fixture(autouse=True)
def _ask_again():
    yield
    from AIO_QC import utils
    utils.app_settings().setValue("edit/confirm_delete", True)       # these tests switch the question off


def _ready(shell, qtbot, root):  # noqa: F811
    w = _open(shell, qtbot, root)
    w.resize(1300, 850)
    w.settings.setValue("edit/confirm_delete", False)
    qtbot.waitUntil(lambda: w.canvas.editable and len(w.canvas.shape_groups) >= 3, timeout=10000)
    w.canvas.setMode(w.canvas.SELECT)
    return w, w.canvas


def test_multi_select_with_ctrl_shift_and_select_all(shell, qtbot, tmp_path):  # noqa: F811
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    assert c.select_all() == 0 and not w.a.select_all_shapes.isEnabled()      # Ctrl+A needs a selected shape first
    _click(c, 0)
    assert c.selected_indices() == [0] and w.a.copy_shapes.isEnabled()
    _click(c, 2, Qt.KeyboardModifier.ControlModifier)
    assert c.selected_indices() == [0, 2] and c.selected_group.index == 2
    _click(c, 2, Qt.KeyboardModifier.ControlModifier)                        # Ctrl+click again: out
    assert c.selected_indices() == [0]
    c.add_shape(1)                                                           # Shift+click adds
    assert c.selected_indices() == [0, 1]
    w.inline.select_all()
    assert c.selected_indices() == [0, 1, 2]
    c.select_shape(1)                                                        # a plain select: just that one
    assert c.selected_indices() == [1]
    c.escape()
    assert c.selected_indices() == []
    for act, key in ((w.a.copy_shapes, "Ctrl+C"), (w.a.cut_shapes, "Ctrl+X"), (w.a.paste_shapes, "Ctrl+V"),
                     (w.a.select_all_shapes, "Ctrl+A")):
        assert [k.toString() for k in act.shortcuts()] == [key]         # only these (Qt: ⌘ on macOS)
        assert act in w.actions() and act.shortcutContext() == Qt.ShortcutContext.WindowShortcut


def test_copy_paste_and_undo_one_by_one(shell, qtbot, tmp_path):  # noqa: F811
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    key = w.current_key
    n0 = len(_shapes(w))
    c.select_many([0, 1])
    assert w.inline.copy() == 2
    assert w.inline.paste() == 2
    qtbot.waitUntil(lambda: c.selected_indices() == [n0, n0 + 1], timeout=3000)   # the pasted ones are selected
    sh = _shapes(w)
    assert len(sh) == n0 + 2 and [x["label"] for x in sh[n0:]] == ["car", "plate"]
    assert sh[n0]["points"] == sh[0]["points"]                                    # exactly on top of its source
    assert sh[n0 + 1]["description"] == sh[1]["description"] and sh[n0 + 1]["shape_type"] == "polygon"
    assert w.inline.paste() == 2                                                  # again: the same place again
    assert _shapes(w)[n0 + 2]["points"] == sh[0]["points"]
    for left in (n0 + 3, n0 + 2, n0 + 1, n0):                                     # undo: one shape at a time
        w.inline.undo()
        assert len(_shapes(w, key)) == left
    w.inline.redo()
    assert len(_shapes(w, key)) == n0 + 1
    # into another image: same place, no offset
    w.inline.undo()
    other = next(k for k in w.session.by_key if k != key)
    w.open_sample(other)
    qtbot.waitUntil(lambda: w.current_key == other and c.editable, timeout=5000)
    m0 = len(_shapes(w))
    assert w.inline.paste() == 2
    assert _shapes(w)[m0]["points"] == sh[0]["points"] and len(_shapes(w, key)) == n0


def test_delete_move_relabel_and_cut_together(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    key = w.current_key
    before = _shapes(w)
    c.select_many([0, 1])
    QTest.keyClick(c, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)        # both move 10 px, one write
    c._flush_nudge()
    qtbot.wait(50)
    sh = _shapes(w)
    assert sh[0]["points"][0][0] == before[0]["points"][0][0] + 10
    assert sh[1]["points"][0][0] == before[1]["points"][0][0] + 10
    assert sh[2]["points"] == before[2]["points"]
    qtbot.waitUntil(lambda: c.selected_indices() == [0, 1], timeout=3000)
    w.inline.undo()
    assert _shapes(w)[1]["points"] == before[1]["points"]                         # one step for both
    c.select_many([0, 2])
    w.inline.on_relabel(0, 2)                                                     # both get class 2
    names = w.inline.classes()
    assert [_shapes(w)[i]["label"] for i in (0, 2)] == [names[1], names[1]]
    w.inline.undo()
    c.select_many([0, 2])
    assert w.inline.cut() == 2
    assert [x["label"] for x in _shapes(w)] == ["plate"]
    assert w.inline.paste() == 2                                                  # cut + paste: back in place
    labs = sorted(x["label"] for x in _shapes(w))
    assert labs == ["car", "plate", "wheel"]
    monkeypatch.setattr(QMessageBox, "exec", lambda self: int(QMessageBox.StandardButton.Yes))
    w.settings.setValue("edit/confirm_delete", True)
    c.select_many([0, 1, 2])
    QTest.keyClick(c, Qt.Key.Key_Delete)
    assert _shapes(w, key) == []
    w.inline.undo()
    assert len(_shapes(w, key)) == 3


def test_paste_respects_the_target_format(shell, qtbot, tmp_path):  # noqa: F811
    """YOLO: only its classes; a circle has no YOLO form; a box from a bigger image is brought inside."""
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    c.select_many([0, 1, 2])
    w.inline.copy()
    clip = dict(w.inline._clip)
    w2 = _open(shell, qtbot, FX.yolo(tmp_path / "y"))
    qtbot.waitUntil(lambda: w2.canvas.editable, timeout=10000)
    from PyQt6.QtWidgets import QApplication
    QApplication.clipboard().clear()
    w2.inline._clip = clip
    n0 = len(_shapes(w2))
    assert w2.inline.paste() == 1                                                 # the car box; plate polygon would
    sh = _shapes(w2)                                                              # mix det / seg, wheel is a circle
    assert len(sh) == n0 + 1 and sh[-1]["label"] == "car"
    assert IE.fit_points([[300, 200], [400, 260]], 320, 240) == [[220.0, 180.0], [320.0, 240.0]]
    assert IE.fit_points([[500, 500], [900, 900]], 320, 240) is None               # larger than the image, all outside
    assert IE.fit_points([[500, 500], [500.2, 500.3]], 320, 240) == [[319.8, 239.7], [320.0, 240.0]]   # tiny: kept
    assert IE.fit_points([[-50, 10], [700, 60]], 320, 240) == [[0.0, 10.0], [320.0, 60.0]]


def test_mouse_shift_click_and_dragging_the_selection(shell, qtbot, tmp_path):  # noqa: F811
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    before = _shapes(w)
    _click(c, 0)
    _click(c, 1, Qt.KeyboardModifier.ShiftModifier)
    assert c.selected_indices() == [0, 1] and c.selected_group.index == 1
    vp = c.viewport()
    r = c.index_to_group[0].shape_item.sceneBoundingRect()
    a = c.mapFromScene(r.center().x(), r.top() + 6)                    # on the car, a selected (not primary) shape
    b = QPoint(a.x() + 40, a.y() + 20)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for k in range(1, 6):
        QTest.mouseMove(vp, QPoint(a.x() + 8 * k, a.y() + 4 * k))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    qtbot.wait(80)
    sh = _shapes(w)
    dx = sh[0]["points"][0][0] - before[0]["points"][0][0]
    dy = sh[0]["points"][0][1] - before[0]["points"][0][1]
    assert dx > 5 and dy > 2
    assert abs(sh[1]["points"][0][0] - before[1]["points"][0][0] - dx) < 0.02       # the same step for both
    assert sh[2]["points"] == before[2]["points"]
    qtbot.waitUntil(lambda: c.selected_indices() == [0, 1], timeout=3000)
    _click(c, 0)                                                      # a plain click in the selection: just it
    assert c.selected_indices() == [0]


def _key(w, k):
    from PyQt6.QtWidgets import QApplication
    QTest.keyClick(QApplication.focusWidget() or w, k, Qt.KeyboardModifier.ControlModifier)


def _activate(w, qtbot):
    win = w.window()
    win.show()
    win.activateWindow()
    qtbot.waitUntil(lambda: win.isActiveWindow(), timeout=5000)


def test_drag_mode_and_annotations_panel(shell, qtbot, tmp_path):  # noqa: F811
    """What a user does first: Drag mode (the start mode) and the Annotations panel — Ctrl / Shift + click there,
    Ctrl+A / C there, Ctrl+V after moving on in the file list."""
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    _activate(w, qtbot)
    c.setMode(c.DRAG)
    _click(c, 0)
    assert c.selected_indices() == []                                    # Drag mode: a plain click never selects
    _click(c, 0, Qt.KeyboardModifier.ControlModifier)                    # … a Ctrl+click does (and switches to Select)
    _click(c, 2, Qt.KeyboardModifier.ControlModifier)
    assert c.selected_indices() == [0, 2] and c.mode == c.SELECT
    lst = w.annotations_list
    assert sorted(lst.row(it) for it in lst.selectedItems()) == [0, 2]   # the panel shows both
    c.clearSelection()
    c.setMode(c.DRAG)
    lst.setFocus()

    def lclick(row, mods=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, mods, lst.visualItemRect(lst.item(row)).center())
    _key(w, Qt.Key.Key_A)
    assert c.selected_indices() == [] and not lst.selectedItems()        # Ctrl+A needs one selected first
    lclick(1)
    lclick(2, Qt.KeyboardModifier.ControlModifier)
    assert c.selected_indices() == [1, 2] and c.selected_group.index == 2
    lclick(2, Qt.KeyboardModifier.ControlModifier)
    assert c.selected_indices() == [1]
    lclick(0, Qt.KeyboardModifier.ShiftModifier)                       # a list's Shift+click: the range of rows
    assert c.selected_indices() == [0, 1, 2]                             # from the last clicked one (2)
    lclick(0)
    assert c.selected_indices() == [0]
    _key(w, Qt.Key.Key_A)
    assert c.selected_indices() == [0, 1, 2] and len(lst.selectedItems()) == 3
    w.inline._clip = None
    _key(w, Qt.Key.Key_C)                                                # focus in the panel: still copies shapes
    assert w.inline._clip is not None and len(w.inline._clip["shapes"]) == 3
    fl = w.file_dock.view
    fl.setFocus()
    first = w.current_key
    QTest.keyClick(fl, Qt.Key.Key_Down)
    qtbot.waitUntil(lambda: w.current_key != first, timeout=5000)
    n0 = len(_shapes(w))
    _key(w, Qt.Key.Key_V)                                                # focus in the file list: pastes
    assert len(_shapes(w)) == n0 + 3
    w.inline.undo()
    assert len(_shapes(w)) == n0 + 2                                     # one by one
    from PyQt6.QtWidgets import QLineEdit
    le = next(x for x in w.findChildren(QLineEdit) if x.isVisible() and x.isEnabled() and not x.isReadOnly())
    le.setFocus()
    le.setText("hello")
    le.deselect()
    before = c.selected_indices()
    _key(w, Qt.Key.Key_A)                                                # a text field keeps its own Ctrl+A
    assert le.selectedText() == "hello" and c.selected_indices() == before
    le.clear()


def test_ctrl_click_on_a_neighbour_selects_it(shell, qtbot, tmp_path):  # noqa: F811
    """With one polygon selected, Ctrl+click on another shape selects that shape — never a new vertex on the
    selected polygon's outline (adjacent polygons share edges)."""
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    poly = next(i for i, s in enumerate(c.shapes) if s.get("shape_type", "polygon") == "polygon")
    other = next(i for i in range(len(c.shapes)) if i != poly and i in c.index_to_group)
    c.select_shape(poly)
    before = [list(p) for p in _shapes(w)[poly]["points"]]
    _click(c, other, Qt.KeyboardModifier.ControlModifier)
    assert sorted(c.selected_indices()) == sorted([poly, other])
    assert _shapes(w)[poly]["points"] == before


def test_relabel_keys_work_wherever_the_focus_is(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """L / F2 (choose) and 1 … 9 (that class) change the selected shape's label from the image, the Files list,
    the Annotations list or the window — but type normally in a text box."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLineEdit
    w, c = _ready(shell, qtbot, FX.labelme(tmp_path / "ds"))
    key = w.current_key
    names = w.inline.classes()
    asked = []
    monkeypatch.setattr(w.inline, "_choose_label", lambda *a, **k: asked.append(1) or "truck")
    for target in (c, w.file_dock, w.annotations_list, w):
        c.select_shape(0)
        _shapes(w)                                                              # (reloaded)
        before = len(asked)
        target.setFocus()
        qtbot.keyClick(target, Qt.Key.Key_L)
        assert len(asked) == before + 1, target
        assert _shapes(w, key)[0]["label"] == "truck"
        w.inline.undo()
    c.select_shape(0)
    w.annotations_list.setFocus()
    qtbot.keyClick(w.annotations_list, Qt.Key.Key_2)
    assert _shapes(w, key)[0]["label"].strip() == names[1].strip()
    w.inline.undo()
    c.select_shape(0)
    qtbot.keyClick(w.file_dock, Qt.Key.Key_F2)
    assert _shapes(w, key)[0]["label"] == "truck"
    w.inline.undo()
    # several selected: all of them, one undo step
    c.select_many([0, 1])
    qtbot.keyClick(w, Qt.Key.Key_L)
    assert [x["label"] for x in _shapes(w, key)[:2]] == ["truck", "truck"]
    w.inline.undo()
    assert _shapes(w, key)[0]["label"] != "truck" and _shapes(w, key)[1]["label"] != "truck"
    # a text box keeps the key
    box = QLineEdit(w)
    box.show()
    box.setFocus()
    c.select_shape(0)
    n = len(asked)
    w._relabel_key(0)
    assert box.text() == "l" and len(asked) == n
    box.close()
    box.setParent(None)
    # nothing selected: said, nothing changes
    c.setFocus()
    c.clearSelection()
    w._relabel_key(0)
    assert "Select a shape first" in w.statusBar().currentMessage()
