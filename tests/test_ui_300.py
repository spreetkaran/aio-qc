# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""3.0 layout (breadcrumb, tab hints, palette field, panel rows, canvas read-outs) and the flagged-files list."""
import csv
from pathlib import Path

from PyQt6.QtCore import QPointF

from AIO_QC.core.flagged_export import FlaggedItem, write_list
from AIO_QC.widgets.panels import COUNT_ROLE, HINT_ROLE

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_layout_pieces(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 30))
    qtbot.waitUntil(lambda: w.current_key is not None)
    # breadcrumb: dataset › file + a pill
    assert "ds" in w.dataset_label.path.text() and "cam_frame_0000" in w.dataset_label.path.text()
    assert "LabelMe" in w.dataset_label.pill.text() and "30 files" in w.dataset_label.pill.text()
    # tab hints and the command palette field
    assert w.tabs.bar.tabData(0) == "Alt 1"                     # painted beside the name in the mono font
    from PyQt6.QtWidgets import QLabel
    assert any("Search tools & actions" in lb.text() for lb in w.palette_field.findChildren(QLabel))
    # Labels: name + count badge + "same as" hint; Annotations: "#n" on the right
    rows = {w.unique_labels.item(i).text(): w.unique_labels.item(i) for i in range(w.unique_labels.count())}
    assert rows["car"].data(COUNT_ROLE) == "30" and "same as" in (rows["Car "].data(HINT_ROLE) or "")
    assert w.labels_title.count.text() == str(len(rows))
    assert w.annotations_list.item(0).data(COUNT_ROLE) == "#0" and w.annotations_title.count.text() == "3"
    assert w.files_title.count.text() == "30"
    # canvas read-outs: file, flag note, pointer / selected box
    assert w.hud.file.isVisibleTo(w.canvas) and w.hud.file.main.text() == "cam_frame_0000.jpg"
    assert "320 × 240" in w.hud.file.extra.text() and "JPG" in w.hud.file.extra.text()
    w.session.set_review([w.current_key], "flag", "box too loose")
    assert w.hud.flag.isVisibleTo(w.canvas) and "box too loose" in w.hud.flag.main.text()
    w.session.set_review([w.current_key], None)
    assert not w.hud.flag.isVisibleTo(w.canvas)
    w.canvas.select_shape(0)
    assert "box 100 × 80" in w.hud.pos.extra.text()
    w.canvas.report_cursor(w.canvas.mapFromScene(QPointF(12, 30)))
    assert __import__("re").search(r"x +12", w.hud.pos.extra.text()) and "box" in w.hud.pos.extra.text()
    # the read-outs can be switched off; the status bar keeps the facts
    w.hud.set_enabled(False)
    assert not w.hud.file.isVisibleTo(w.canvas)
    assert w.statusBar().image.text.text() == "320 × 240px"
    # the zoom slider is optional
    assert not w._slider_action.isVisible()
    # mock-up chrome: mode switch without icons, segmented Drag / Select, − 37 % + group, files grouped by split
    from PyQt6.QtWidgets import QToolButton
    assert all(b.icon().isNull() for b in w.mode_switch.buttons.values())
    assert w.tool_segment.buttons[0].defaultAction() is w.a.drag
    assert [b.text() for b in w.zoom_group.findChildren(QToolButton) if b.text() in ("−", "+")] == ["−", "+"]
    assert w.open_dir_btn.button.hint.text() == "Ctrl+O"


def test_flagged_list_csv_and_txt(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    items = [FlaggedItem("a", [root / "a.json", root / "a.jpg"], "flag", "box, too loose", "2026-10-30 10:00"),
             FlaggedItem("b", [root / "b.json"], "fixed", "", "")]
    n = write_list(items, root, tmp_path / "ds_flagged.csv", "relative")
    assert n == 2
    with open(tmp_path / "ds_flagged.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["file", "image", "status", "note", "flagged_at"]
    assert rows[1] == ["a.json", "a.jpg", "flag", "box, too loose", "2026-10-30 10:00"]
    assert write_list(items, root, tmp_path / "x.txt", "names", True) == 2
    assert (tmp_path / "x.txt").read_text().splitlines()[0] == "a.json\tbox, too loose"


def test_flagged_dialog_defaults_and_live_counts(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.batch_shots import BatchShotsDialog
    from AIO_QC.widgets.flagged_dialog import FlaggedFilesDialog
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 6))
    keys = sorted(w.session.by_key)
    w.session.set_review(keys[:3], "flag")
    d = FlaggedFilesDialog(w)
    d.show()
    assert Path(d.txt.text()).name == "ds_flagged.csv" and d.fmt.currentData() == "csv"
    d.fmt.setCurrentIndex(d.fmt.findData("txt"))
    assert Path(d.txt.text()).name == "ds_flagged.txt"
    b = BatchShotsDialog(w)
    b.show()
    assert "(3)" in d.scope.currentText() and "(3)" in b.scope.currentText()
    # show only flagged files, then clear a mark: both windows follow
    w.toggle_flagged_filter()
    w.session.set_review([keys[0]], None)
    qtbot.waitUntil(lambda: "(2)" in d.scope.itemText(0) and "(2)" in b.scope.itemText(0), timeout=3000)
    w.settings.setValue("flagged/list_format", "csv")
    d.close()
    b.close()


def test_file_groups_fold_and_labels_swatch(shell, qtbot, tmp_path):  # noqa: F811
    root = tmp_path / "LPR"
    FX.labelme(root / "train", 6)
    FX.labelme(root / "val", 3)
    w = _open(shell, qtbot, root)
    qtbot.waitUntil(lambda: w.current_key is not None)
    dock = w.file_dock
    assert not dock.split_box.isVisible()                       # the groups replace the split filter
    n = dock.proxy.rowCount()
    val = [s for s in w.session.samples if s.split == "val"][-1]       # not the group's header row
    dock.set_collapsed("val", True)
    assert dock.proxy.rowCount() == n - 2                        # the header row of 'val' stays
    first_val = next(s.key for s in dock.model.samples if s.split == "val")
    assert dock.proxy_row_of(first_val) >= 0
    assert dock.select_key(val.key)                              # going to a folded file opens its group
    assert "val" not in dock.proxy.collapsed and dock.proxy.rowCount() == n
    # the delegate folds on a header click
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import Qt
    idx = dock.proxy.index(dock.proxy_row_of(first_val), 0)
    rect = dock.view.visualRect(idx)
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(rect.left() + 20, rect.top() + 5),
                     QPointF(rect.left() + 20, rect.top() + 5), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    from PyQt6.QtWidgets import QStyleOptionViewItem
    opt = QStyleOptionViewItem()
    opt.rect = rect
    dock.view.itemDelegate().editorEvent(ev, dock.proxy, opt, idx)
    assert "val" in dock.proxy.collapsed
    # Labels: the swatch switches a label off and on (no check box)
    lst = w.unique_labels
    it = lst.item(0)
    r = lst.visualItemRect(it)
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(r.left() + 20, r.center().y()),
                     QPointF(r.left() + 20, r.center().y()), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    o = QStyleOptionViewItem()
    o.rect = r
    before = it.checkState()
    lst.itemDelegate().editorEvent(ev, lst.model(), o, lst.indexFromItem(it))
    assert it.checkState() != before
