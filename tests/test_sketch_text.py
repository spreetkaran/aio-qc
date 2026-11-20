# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Sketch text: templates (fields filled in, kept in the settings), the text size — in Sketch on Image's Text tool
(new text, editing a placed one, undo / redo) and in auto-sketch (batch text and size, a shape's own text)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QSettings

from AIO_QC.core import autosketch as AS
from AIO_QC.core import sketch_text as ST

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_expand_and_store(tmp_path):
    assert ST.expand("{label} #{n} in {file}", label="plate", n=3, file="a.jpg") == "plate #3 in a.jpg"
    assert ST.expand("{unknown} {label}", label="car") == "{unknown} car"        # unknown fields stay as typed
    assert ST.expand("{label}") == ""
    st = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    assert ST.load_templates(st) == ST.DEFAULT_TEMPLATES
    ST.save_templates(st, ["Loose box", "  ", "Loose box", "Wrong {label}"])
    assert ST.load_templates(st) == ["Loose box", "Wrong {label}"]
    ST.save_templates(st, ST.DEFAULT_TEMPLATES)                                  # back to the built-in list
    assert ST.load_templates(st) == ST.DEFAULT_TEMPLATES


def test_auto_text_size_and_own_text():
    shapes = [{"label": "car", "shape_type": "rectangle", "points": [[100, 100], [200, 180]]},
              {"label": "plate", "shape_type": "rectangle", "points": [[400, 300], [460, 330]]}]
    texts = AS.arrow_texts(shapes, [0, 1], "Check {label}", {1: "Loose box"}, "a.jpg")
    assert texts == ["Check car", "Loose box"]
    assert AS.arrow_texts(shapes, [0], "", {}) is None
    boxes = [AS.shape_box(s["points"], "rectangle") for s in shapes]
    fs = {k: [e["fs"] for e in AS.plan(800, 600, boxes, style="arrow", labels=texts, text_scale=v)
              if e["t"] == "text"] for k, v in ST.TEXT_SIZES}
    assert fs["S"][0] < fs["M"][0] < fs["L"][0] < fs["XL"][0]
    ring = AS.plan(800, 600, boxes[:1], style="ring", labels=["Look"])
    assert [e["t"] for e in ring] == ["ellipse", "text"]                          # ring only: the text above it


def test_manual_text_size_templates_edit_undo(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.widgets import sketch_text_dialog as D
    from AIO_QC.widgets.sketch import text_px
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None and bool(w.canvas.shape_groups), timeout=10000)
    sk = w.sketch
    sk.toggle()
    sk.set_tool("text")
    assert sk.slider.maximum() == 400 and sk.size_lbl.text().startswith("Aa ")
    sk.slider.setValue(64)
    assert sk.tsize == 64
    seen = {}

    def fake_exec(self):                                          # pick the "{label}" template on a car click
        seen["fields"] = dict(self.fields)
        seen["size"] = self.value()
        self._use(self.templates.item(0))
        if seen.get("resize"):
            self.size.setValue(seen["resize"])
            self.edit.setPlainText("Loose box")
        return 1
    monkeypatch.setattr(D.TextDialog, "exec", fake_exec)
    (x1, y1), (x2, y2) = w.current_ann.shapes[0]["points"]          # the car (labels are hidden in sketch mode)
    pt = QPointF((x1 + x2) / 2, y1 + 6)
    sk._text_at(pt)
    el = sk.els[-1]
    assert seen["fields"]["label"] == "car" and seen["size"] == 64
    assert el["text"] == "car" and text_px(el) == 64
    # click on that text again: edit it — new size and text, the old one hidden
    seen["resize"] = 120
    sk._text_at(QPointF(el["pts"][0][0] + 5, el["pts"][0][1] - 10))
    assert len(sk.els) == 2 and sk.els[0].get("gone") and sk.els[1]["text"] == "Loose box"
    assert text_px(sk.els[1]) == 120
    sk.undo()                                                     # the edit undone: the first text is back
    assert len(sk.els) == 1 and not sk.els[0].get("gone")
    sk.redo()
    assert sk.els[0].get("gone") and sk.els[1]["text"] == "Loose box"
    sk.toggle()


def test_old_text_elements_keep_their_size():
    from AIO_QC.widgets.sketch import text_px
    assert text_px({"t": "text", "w": 8, "text": "x"}) == 40                      # drawings saved before 3.0
    assert text_px({"t": "text", "w": 8, "fs": 90, "text": "x"}) == 90


# ---------------------------------------------------------------------------------------------------- looks
def test_auto_colour_contrasts_with_the_background():
    for bg in [(0.95, 0.95, 0.92), (0.05, 0.06, 0.08), (0.9, 0.2, 0.2), (0.2, 0.4, 0.9), (0.5, 0.5, 0.5)]:
        for key in ST.LOOK_KEYS:
            r = ST.resolve_look(key, bg, 0.0, "#ff3b30")
            if ST.look(key)["fg"] == "auto":
                against = r["bg"] if r["ts"] == "chip" else bg
                cands = ["#ffffff", "#111111"] if r["ts"] == "chip" else ST.AUTO_TEXT
                best = max(ST.contrast(c, against) for c in cands)
                assert ST.contrast(r["c"], against) >= min(best, 4.5) - 1e-9, (key, bg, r)   # the best there is
                assert ST.contrast(r["c"], against) >= 3.0, (key, bg, r)          # WCAG for large text
            if r["ts"] == "chip" and ST.look(key)["bg"] == "auto":
                assert ST.contrast(r["bg"], bg) >= 2.0, (key, bg, r)              # the chip stands out
    busy = ST.resolve_look("auto_outline", (0.5, 0.5, 0.5), 0.6)
    assert busy["c"] in ("#ffffff", "#111111")
    assert ST.look("no such look")["key"] == ST.DEFAULT_LOOK


def _img(color):
    from PyQt6.QtGui import QColor, QImage
    img = QImage(400, 300, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    return img


def test_styled_text_on_light_and_dark_images(qapp):
    from PyQt6.QtGui import QColor, QImage, QPainter
    from AIO_QC.widgets.sketch import draw_el, styled_text_el, text_rect
    for bg, want in (("#f2f2ee", "#111111"), ("#101418", "#ffffff")):
        img = _img(bg)
        el = styled_text_el("Loose box", (40, 120), 48, "auto_outline", "#ff3b30", img)
        assert el["c"] == want and el["ts"] == "outline" and el["fs"] == 48
        chip = styled_text_el("Loose box", (40, 120), 48, "auto_chip", "#ff3b30", img)
        assert chip["ts"] == "chip" and ST.contrast(chip["bg"], bg) > 2
        for key in ST.LOOK_KEYS:                                   # every look draws something, inside its rect
            e = styled_text_el("Check", (40, 120), 40, key, "#ff3b30", img)
            out = QImage(img)
            p = QPainter(out)
            draw_el(p, e)
            p.end()
            r = text_rect(e).toAlignedRect().intersected(out.rect())
            changed = sum(out.pixelColor(x, y) != QColor(bg)
                          for x in range(r.left(), r.right(), 3) for y in range(r.top(), r.bottom(), 3))
            assert changed > 20, key


def test_dialog_preview_follows_the_inputs(qapp):
    from AIO_QC.widgets.sketch_text_dialog import TextDialog
    st = QSettings()
    d = TextDialog(st, None, text="", size=40, look="auto_outline", image=_img("#f2f2ee"), point=(30, 100))
    assert d.preview.el is None
    d.edit.setPlainText("Wrong label")
    assert d.preview.el["text"] == "Wrong label" and d.preview.el["c"] == "#111111"
    d.size.setValue(80)
    assert d.preview.el["fs"] == 80
    d.look_box.setCurrentIndex(d.look_box.findData("alert_chip"))
    assert d.preview.el["ts"] == "chip" and d.preview.el["bg"] == "#e5332a"
    assert d.preview.grab().width() > 0                          # paints


def test_sketch_toolbar_sits_in_the_tab_row(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    c = w.canvas
    size = c.size()
    w.sketch.toggle()
    bar = w.sketch.bar
    assert bar.docked and w.tabs.mode_bar is bar and not w.tabs.row.isVisible()
    assert not bar.geometry().intersects(c.viewport().geometry()) or bar.parent() is not c   # not over the image
    qtbot.wait(50)
    assert c.size() == size                                     # the image did not move or shrink
    w.tabs.open_tab_keyboard(0)                                 # no drawer while sketching
    assert not w.tabs.is_open()
    w.sketch.toggle()
    assert not bar.docked and w.tabs.mode_bar is None and w.tabs.row.isVisible()
    w.roi.toggle() if hasattr(w, "roi") else None
    if hasattr(w, "roi"):
        assert w.roi.bar.docked
        w.roi.toggle()
        assert w.tabs.row.isVisible()


def test_tab_not_selected_when_the_drawer_is_closed(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    rb = w.tabs
    rb.set_pinned(False)
    painted = []
    orig = rb.bar.currentIndex

    def spy():
        painted.append(rb._open or rb.pinned)
        return orig()
    rb.open_tab_keyboard(1)
    assert rb.is_open()
    rb.close_drop(animate=False)
    img_closed = rb.bar.grab().toImage()
    rb.open_tab_keyboard(1)
    img_open = rb.bar.grab().toImage()
    rb.close_drop(animate=False)
    assert img_open != img_closed                               # the open tab is drawn selected, the closed is not
    assert rb.bar.grab().toImage() == img_closed


def test_text_never_cut_off_at_the_edge(qapp):
    from AIO_QC.widgets.sketch import text_rect
    from AIO_QC.widgets.sketch_text_dialog import TextDialog
    d = TextDialog(QSettings(), None, text="Loose box — tighten", size=24, look="auto_chip", image=_img("#808080"),
                   point=(380, 295))                                # bottom-right corner of a 400 × 300 image
    r = text_rect(d.styled_el())
    assert r.left() >= -0.01 and r.top() >= -0.01 and r.right() <= 400.01 and r.bottom() <= 300.01
    d.size.setValue(40)                                             # wider than the image: starts at its left edge
    r = text_rect(d.styled_el())
    assert abs(r.left()) < 0.01 and r.bottom() <= 300.01
