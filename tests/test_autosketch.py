# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Auto-sketch geometry: marks per style, arrows inside the image, from free space, never crossing when avoidable,
heads at the mark, sizes that follow the image."""
import math

from AIO_QC.core.autosketch import STYLES, plan, shape_box


def _arrows(els):
    return [e for e in els if e["t"] == "arrow"]


def _inside(el, W, H):
    return all(0 <= x <= W and 0 <= y <= H for x, y in el["pts"])


def test_styles():
    b = [(400, 300, 500, 380)]
    kinds = {k: sorted(e["t"] for e in plan(1000, 800, b, style=k)) for k, _t in STYLES}
    assert kinds == {"arrow_ring": ["arrow", "ellipse"], "arrow": ["arrow"], "arrow_box": ["arrow", "rect"],
                     "ring": ["ellipse"]}
    ring = next(e for e in plan(1000, 800, b, style="ring"))
    (x1, y1), (x2, y2) = ring["pts"]
    assert x1 < 400 and y1 < 300 and x2 > 500 and y2 > 380        # around the shape


def test_arrows_inside_and_pointing_at_their_shape():
    W, H = 1280, 720
    targets = [(50, 40, 200, 160), (1100, 560, 1260, 700), (600, 300, 680, 380)]
    els = plan(W, H, targets, style="arrow")
    arrows = _arrows(els)
    assert len(arrows) == 3
    for a, t in zip(arrows, targets):
        assert _inside(a, W, H)
        (tx, ty), (hx, hy) = a["pts"]
        cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        assert math.hypot(hx - cx, hy - cy) < math.hypot(tx - cx, ty - cy)     # the head is the end near the shape
        assert t[0] - 30 <= hx <= t[2] + 30 and t[1] - 30 <= hy <= t[3] + 30


def test_arrow_avoids_busy_side():
    W, H = 1000, 1000
    target = [(450, 450, 550, 550)]
    others = [(0, 0, 1000, 440), (0, 0, 440, 1000)]                # everything above and left is full of objects
    (tx, ty), (hx, hy) = _arrows(plan(W, H, target, others, style="arrow"))[0]["pts"]
    for k in range(11):                                            # the whole arrow lies in the empty part
        x, y = tx + (hx - tx) * k / 10, ty + (hy - ty) * k / 10
        assert x > 440 and y > 440


def test_arrows_do_not_cross():
    from AIO_QC.core.autosketch import _cross
    W, H = 1600, 1000
    targets = [(300 + 220 * i, 420, 380 + 220 * i, 520) for i in range(5)]
    arrows = _arrows(plan(W, H, targets, style="arrow_ring"))
    for i in range(len(arrows)):
        for j in range(i + 1, len(arrows)):
            assert not _cross(*arrows[i]["pts"], *arrows[j]["pts"])


def test_sizes_follow_the_image_and_labels():
    small = _arrows(plan(640, 480, [(200, 200, 260, 260)], style="arrow"))[0]["w"]
    big = _arrows(plan(6000, 4000, [(2000, 2000, 2600, 2600)], style="arrow"))[0]["w"]
    assert big > 3 * small
    els = plan(800, 600, [(300, 300, 400, 380)], style="arrow", labels=["plate"])
    text = [e for e in els if e["t"] == "text"]
    assert text and text[0]["text"] == "plate" and _inside(text[0], 800, 600)


def test_edge_cases():
    assert plan(500, 400, []) == []
    els = plan(200, 150, [(0, 0, 200, 150)], style="arrow")       # a shape as big as the image: still drawn
    assert len(_arrows(els)) == 1 and _inside(_arrows(els)[0], 200, 150)
    assert shape_box([[10, 10], [13, 14]], "circle") == (5.0, 5.0, 15.0, 15.0)
    assert shape_box([[5, 9], [1, 2], [7, 3]], "polygon") == (1.0, 2.0, 7.0, 9.0)


def _R(label, x1, y1, x2, y2):
    return {"label": label, "shape_type": "rectangle", "points": [[x1, y1], [x2, y2]]}


def test_choices_survive_edits(tmp_path):
    from AIO_QC.core.autosketch import ChoiceStore
    key = str(tmp_path / "a.jpg")
    shapes = [_R("car", 0, 0, 100, 100), _R("plate", 10, 60, 40, 75), _R("car", 200, 0, 300, 100)]
    st = ChoiceStore(tmp_path)
    st.set(key, shapes, [1, 2])
    st.flush()
    st = ChoiceStore(tmp_path)                                  # read back from disk
    assert st.resolve(key, shapes) == [1, 2] and st.count(key) == 2
    edited = [_R("plate", 11, 61, 41, 76), _R("car", 202, 1, 301, 99)]        # first car deleted, boxes nudged
    assert st.resolve(key, edited) == [0, 1]
    assert st.resolve(key, [_R("car", 0, 0, 100, 100)]) == []                 # both gone
    st.set(key, shapes, [])
    assert st.count(key) == 0 and not st.keys_with_choices()


def test_labels_never_overlap():
    targets = [(300, 300, 340, 340), (330, 340, 380, 400), (310, 400, 360, 450), (350, 290, 400, 330)]
    els = plan(800, 600, targets, style="arrow", labels=["elephant", "person", "person", "dog"])
    rects = []
    for e in els:
        if e["t"] == "text":
            (x, y), fh = e["pts"][0], e["w"] * 5
            rects.append((x, y - fh, x + fh * 0.62 * len(e["text"]), y + fh * 0.25))
    assert len(rects) == 4
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
