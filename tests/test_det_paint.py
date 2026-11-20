# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Detections on video: class colours from Image QC's palette (most distinct first, in a model's class order, kept
across sessions), black / white label text by contrast, model badges readable on their dark chip, and the painter
(labels on / off, fill, badges, labels kept apart) drawing what it should."""
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter


class _Store(dict):
    def value(self, k, d=None):
        return self.get(k, d)

    def setValue(self, k, v):
        self[k] = v


def test_class_colours_distinct_and_kept(qapp):
    from AIO_QC.tools.frame_extractor.widgets import det_paint as DP
    from AIO_QC.utils.colors import palette
    st = _Store()
    cc = DP.ClassColors(st)
    cc.ensure(["car", "plate", "truck"])
    assert [cc.get(n).name() for n in ("car", "plate", "truck")] == list(palette()[:3])   # the most distinct first
    assert len({cc.get(n).name() for n in ("car", "plate", "truck", "bus", "person")}) == 5
    again = DP.ClassColors(st)                                    # another session: same colours
    assert again.get("plate").name() == cc.get("plate").name() and again.get("bus").name() == cc.get("bus").name()


def test_contrast_and_badges(qapp):
    from AIO_QC.tools.frame_extractor.widgets import det_paint as DP
    assert DP.text_on(QColor("#ffd400")) == QColor(0, 0, 0) and DP.text_on(QColor("#1f69ff")) == QColor(255, 255, 255)
    ink = DP.badge_ink(QColor("#1e3a8a"))                         # a dark model colour is lightened on the badge
    assert DP.text_on(ink) == QColor(0, 0, 0)


def test_painter_draws_boxes_labels_and_badges(qapp):
    from AIO_QC.tools.frame_extractor.widgets import det_paint as DP
    red, green = QColor("#ff261f"), QColor("#00ff00")
    assert DP.label_text("car", 0.912) == "car : 0.91" and DP.split_label("trailer text : 0.82") == "trailer text"
    dets = [(20, 30, 120, 110, "car : 0.91", red, False, ("1", QColor("#1e88e5"), "model A")),
            (60, 40, 150, 120, "plate : 0.80", green, True, None)]

    def render(**kw):
        img = QImage(200, 160, QImage.Format.Format_RGB888)
        img.fill(QColor(128, 128, 128))
        p = QPainter(img)
        DP.paint(p, dets, lambda x, y: QPointF(x, y), QRectF(0, 0, 200, 160), 9, **kw)
        p.end()
        return img

    img = render()
    assert img.pixelColor(20, 70).red() > 200 and img.pixelColor(20, 70).green() < 120        # red outline
    off = render(labels=False)
    on = render(labels=True)
    diff = [(x, y) for x in range(0, 200, 2) for y in range(0, 160, 2) if on.pixelColor(x, y) != off.pixelColor(x, y)]
    assert len(diff) > 40                                                                      # labels drawn
    # the two labels do not cover each other: the car's label (first seen) on top of its box, the plate's label,
    # whose place on top is taken, inside the top of its own box
    assert any(y < 30 for _x, y in diff) and any(40 <= y <= 60 and x > 62 for x, y in diff)
    filled = render(fill=True, labels=False)
    inside = filled.pixelColor(90, 70)
    assert inside != QColor(128, 128, 128)                                                     # light fill


def test_steady_labels_during_playback(qapp):
    """Boxes wobbling a few pixels in a crowded scene: labels keep their places (identities + hysteresis)."""
    import random
    from AIO_QC.tools.frame_extractor.widgets import det_paint as DP

    class Rec(QPainter):
        def __init__(self, dev):
            super().__init__(dev)
            self.pills = []

        def fillPath(self, path, c):
            b = path.boundingRect()
            if b.width() > 30:
                self.pills.append(b.center())
            return super().fillPath(path, c)

    random.seed(5)
    base = []
    for row in range(4):
        for col in range(7):
            x, y = 30 + col * 75 + random.uniform(-8, 8), 40 + row * 95 + random.uniform(-8, 8)
            w, h = random.uniform(55, 95), random.uniform(55, 90)
            k = random.choice(["car", "truck", "person", "bus"])
            base += [(x, y, x + w, y + h, k, 0), (x + 4, y + 3, x + w + 3, y + h - 2, k, 1)]
    cols = {"car": QColor("#ff261f"), "truck": QColor("#00ff00"), "person": QColor("#1f69ff"), "bus": QColor("#ffd400")}
    painter, prev, jumps, n = DP.Painter(), None, 0, 0
    for _t in range(40):
        dets = []
        for x1, y1, x2, y2, k, mi in base:
            dx, dy, g = random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-2, 2)
            dets.append((x1 + dx - g, y1 + dy - g, x2 + dx + g, y2 + dy + g, DP.label_text(k, random.uniform(.4, .99)),
                         cols[k], False, (str(mi + 1), QColor("#e53935"), "m"), 0.8))
        img = QImage(640, 460, QImage.Format.Format_RGB888)
        p = Rec(img)
        painter.paint(p, dets, lambda x, y: QPointF(x, y), QRectF(0, 0, 640, 460), 9.5)
        p.end()
        if prev is not None:
            for c in p.pills:
                n += 1
                if min(((c.x() - q.x()) ** 2 + (c.y() - q.y()) ** 2) ** 0.5 for q in prev) > 12:
                    jumps += 1
        prev = p.pills
    assert n and jumps / n < 0.06, jumps / n                 # was ~0.24 before labels kept their places
