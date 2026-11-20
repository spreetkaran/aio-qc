# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Detections drawn on video — the player's overlay (Review playlist, paused-frame Detect) and the saved "frame with
detections" picture share this painter.

  * colours: one colour per CLASS from Image QC's palette (colours picked to be as different as possible, the first
    classes the most distinct), remembered across videos and sessions — a class keeps its colour everywhere; or one
    colour per model ("Colour by: model")
  * several models at once: a small dark badge with the model's number (in the model's colour, underlined in it) leads
    each label, so class and model both read at a glance and a model is never mistaken for a class
  * Image QC's tag look: rounded, slightly see-through, a hairline darker border, black or white text by contrast; the
    class in semi-bold, a lighter " : ", the score in tabular digits (0.91 and 0.88 are the same width — no flutter);
    long names shortened with "…" (full name on hover)
  * steady during playback (Painter): every box keeps an identity from frame to frame (same model + class, overlap)
    and its label keeps last frame's spot while that spot is mostly free; labels are placed oldest first. First choice
    on top touching the box, then on top at its right edge, inside, below; when crowded the label steps away and a
    thin line with a dot in the class colour ties it to its box. Optional glide: a label that has to move slides there
  * tiny boxes: a dot in the class colour (label on hover); boxes the models did not see themselves (interpolated /
    held, or uncertain) are fainter with a short dash; optional confidence shading (fainter = less sure); optional fill
A detection is (x1, y1, x2, y2, text, QColor, dashed[, badge[, conf]]) in video pixels; badge = (number, QColor,
model name) or None; conf a float (for shading) or None.
"""
from __future__ import annotations

import json

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

SETTING = "fx/class_colors"


SEP = " : "                                     # "car : 0.64" — the colon stands clear of the name and the score


def label_text(name, conf) -> str:
    return f"{name}{SEP}{conf:.2f}"


def split_label(text: str) -> str:
    """The class part of a label ("car : 0.64" -> "car")."""
    return str(text).split(SEP)[0]


# ---------------------------------------------------------------------------------------------------- colours
def text_on(bg: QColor) -> QColor:
    """Black or white text, whichever reads better on bg (WCAG relative luminance)."""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = 0.2126 * lin(bg.red()) + 0.7152 * lin(bg.green()) + 0.0722 * lin(bg.blue())
    return QColor(0, 0, 0) if y > 0.36 else QColor(255, 255, 255)


BADGE_BG = QColor(24, 24, 27, 235)


def badge_ink(model_color) -> QColor:
    """The model's colour, lightened until it reads on the dark badge."""
    c = QColor(model_color)
    for _ in range(4):
        if text_on(c) == QColor(0, 0, 0):           # light enough
            break
        c = c.lighter(135)
    return c


class ClassColors:
    """Class name -> colour, from Image QC's palette (most distinct first), kept in the app's settings so a class
    has the same colour in every video, in the review and in the paused-frame overlay."""
    _inst = None

    @classmethod
    def shared(cls):
        if cls._inst is None:
            cls._inst = ClassColors()
        return cls._inst

    def __init__(self, store=None):
        self._store = store
        self.map: dict[str, QColor] = {}
        self._load()

    def _settings(self):
        if self._store is not None:
            return self._store
        from AIO_QC.utils import app_settings
        return app_settings()

    def _load(self):
        try:
            raw = self._settings().value(SETTING, "") or ""
            d = json.loads(raw) if raw else {}
            for k, v in (d.items() if isinstance(d, dict) else ()):
                c = QColor(str(v))
                if c.isValid():
                    self.map[str(k)] = c
        except (ValueError, TypeError):
            self.map = {}

    def _save(self):
        try:
            self._settings().setValue(SETTING, json.dumps({k: v.name() for k, v in self.map.items()}))
        except Exception:
            pass

    def ensure(self, names) -> None:
        """Give every class without a colour the most distinct free one — in the order given (a model's class ids),
        so a model's first classes get the most different colours."""
        from AIO_QC.utils import colors as C
        new = False
        for n in names:
            n = str(n)
            if n not in self.map:
                C.assign(n, self.map)
                new = True
        if new:
            self._save()

    def get(self, name) -> QColor:
        name = str(name)
        if name not in self.map:
            self.ensure([name])
        return self.map[name]


# ---------------------------------------------------------------------------------------------------- painting
def font_pt_for(view_h: float, scale: float = 1.0) -> float:
    """Label size for a view of this height: readable from small windows to full screen."""
    return max(7.5, min(13.0, view_h / 75.0)) * scale


MAX_NAME_W = 170.0                 # longer class names end in "…"
PAD_X, PAD_Y, RADIUS = 5.0, 1.5, 4.0
TINY = 10.0                        # boxes smaller than this (both sides, on screen) are drawn as a dot
KEEP = 0.35                        # last frame's spot is kept while at most this share of it is covered
GLIDE = 0.45                       # share of the remaining way a moving label covers per paint


def _rect_of(m, x1, y1, x2, y2) -> QRectF:
    a, b = m(x1, y1), m(x2, y2)
    return QRectF(QPointF(min(a.x(), b.x()), min(a.y(), b.y())), QPointF(max(a.x(), b.x()), max(a.y(), b.y())))


def _area(r: QRectF) -> float:
    return max(0.0, r.width()) * max(0.0, r.height()) if r.isValid() else 0.0


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _fonts(base: QFont, pt: float):
    name = QFont(base)
    name.setPointSizeF(pt)
    name.setWeight(QFont.Weight.DemiBold)
    num = QFont(name)
    num.setWeight(QFont.Weight.Medium)
    if hasattr(num, "setFeature"):                  # Qt 6.7+: tabular digits
        try:
            num.setFeature(QFont.Tag("tnum"), 1)
        except Exception:
            pass
    return name, num


class Painter:
    """Draws detections and remembers where each box's label went, so labels stay put while boxes move."""

    def __init__(self):
        self.memory: list = []        # [(key, video box, slot, ident, offset QPointF)]
        self._next = 0
        self.moving = False           # a label is still gliding: paint again

    def reset(self):
        self.memory, self.moving = [], False

    def paint(self, p: QPainter, dets, m, bounds: QRectF, font_pt: float, fill: bool = False, labels: bool = True,
              line_w: float = 2.0, shade: bool = False, glide: bool = False) -> bool:
        """Draw; m(x, y) -> QPointF maps video pixels to the painter; bounds = where labels may go.
        -> True while a label is still gliding (paint again soon)."""
        self.moving = False
        answers = [d for d in dets or [] if len(d) > 9 and d[9] == "class"]
        if answers:                                   # a classifier's answer: a pill in the corner, no frame box
            dets = [d for d in dets if not (len(d) > 9 and d[9] == "class")]
            draw_answers(p, answers, m, bounds, font_pt, labels)
        if not dets:
            self.memory = []
            return False
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        f_name, f_num = _fonts(p.font(), font_pt)
        fmn, fmd = QFontMetricsF(f_name), QFontMetricsF(f_num)
        th = math.ceil(max(fmn.height(), fmd.height()) + 2 * PAD_Y)
        num_w = max(fmd.horizontalAdvance(f"{d}.{d}{d}") for d in "0123456789")
        sep_w = fmn.horizontalAdvance(SEP)
        items = []
        for d in dets:
            x1, y1, x2, y2, text, color, dashed = d[:7]
            color = QColor(color)
            r = _rect_of(m, x1, y1, x2, y2)
            conf = d[8] if len(d) > 8 else None
            k = 1.0
            if shade and conf is not None:
                k = 0.35 + 0.65 * max(0.0, min(1.0, (float(conf) - 0.25) / 0.7))      # fainter = less sure
            if dashed:
                k *= 0.6
            items.append((r, d, color, k))
            tiny = r.width() < TINY and r.height() < TINY
            if tiny:
                c = QColor(color)
                c.setAlphaF(k)
                p.setBrush(c)
                p.setPen(QPen(QColor(0, 0, 0, int(160 * k)), 1))
                p.drawEllipse(r.center(), 3.5, 3.5)
                continue
            if fill:
                c = QColor(color)
                c.setAlpha(int(34 * k))
                p.fillRect(r, c)
            p.setBrush(Qt.BrushStyle.NoBrush)
            hp = QPen(QColor(0, 0, 0, int(140 * k)), line_w + 2.0)
            cc = QColor(color)
            cc.setAlphaF(k)
            cp = QPen(cc, line_w)
            for pen in (hp, cp):
                pen.setCosmetic(True)
                if dashed:
                    pen.setDashPattern([4, 3])
            p.setPen(hp)
            p.drawRect(r)
            p.setPen(cp)
            p.drawRect(r)
        if not labels:
            self.memory = []
            p.restore()
            return False
        # identities: each box takes the identity of last frame's best match (same model + class, overlapping)
        prev, used, order = list(self.memory), set(), []
        for r, d, color, k in items:
            badge = d[7] if len(d) > 7 else None
            key = (badge[0] if badge else "", split_label(d[4] or ""))
            box = tuple(float(v) for v in d[:4])
            best, bi = 0.3, None
            for j, (pk, pb, _s, _i, _o) in enumerate(prev):
                if j not in used and pk == key:
                    v = _iou(box, pb)
                    if v > best:
                        best, bi = v, j
            if bi is not None:
                used.add(bi)
                _pk, _pb, slot, ident, off = prev[bi]
            else:
                self._next += 1
                slot, ident, off = None, self._next, None
            order.append((ident, r, d, color, k, key, box, slot, off))
        order.sort(key=lambda t: t[0])                                  # oldest first: they keep their places
        placed, mem = [], []
        for ident, r, d, color, k, key, box, prev_slot, prev_off in order:
            text = str(d[4] or "")
            badge = d[7] if len(d) > 7 else None
            if r.width() < TINY and r.height() < TINY or (not text and not badge):
                mem.append((key, box, None, ident, None))
                continue
            name, conf = (text.split(SEP, 1) + [""])[:2] if SEP in text else (text, "")
            name_el = fmn.elidedText(name, Qt.TextElideMode.ElideRight, MAX_NAME_W)
            bw = (fmn.horizontalAdvance(badge[0]) + 2 * PAD_X) if badge else 0.0
            nw = fmn.horizontalAdvance(name_el) + PAD_X + (0.0 if conf else PAD_X) if name_el else 0.0
            sw = sep_w if conf and name_el else 0.0
            cw = (num_w + PAD_X + (0.0 if name_el else PAD_X)) if conf else 0.0
            w = math.ceil(bw + nw + sw + cw)
            h2 = line_w / 2
            cands = [QRectF(r.left() - h2, r.top() - th - h2, w, th),        # on top, touching the box
                     QRectF(r.right() - w + h2, r.top() - th - h2, w, th),   # on top, at the right edge
                     QRectF(r.left() + h2, r.top() + h2, w, th),             # inside, top
                     QRectF(r.left() - h2, r.bottom() + h2, w, th),          # below
                     QRectF(r.left() + h2, r.bottom() - th - h2, w, th)]     # inside, bottom
            tries = list(range(len(cands)))
            if prev_slot is not None and 0 <= prev_slot < len(cands):
                tries.remove(prev_slot)
                tries.insert(0, prev_slot)
            chosen = slot = None
            for i in tries:
                c = QRectF(cands[i])
                c.moveLeft(min(max(c.left(), bounds.left()), bounds.right() - c.width()))
                if c.top() < bounds.top() or c.bottom() > bounds.bottom():
                    continue
                over = sum(_area(c.intersected(q)) for q in placed)
                if over <= (KEEP * _area(c) if i == prev_slot else 1.0):     # hysteresis: no flip on a 1-px touch
                    chosen, slot = c, i
                    break
            leader = False
            if chosen is None:                                # crowded: step away, tied back by a thin line
                for step in range(1, 6):
                    for sgn in (-1, 1):
                        c = QRectF(cands[0])
                        c.translate(0, sgn * step * (th + 2))
                        c.moveLeft(min(max(c.left(), bounds.left()), bounds.right() - c.width()))
                        if bounds.top() <= c.top() and c.bottom() <= bounds.bottom() and \
                                not any(c.adjusted(1, 1, -1, -1).intersects(q) for q in placed):
                            chosen, slot, leader = c, 10 + sgn * step, True
                            break
                    if chosen is not None:
                        break
            if chosen is None:
                chosen, slot = QRectF(cands[2]), 2
            placed.append(QRectF(chosen))
            off = chosen.topLeft() - r.topLeft()                # where the label sits relative to its box
            if glide and prev_off is not None:
                dx, dy = off.x() - prev_off.x(), off.y() - prev_off.y()
                if abs(dx) > 0.75 or abs(dy) > 0.75:
                    off = QPointF(prev_off.x() + dx * GLIDE, prev_off.y() + dy * GLIDE)
                    self.moving = True
                    chosen.moveTopLeft(r.topLeft() + off)
            mem.append((key, box, slot, ident, QPointF(off)))
            chosen = QRectF(round(chosen.left()), round(chosen.top()), chosen.width(), chosen.height())
            la = (200 if d[6] else 238) * (0.55 + 0.45 * k)
            if leader or chosen.bottom() < r.top() - 3 or chosen.top() > r.bottom() + 3:
                lc = QColor(color)
                lc.setAlphaF(min(1.0, 0.9 * k + 0.1))
                p.setPen(QPen(lc, 1.2))
                corner = r.topLeft() if chosen.bottom() <= r.top() else r.bottomLeft()
                p.drawLine(QPointF(chosen.left() + 4, chosen.bottom() if chosen.bottom() <= corner.y()
                                   else chosen.top()), corner)
                p.setBrush(lc)
                p.drawEllipse(corner, 2.5, 2.5)
            path = QPainterPath()
            path.addRoundedRect(chosen, RADIUS, RADIUS)
            fillc = QColor(color)
            fillc.setAlpha(int(la))
            p.setPen(Qt.PenStyle.NoPen)
            p.fillPath(path, fillc)
            x = chosen.left()
            if badge:                                       # dark chip, model-coloured number: never a class
                bp = QPainterPath()
                bp.addRoundedRect(QRectF(x, chosen.top(), bw + RADIUS, th), RADIUS, RADIUS)
                clip = QPainterPath()
                clip.addRect(QRectF(x, chosen.top(), bw, th))
                p.fillPath(bp.intersected(clip), BADGE_BG)
                p.fillRect(QRectF(x + 2, chosen.bottom() - 2, bw - 4, 2), QColor(badge[1]))
                p.setFont(f_name)
                p.setPen(badge_ink(badge[1]))
                p.drawText(QRectF(x, chosen.top(), bw, th), Qt.AlignmentFlag.AlignCenter, badge[0])
                x += bw
            p.setPen(QPen(color.darker(165), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            ink = text_on(color)
            if name_el:
                p.setFont(f_name)
                p.setPen(ink)
                p.drawText(QRectF(x + PAD_X, chosen.top(), nw, th), Qt.AlignmentFlag.AlignVCenter, name_el)
                x += nw
            if conf:
                if sw:
                    sc = QColor(ink)
                    sc.setAlpha(150)
                    p.setPen(sc)
                    p.setFont(f_name)
                    p.drawText(QRectF(x, chosen.top(), sw, th), Qt.AlignmentFlag.AlignCenter, SEP)
                    x += sw
                p.setFont(f_num)
                p.setPen(ink)
                p.drawText(QRectF(x, chosen.top(), cw - PAD_X, th),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, conf)
        self.memory = mem
        p.restore()
        return self.moving


def draw_answers(p: QPainter, answers, m, bounds: QRectF, font_pt: float, labels: bool = True):
    """Classifiers' answers for the whole frame — (0, 0, w, h, "class : 0.97", colour, dashed, badge, conf,
    "class") — as pills stacked in the frame's top-left corner, one per model (its badge first). No rectangle:
    a box around the whole picture would read as a detection."""
    if not answers:
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    f_name, f_num = _fonts(p.font(), font_pt * 1.15)
    fmn = QFontMetricsF(f_name)
    th = math.ceil(fmn.height() + 2 * PAD_Y + 2)
    r = _rect_of(m, *answers[0][:4]).intersected(bounds)
    x0, y = r.left() + 8, r.top() + 8
    for d in answers:
        text = str(d[4] or "") if labels else split_label(str(d[4] or ""))
        badge = d[7] if len(d) > 7 else None
        color = QColor(d[5])
        bw = (fmn.horizontalAdvance(badge[0]) + 2 * PAD_X) if badge else 0.0
        tw = fmn.horizontalAdvance(text) + 2 * PAD_X + 4
        box = QRectF(x0, y, math.ceil(bw + tw), th)
        path = QPainterPath()
        path.addRoundedRect(box, RADIUS + 1, RADIUS + 1)
        fillc = QColor(color)
        fillc.setAlpha(150 if d[6] else 235)
        p.setPen(Qt.PenStyle.NoPen)
        p.fillPath(path, fillc)
        x = box.left()
        if badge:
            bp = QPainterPath()
            bp.addRoundedRect(QRectF(x, box.top(), bw + RADIUS, th), RADIUS, RADIUS)
            clip = QPainterPath()
            clip.addRect(QRectF(x, box.top(), bw, th))
            p.fillPath(bp.intersected(clip), BADGE_BG)
            p.fillRect(QRectF(x + 2, box.bottom() - 2, bw - 4, 2), QColor(badge[1]))
            p.setFont(f_name)
            p.setPen(badge_ink(badge[1]))
            p.drawText(QRectF(x, box.top(), bw, th), Qt.AlignmentFlag.AlignCenter, badge[0])
            x += bw
        p.setPen(QPen(color.darker(165), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setFont(f_name)
        p.setPen(text_on(color))
        p.drawText(QRectF(x + PAD_X, box.top(), tw - PAD_X, th), Qt.AlignmentFlag.AlignVCenter, text)
        y += th + 4
    p.restore()


def classifier_answer(rows):
    """Rows of one model on a frame -> the best row when they are a classifier's whole-frame answers, else None."""
    from AIO_QC.tools.frame_extractor.core.detector import is_cls_rows
    import numpy as np
    a = np.asarray(rows, np.float32).reshape(-1, 6) if rows is not None and len(rows) else None
    if a is None or not is_cls_rows(a) or a[0, 2] < 2 or a[0, 3] < 2:
        return None
    return a[int(np.argmax(a[:, 4]))]


def paint(p: QPainter, dets, m, bounds: QRectF, font_pt: float, fill: bool = False, labels: bool = True,
          line_w: float = 2.0, shade: bool = False):
    """One picture, no memory (a saved frame)."""
    Painter().paint(p, dets, m, bounds, font_pt, fill, labels, line_w, shade=shade)
