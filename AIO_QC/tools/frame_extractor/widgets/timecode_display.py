# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Current-position display, NLE style.

  ┌────────────────────────────────────────────────────────────┐
  │ ● 00:01:23.450 / 00:05:00.000                     [TIME ▾] │
  │   FR 2503 / 8999 · 27.8%   2×  ≈KF  VFR  IN→OUT 12.4s  ◆ ▲ │
  └────────────────────────────────────────────────────────────┘

  click            type a position (1:23.5 · 00:01:23:12 · 1234f · 42% · +5s · in / out / end)
  wheel on a unit  nudge by that unit (hours, minutes, seconds, frames / digit); Shift = x10
  drag sideways    scrub by the unit under the cursor
  format chip      cycle Time -> Timecode (SMPTE, drop-frame aware) -> Frames -> Seconds -> Remaining
  right-click      formats, copy (time / timecode / frame / "file @ time"), go to, overlay on video
"""

from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QFontDatabase, QFontMetricsF, QPen, QGuiApplication, QActionGroup
from PyQt6.QtWidgets import QWidget, QLineEdit, QMenu, QToolTip, QSizePolicy

from AIO_QC.tools.frame_extractor.core.timecode import (
    FORMATS,
    FORMAT_LABELS,
    FORMAT_CHIPS,
    format_position,
    format_duration,
    parse_position,
    nudge,
    frames_to_smpte,
    nominal_rate,
)
from AIO_QC.utils.fx_settings import fx_settings

GREEN, BLUE, ORANGE, RED, PURPLE = (QColor(c) for c in ("#43A047", "#1E88E5", "#FB8C00", "#E53935", "#8E24AA"))
HINT = "1:23.5 · 00:01:23:12 · 1234f · 42% · +5s · -30f · in · out · end"


def _mono(size, bold=False):
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    for fam in ("Cascadia Mono", "JetBrains Mono", "Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono"):
        if fam in QFontDatabase.families():
            f = QFont(fam)
            break
    f.setPointSizeF(size)
    f.setBold(bold)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


class _Editor(QLineEdit):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self.owner.cancel_edit()
            return
        if ev.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):      # arrow keys nudge the typed value
            self.owner.edit_nudge(1 if ev.key() == Qt.Key.Key_Up else -1)
            return
        super().keyPressEvent(ev)

    def focusOutEvent(self, ev):
        super().focusOutEvent(ev)
        QTimer.singleShot(0, self.owner.cancel_edit)


class TimecodeDisplay(QWidget):
    seek_requested = pyqtSignal(int, bool)      # frame, scrubbing (True while dragging / wheel-spinning)
    format_changed = pyqtSignal(str)
    overlay_toggled = pyqtSignal(bool)

    SCRUB_PX = 7                                # pixels per unit step while dragging

    def __init__(self, parent=None):
        super().__init__(parent)
        self.store = fx_settings()
        self.fmt = self.store.value("ui/timecode_format", "time")
        self.fmt = self.fmt if self.fmt in FORMATS else "time"
        self.overlay = self.store.value("ui/timecode_overlay", False, type=bool)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(50)
        self.setMinimumWidth(380)
        self.setToolTip("")
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.big, self.small, self.chipf = _mono(17, True), _mono(9.5), _mono(8, True)
        self.editor = _Editor(self)
        self.editor.setFont(self.big)
        self.editor.hide()
        self.editor.returnPressed.connect(self.commit_edit)
        self.editor.setPlaceholderText("go to…")
        self.clear()

    # ------------------------------------------------------------------ data
    def clear(self):
        self.loaded = False
        self.frame, self.frame_count, self.fps = 0, 0, 30.0
        self.time_of = self.index_of = None
        self.duration, self.vfr, self.video_name = 0.0, False, ""
        self.preview = self.playing = self.ended = False
        self.speed = 1.0
        self.in_frame = self.out_frame = None
        self.marker = None
        self.bookmark = False
        self._segs, self._hover, self._press = [], None, None
        self._chip_rect = self._frame_rect = self._primary_rect = QRectF()
        self.update()

    def set_video(self, fps, frame_count, time_of, index_of, duration, vfr=False, name=""):
        self.loaded = True
        self.fps, self.frame_count = fps or 30.0, frame_count
        self.time_of, self.index_of = time_of, index_of
        self.duration, self.vfr, self.video_name = duration, vfr, name
        self.update()

    def set_length(self, frame_count, duration):
        self.frame_count, self.duration = frame_count, duration
        self.update()

    def set_position(self, frame, preview=False):
        if frame != self.frame or preview != self.preview:
            self.frame, self.preview = frame, preview
            self.update()

    def set_state(self, playing=None, speed=None, ended=None):
        if playing is not None:
            self.playing = playing
        if speed is not None:
            self.speed = speed
        if ended is not None:
            self.ended = ended
        self.update()

    def set_context(self, in_frame=None, out_frame=None, marker=None, bookmark=False):
        self.in_frame, self.out_frame, self.marker, self.bookmark = in_frame, out_frame, marker, bookmark
        self.update()

    def set_format(self, fmt):
        if fmt in FORMATS and fmt != self.fmt:
            self.fmt = fmt
            self.store.setValue("ui/timecode_format", fmt)
            self.format_changed.emit(fmt)
            self.update()

    def cycle_format(self, direction=1):
        i = FORMATS.index(self.fmt)
        self.set_format(FORMATS[(i + direction) % len(FORMATS)])

    def set_overlay(self, on):
        self.overlay = bool(on)
        self.store.setValue("ui/timecode_overlay", self.overlay)
        self.overlay_toggled.emit(self.overlay)

    # ------------------------------------------------------------- text helpers
    def text_for(self, fmt, frame=None):
        f = self.frame if frame is None else frame
        # "remaining" counts down to the last frame (0 at the end), consistent with the frame counter
        return format_position(fmt, f, self.fps, self.time_of, self.end_time(), self.frame_count)

    def end_time(self):
        """Timestamp of the last frame. Totals use it - like the frame counter's '#last / last' - so the
        display reads e.g. 04:01.282 / 04:01.282 at the end. (The file's duration also includes how long
        the last frame stays on screen; that is shown in the tooltip.)"""
        if self.frame_count > 0:
            return self.seconds(self.frame_count - 1)
        return self.duration

    def seconds(self, frame=None):
        f = self.frame if frame is None else frame
        return self.time_of(f) if self.time_of else f / (self.fps or 30.0)

    def overlay_lines(self):
        """Text for the on-video timecode overlay."""
        tc = frames_to_smpte(self.frame, self.fps) if not self.vfr else self.text_for("time")[0]
        return [self.text_for("time")[0] + ("  ≈" if self.preview else ""),
                f"{tc}   #{self.frame}"]

    # ------------------------------------------------------------------ paint
    def _pal(self):
        pal = self.palette()
        base = pal.window().color()
        dark = base.lightness() < 128
        panel = base.lighter(118) if dark else base.darker(104)
        text = pal.windowText().color()
        dim = QColor(text)
        dim.setAlpha(120)
        accent = pal.highlight().color()
        return dark, panel, text, dim, accent

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark, panel, text, dim, accent = self._pal()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(text.red(), text.green(), text.blue(), 40), 1))
        p.setBrush(panel)
        p.drawRoundedRect(r, 8, 8)

        # state LED
        led = (RED if self.ended else ORANGE if self.preview else GREEN if self.playing else dim)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(led)
        p.drawEllipse(QPointF(14, 17), 4.5, 4.5)

        # format chip (right)
        p.setFont(self.chipf)
        fmc = QFontMetricsF(self.chipf)
        chip = FORMAT_CHIPS[self.fmt] + " ▾"
        cw = fmc.horizontalAdvance(chip) + 14
        self._chip_rect = QRectF(self.width() - cw - 8, 7, cw, 20)
        hov_chip = self._hover == "chip"
        c = QColor(accent)
        c.setAlpha(70 if hov_chip else 38)
        p.setBrush(c)
        p.drawRoundedRect(self._chip_rect, 5, 5)
        p.setPen(text)
        p.drawText(self._chip_rect, Qt.AlignmentFlag.AlignCenter, chip)

        if not self.loaded:
            p.setFont(self.big)
            p.setPen(dim)
            p.drawText(QRectF(26, 3, self.width() - 40, 30), Qt.AlignmentFlag.AlignVCenter, "--:--:--.---")
            p.setFont(self.small)
            p.drawText(QRectF(26, 30, self.width() - 40, 16), Qt.AlignmentFlag.AlignVCenter, "no video")
            return

        # primary readout, segment by segment
        txt, segs = self.text_for(self.fmt)
        p.setFont(self.big)
        fm = QFontMetricsF(self.big)
        x0, y0, h0 = 26.0, 3.0, 30.0
        seg_at = {i: s for s in segs for i in range(s[0], s[1])}
        self._segs = []
        x = x0
        for i, ch in enumerate(txt):
            w = fm.horizontalAdvance(ch)
            if i in seg_at:
                s = seg_at[i]
                if i == s[0]:
                    sw = fm.horizontalAdvance(txt[s[0]:s[1]])
                    rect = QRectF(x - 2, y0 + 3, sw + 4, h0 - 4)
                    self._segs.append((rect, s[2]))
                    if self._hover == ("seg", len(self._segs) - 1) or (self._press and self._press.get("seg") == len(self._segs) - 1
                                                                         and self._press.get("scrub")):
                        hc = QColor(accent)
                        hc.setAlpha(60)
                        p.setPen(Qt.PenStyle.NoPen)
                        p.setBrush(hc)
                        p.drawRoundedRect(rect, 4, 4)
                p.setPen(ORANGE if self.preview else text)
            else:
                p.setPen(dim)
            p.drawText(QRectF(x, y0, w + 1, h0), Qt.AlignmentFlag.AlignVCenter, ch)
            x += w
        self._primary_rect = QRectF(x0 - 4, y0, x - x0 + 8, h0)

        # total
        p.setFont(self.small)
        fms = QFontMetricsF(self.small)
        if self.fmt == "frames":
            total = f"/ {max(0, self.frame_count - 1)}"
        elif self.fmt == "remaining":
            total = f"of {format_position('time', 0, self.fps, lambda f: self.end_time(), self.end_time(), 1)[0]}"
        elif self.fmt == "smpte":
            total = "/ " + frames_to_smpte(max(0, self.frame_count - 1), self.fps)
        elif self.fmt == "seconds":
            total = f"/ {self.end_time():.3f} s"
        else:
            total = "/ " + format_position("time", 0, self.fps, lambda f: self.end_time(), self.end_time(), 1)[0]
        p.setPen(dim)
        p.drawText(QRectF(x + 8, y0 + 6, fms.horizontalAdvance(total) + 4, h0 - 6), Qt.AlignmentFlag.AlignVCenter, total)

        # second line: secondary position + chips
        y1 = 33.0
        pct = 100.0 * self.frame / max(1, self.frame_count - 1)
        if self.fmt == "frames":
            second = f"{self.text_for('time')[0]} · {pct:.1f}%"
        else:
            second = f"#{self.frame} / {max(0, self.frame_count - 1)} · {pct:.1f}%"
        sw = fms.horizontalAdvance(second)
        self._frame_rect = QRectF(24, y1 - 1, sw + 6, 15)
        if self._hover == "frame":
            hc = QColor(accent)
            hc.setAlpha(45)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(hc)
            p.drawRoundedRect(self._frame_rect, 3, 3)
        p.setPen(dim)
        p.drawText(QRectF(27, y1, sw + 2, 13), Qt.AlignmentFlag.AlignVCenter, second)
        cx = 27 + sw + 10
        chips = []
        if self.ended:
            chips.append(("END", RED))
        if abs(self.speed - 1.0) > 1e-6:
            chips.append((f"{self.speed:g}×", accent))
        if self.preview:
            chips.append(("≈ KEYFRAME", ORANGE))
        if self.vfr:
            chips.append(("VFR", PURPLE))
        if self.in_frame is not None or self.out_frame is not None:
            a = self.in_frame or 0
            b = self.out_frame if self.out_frame is not None else max(0, self.frame_count - 1)
            inside = a <= self.frame <= b
            chips.append((f"IN→OUT {format_duration(self.seconds(b) - self.seconds(a))}", ORANGE if inside else dim))
        if self.marker:
            chips.append(("◆ MARKER" if self.marker != "saved" else "◆ SAVED", BLUE if self.marker == "saved" else GREEN))
        if self.bookmark:
            chips.append(("▼ BOOKMARK", ORANGE))
        p.setFont(self.chipf)
        for label, col in chips:
            w = fmc.horizontalAdvance(label) + 10
            if cx + w > self.width() - 8:
                break
            bg = QColor(col)
            bg.setAlpha(40)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            rr = QRectF(cx, y1, w, 14)
            p.drawRoundedRect(rr, 4, 4)
            p.setPen(col)
            p.drawText(rr, Qt.AlignmentFlag.AlignCenter, label)
            cx += w + 5
        p.end()

    # ------------------------------------------------------------------ hit testing
    def _hit(self, pos):
        if self._chip_rect.contains(pos):
            return "chip"
        for i, (rect, _) in enumerate(self._segs):
            if rect.contains(pos):
                return ("seg", i)
        if self._frame_rect.contains(pos):
            return "frame"
        return None

    def _step(self, hit):
        if isinstance(hit, tuple):
            return self._segs[hit[1]][1]
        return ("f", 1)

    # ------------------------------------------------------------------ mouse
    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._press and ev.buttons() & Qt.MouseButton.LeftButton and self.loaded:
            dx = pos.x() - self._press["x"]
            if not self._press["scrub"] and abs(dx) >= 4:
                self._press["scrub"] = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            if self._press["scrub"]:
                steps = int(dx / self.SCRUB_PX)
                if steps != self._press["steps"]:
                    self._press["steps"] = steps
                    f = nudge(self._press["frame"], self._press["step"], steps, self.fps, self.time_of,
                              self.index_of, self.frame_count)
                    self._press["target"] = f
                    self.seek_requested.emit(f, True)
            return
        hit = self._hit(pos)
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit == "chip" else
                           Qt.CursorShape.SizeHorCursor if isinstance(hit, tuple) else Qt.CursorShape.IBeamCursor)
            self.update()
        if self.loaded:
            QToolTip.showText(ev.globalPosition().toPoint(), self._tooltip(hit), self)

    def leaveEvent(self, ev):
        self._hover = None
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            hit = self._hit(ev.position())
            if hit == "chip":
                self.cycle_format(-1 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1)
                return
            self._press = dict(x=ev.position().x(), hit=hit, frame=self.frame, step=self._step(hit),
                               seg=hit[1] if isinstance(hit, tuple) else None, scrub=False, steps=0)

    def mouseReleaseEvent(self, ev):
        press, self._press = self._press, None
        if press is None:
            return
        if press["scrub"]:
            self.seek_requested.emit(press.get("target", self.frame), False)   # settle on the exact frame
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.update()
        elif self.loaded:
            self.start_edit(mode="frames" if press["hit"] == "frame" else None)

    def wheelEvent(self, ev):
        if not self.loaded:
            return
        dy = ev.angleDelta().y()
        if not dy:
            return
        hit = self._hit(ev.position())
        if hit == "chip":
            self.cycle_format(-1 if dy > 0 else 1)
            return
        n = (1 if dy > 0 else -1) * (10 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1)
        f = nudge(self.frame, self._step(hit), n, self.fps, self.time_of, self.index_of, self.frame_count)
        self.seek_requested.emit(f, False)
        ev.accept()

    def contextMenuEvent(self, ev):
        m = QMenu(self)
        g = QActionGroup(m)
        for f in FORMATS:
            a = m.addAction(f"{FORMAT_LABELS[f]}    {self.text_for(f)[0] if self.loaded else ''}")
            a.setCheckable(True)
            a.setChecked(f == self.fmt)
            a.triggered.connect(lambda _=False, f=f: self.set_format(f))
            g.addAction(a)
        m.addSeparator()
        m.addAction("Go to time / frame…    Ctrl+J").triggered.connect(lambda: self.start_edit())
        if self.loaded:
            m.addSeparator()
            for label, value in (("Copy time", self.text_for("time")[0]),
                                 ("Copy timecode", frames_to_smpte(self.frame, self.fps)),
                                 ("Copy frame number", str(self.frame)),
                                 ("Copy “file @ time”", f"{self.video_name} @ {self.text_for('time')[0]} (#{self.frame})")):
                m.addAction(f"{label}    {value}").triggered.connect(
                    lambda _=False, v=value: QGuiApplication.clipboard().setText(v))
        m.addSeparator()
        ov = m.addAction("Show timecode on the video")
        ov.setCheckable(True)
        ov.setChecked(self.overlay)
        ov.toggled.connect(self.set_overlay)
        m.exec(ev.globalPos())

    def _tooltip(self, hit):
        if hit == "chip":
            return f"<b>{FORMAT_LABELS[self.fmt]}</b> — click to cycle formats (Shift+click: back)"
        if isinstance(hit, tuple):
            kind, v = self._step(hit)
            unit = (f"{abs(v)} frame{'s' if abs(v) != 1 else ''}" if kind == "f" else
                    {3600: "1 hour", 60: "1 minute", 1: "1 second"}.get(abs(v), f"{abs(v)} s"))
            return f"Scroll or drag to move by <b>{unit}</b> (Shift+scroll ×10) · click to type"
        rb, drop = nominal_rate(self.fps)
        t = self.seconds()
        nxt = self.seconds(min(self.frame + 1, max(0, self.frame_count - 1)))
        return (f"<b>{self.text_for('time')[0]}</b> · frame {self.frame}<br>"
                f"timecode {frames_to_smpte(self.frame, self.fps)} ({rb}{' drop-frame' if drop else ''})<br>"
                f"exact {t:.6f} s · frame lasts {max(0.0, nxt - t) * 1000:.2f} ms · {self.fps:.3f} fps"
                + f"<br>last frame at {self.text_for('time', max(0, self.frame_count - 1))[0]} · "
                  f"file duration {self.duration:.3f} s (includes the last frame's display time)"
                + (" (variable)" if self.vfr else "") + "<br><i>Click to type: " + HINT + "</i>")

    # ------------------------------------------------------------------ editing
    def start_edit(self, mode=None):
        if not self.loaded:
            return
        self._edit_mode = mode or self.fmt
        text = str(self.frame) if self._edit_mode == "frames" else self.text_for(
            self._edit_mode if self._edit_mode != "remaining" else "time")[0]
        r = self._primary_rect if not self._primary_rect.isNull() else QRectF(22, 3, 220, 30)
        self.editor.setGeometry(int(r.x()), int(r.y()) + 1, max(260, int(r.width()) + 60), int(r.height()))
        self.editor.setText(text)
        self.editor.setStyleSheet("")
        self.editor.setToolTip(HINT)
        self.editor.show()
        self.editor.setFocus(Qt.FocusReason.MouseFocusReason)
        self.editor.selectAll()
        QToolTip.showText(self.mapToGlobal(r.bottomLeft().toPoint()), "Type a position: " + HINT, self)

    def cancel_edit(self):
        """Esc, or focus left the editor: close it without moving."""
        if self.editor.isVisible():
            self.editor.hide()
            self.editor.clearFocus()

    def _parse(self, text):
        return parse_position(text, current=self.frame, fps=self.fps, frame_count=self.frame_count,
                              time_of=self.time_of, index_of=self.index_of, in_frame=self.in_frame,
                              out_frame=self.out_frame, mode=self._edit_mode)

    def commit_edit(self):
        try:
            f = self._parse(self.editor.text())
        except ValueError as e:
            self.editor.setStyleSheet("QLineEdit { border: 2px solid #E53935; border-radius: 4px; }")
            QToolTip.showText(self.editor.mapToGlobal(self.editor.rect().bottomLeft()), str(e), self.editor)
            return
        self.editor.hide()
        self.seek_requested.emit(f, False)
        self.setFocus()

    def edit_nudge(self, direction):
        try:
            f = self._parse(self.editor.text())
        except ValueError:
            f = self.frame
        f = nudge(f, ("f", 1), direction, self.fps, self.time_of, self.index_of, self.frame_count)
        mode = self._edit_mode
        self.editor.setText(str(f) if mode == "frames" else self.text_for(mode if mode != "remaining" else "time", f)[0])
        self.seek_requested.emit(f, False)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.start_edit()
        else:
            super().keyPressEvent(ev)
