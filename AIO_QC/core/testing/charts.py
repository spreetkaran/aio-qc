# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
SVG charts for the Model Lab - one implementation for the app (rendered with QtSvg) and the HTML reports
(inline SVG with hover tooltips). Pure Python, no plotting library.

Design rules (validated reference palette, see the dataviz guidance):
  * colour = identity of a MODEL, in a fixed slot order that never changes with filtering or rank
    (model i is always slot i); sequential single-hue blue for magnitudes (confusion matrices);
    blue <-> red with a grey midpoint for signed differences
  * thin marks: 2 px lines, bars <= 24 px with a 4 px rounded data end and a square base, 2 px surface
    gaps between touching bars, markers r >= 4 with a 2 px surface ring; hairline solid grid
  * text never wears a series colour (ink tokens only); a coloured key mark sits beside it
  * a legend for >= 2 series plus selective direct labels; every chart has a table twin in the report

target="html" emits style="...: var(--token)" so one SVG follows the page's light / dark theme;
target="qt" emits literal colours for the given mode (QtSvg has no CSS variables).
Every mark carries data-tip="..." which the report's script turns into a tooltip.
"""
import math
from html import escape

from AIO_QC.tools.frame_extractor.core.common import MODEL_PALETTE as SERIES
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TOKENS = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", mid="#f0efec", pos="#2a78d6", neg="#e34948",
                  critical="#d03b3b", good="#0ca30c", empty="#f0efec"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", mid="#383835", pos="#3987e5", neg="#e66767",
                 critical="#d03b3b", good="#0ca30c", empty="#262624"),
}
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"     # Inter is embedded in reports


def seq_ramp(mode):
    """Sequential ramp for a mode: light = pale -> dark; dark = dim -> bright (more is always more visible)."""
    return SEQ if mode == "light" else SEQ[::-1]


def _seq_text(hexc):
    return "#0b0b0b" if _lum(hexc) > 0.3 else "#ffffff"


def css_tokens():
    """CSS custom properties for the report (both modes, theme-toggle aware)."""
    def block(mode):
        t = TOKENS[mode]
        lines = [f"--c-{k}: {v};" for k, v in t.items()]
        lines += [f"--s{i}: {c};" for i, c in enumerate(SERIES[mode])]
        lines += [f"--q{i}: {c}; --qt{i}: {_seq_text(c)};" for i, c in enumerate(seq_ramp(mode))]
        return " ".join(lines)
    return (f":root {{ color-scheme: light; {block('light')} }}\n"
            f"@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme='light'])) {{ color-scheme: dark; {block('dark')} }} }}\n"
            f":root[data-theme='dark'] {{ color-scheme: dark; {block('dark')} }}\n")


class Ink:
    """Colour resolver: CSS variables (html) or literal colours (qt)."""

    def __init__(self, target="html", mode="light"):
        self.target, self.mode = target, mode
        # QtSvg takes one family name: the bundled Inter (registered by utils.fonts.load()); browsers get a list
        self.font = "Inter" if target == "qt" else FONT

    def tok(self, name):
        return f"var(--c-{name})" if self.target == "html" else TOKENS[self.mode][name]

    def series(self, i):
        i = int(i) % 8
        return f"var(--s{i})" if self.target == "html" else SERIES[self.mode][i]

    def seq(self, k):
        """Sequential step k (0 = least) and the text colour that reads on it."""
        if self.target == "html":
            return f"var(--q{k})", f"var(--qt{k})"
        c = seq_ramp(self.mode)[k]
        return c, _seq_text(c)


def text_w(s, size=11):
    """Approximate rendered width of system-ui text."""
    narrow = sum(1 for ch in str(s) if ch in "il.,:;|!'() ")
    return (len(str(s)) - narrow) * size * 0.56 + narrow * size * 0.3


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        if v >= lo - step * 1e-6:
            ticks.append(round(v, 10))
        v += step
    if not ticks or ticks[-1] < hi - step * 1e-6:          # the axis always covers the data
        ticks.append(round((ticks[-1] if ticks else start) + step, 10))
    return ticks


def fmt_num(v, pct=False, nd=None):
    if v is None:
        return "–"
    if pct:
        return f"{v * 100:.{1 if nd is None else nd}f}%"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}K"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    if nd is not None:
        return f"{v:.{nd}f}"
    return f"{v:.3f}".rstrip("0").rstrip(".") if a < 10 and a != int(a) else f"{v:.0f}"


def _attrs(tip=None, link=None):
    """data-tip (hover text) and data-link (click target: the app opens the matching images) attributes."""
    return (f' data-tip="{escape(tip)}"' if tip else "") + (f' data-link="{escape(link)}"' if link else "")


def decimate(xs, ys, n_px):
    """Min/max decimation to about 2 points per pixel column: a 4-hour series stays a few thousand SVG points,
    and every spike (a burst of detections, a dip in agreement) is still drawn."""
    n = len(xs)
    if n <= 2 * n_px or n_px <= 0:
        return list(xs), list(ys)
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    b = np.minimum((np.arange(n) * n_px) // n, n_px - 1)
    edges = np.flatnonzero(np.diff(b)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [n]])
    ox, oy = [], []
    for a, e in zip(starts, ends):
        seg = ys[a:e]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted({i, j}):
            ox.append(float(xs[k]))
            oy.append(float(ys[k]))
    return ox, oy


class Svg:
    def __init__(self, w, h, ink, title=""):
        self.w, self.h, self.ink, self.parts = w, h, ink, []
        self.title = title

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=11, color="ink2", anchor="start", weight=400, tip=None, mono=False, link=None):
        """y is the text's vertical centre (baseline computed: QtSvg ignores dominant-baseline)."""
        fam = "ui-monospace, 'Cascadia Mono', Consolas, monospace" if mono else self.ink.font
        t = _attrs(tip, link)
        self.add(f'<text x="{x:.1f}" y="{y + size * 0.36:.1f}" text-anchor="{anchor}" '
                 f'style="font-family:{fam};font-size:{size}px;font-weight:{weight};fill:{self.ink.tok(color)}"{t}>'
                 f'{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, color="grid", width=1, dash=None, opacity=1):
        d = f";stroke-dasharray:{dash}" if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'style="stroke:{self._c(color)};stroke-width:{width}{d};stroke-opacity:{opacity}"/>')

    def _c(self, color):
        if isinstance(color, int):
            return self.ink.series(color)
        if isinstance(color, str) and color.startswith(("#", "var(", "rgb")):
            return color
        return self.ink.tok(color)

    def rect(self, x, y, w, h, color, tip=None, opacity=1, rx=0, link=None):
        t = _attrs(tip, link)
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, w):.1f}" height="{max(0, h):.1f}" rx="{rx}" '
                 f'style="fill:{self._c(color)};fill-opacity:{opacity}"{t}/>')

    def bar(self, x, y, w, h, color, horizontal, tip=None, r=4, link=None):
        """Bar with a rounded data end and a square base."""
        r = min(r, (h if horizontal else w) / 2, (w if horizontal else h))
        t = _attrs(tip, link)
        if w <= 0 or h <= 0:
            return
        if horizontal:      # grows right: round the right end
            d = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} "
                 f"A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")
        else:               # grows up: round the top end
            d = (f"M{x:.1f},{y + h:.1f} V{y + r:.1f} A{r},{r} 0 0 1 {x + r:.1f},{y:.1f} H{x + w - r:.1f} "
                 f"A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z")
        self.add(f'<path d="{d}" style="fill:{self._c(color)}"{t}/>')

    def polyline(self, pts, color, width=2, opacity=1, tip=None, fill=None):
        if len(pts) < 2:
            return
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        t = f' data-tip="{escape(tip)}"' if tip else ""
        self.add(f'<path d="{d}" style="fill:none;stroke:{self._c(color)};stroke-width:{width};'
                 f'stroke-linejoin:round;stroke-linecap:round;stroke-opacity:{opacity}"{t}/>')
        if fill is not None:
            base = fill
            d2 = d + f" L{pts[-1][0]:.1f},{base:.1f} L{pts[0][0]:.1f},{base:.1f} Z"
            self.add(f'<path d="{d2}" style="fill:{self._c(color)};fill-opacity:0.10;stroke:none"/>')

    def dot(self, x, y, color, r=4, tip=None):
        t = f' data-tip="{escape(tip)}"' if tip else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2}" style="fill:{self.ink.tok("surface")}"/>')
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{self._c(color)}"{t}/>')
        if tip:   # generous invisible hit area (>= 24 px)
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" style="fill:#000;fill-opacity:0" data-tip="{escape(tip)}"/>')

    def render(self):
        title = f"<title>{escape(self.title)}</title>" if self.title else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" class="viz" role="img" aria-label="{escape(self.title)}">{title}'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" style="fill:{self.ink.tok("surface")}"/>'
                + "".join(self.parts) + "</svg>")


def _legend(svg, names, x, y, kind="line", max_w=None):
    """Legend row(s): key mark (line or rect) + ink text. Returns height used."""
    cx, cy = x, y
    max_w = max_w or svg.w - x - 8
    for i, n in names:
        w = 18 + text_w(n, 11) + 16
        if cx + w > x + max_w and cx > x:
            cx, cy = x, cy + 18
        if kind == "line":
            svg.line(cx, cy, cx + 12, cy, i, 2)
        else:
            svg.rect(cx, cy - 5, 10, 10, i, rx=2)
        svg.text(cx + 17, cy, n, 11, "ink2")
        cx += w
    return cy - y + 18


# ------------------------------------------------------------------ line chart
def line_chart(series, x_label="", y_label="", x_range=(0, 1), y_range=(0, 1), w=560, h=320, ink=None,
               title="", markers=None, x_fmt=None, y_fmt=None, area=False, end_labels=True, legend=True):
    """series: [{name, color (slot), x: [...], y: [...]}]; markers: [{x, y, color, label}]."""
    ink = ink or Ink()
    svg = Svg(w, h, ink, title)
    x_fmt = x_fmt or (lambda v: fmt_num(v))
    y_fmt = y_fmt or (lambda v: fmt_num(v))
    lg = 0
    if legend and len(series) >= 2:
        lg = _legend(svg, [(s["color"], s["name"]) for s in series], 52, 12, "line", w - 60)
    yt = nice_ticks(*y_range)
    ml = max(36, max(text_w(y_fmt(t), 10) for t in yt) + 14) + (16 if y_label else 0)
    mr = 16 + (max((text_w(s["name"], 11) for s in series), default=0) + 10 if end_labels and len(series) <= 4 else 0)
    mt, mb = 14 + lg, 40
    pw, ph = w - ml - mr, h - mt - mb
    x0, x1 = x_range
    y0, y1 = min(yt[0], y_range[0]), max(yt[-1], y_range[1])
    X = lambda v: ml + (v - x0) / ((x1 - x0) or 1) * pw
    Y = lambda v: mt + ph - (v - y0) / ((y1 - y0) or 1) * ph
    for t in yt:
        svg.line(ml, Y(t), ml + pw, Y(t), "grid")
        svg.text(ml - 8, Y(t), y_fmt(t), 10, "muted", "end", mono=True)
    for t in nice_ticks(x0, x1, max(2, int(pw / 80))):
        svg.line(X(t), mt + ph, X(t), mt + ph + 4, "axis")
        svg.text(X(t), mt + ph + 14, x_fmt(t), 10, "muted", "middle", mono=True)
    svg.line(ml, mt + ph, ml + pw, mt + ph, "axis")
    if x_label:
        svg.text(ml + pw / 2, h - 10, x_label, 11, "ink2", "middle")
    if y_label:
        svg.add(f'<text transform="translate(12,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
                f'style="font-family:{ink.font};font-size:11px;fill:{ink.tok("ink2")}">{escape(y_label)}</text>')
    ends = []
    for s in series:
        pts = [(X(a), Y(b)) for a, b in zip(s["x"], s["y"]) if a is not None and b is not None]
        svg.polyline(pts, s["color"], 2, fill=(mt + ph) if area and len(series) == 1 else None,
                     tip=s.get("tip") or s["name"])
        if pts:
            ends.append((pts[-1][1], s))
    if end_labels and len(series) <= 4:
        ends.sort(key=lambda e: e[0])
        last = -1e9
        for yy, s in ends:
            yy = max(yy, last + 13)
            last = yy
            pt = [(X(a), Y(b)) for a, b in zip(s["x"], s["y"])][-1]
            if abs(yy - pt[1]) > 2:
                svg.line(pt[0] + 3, pt[1], ml + pw + 6, yy, "axis")
            svg.text(ml + pw + 8, yy, s["name"], 11, "ink2")
    for m in markers or []:
        svg.dot(X(m["x"]), Y(m["y"]), m["color"], 4, m.get("tip"))
        if m.get("label"):
            svg.text(X(m["x"]) + 8, Y(m["y"]) - 10, m["label"], 10, "ink2")
    return svg.render()


# ------------------------------------------------------------------ grouped bars
def bar_chart(categories, series, w=560, ink=None, title="", value_fmt=None, max_value=None, horizontal=True,
              bar=12, legend=True, value_labels=True, cat_links=None):
    """categories: [str]; series: [{name, color, values: [v per category], tips: [..], links: [..] optional}].
    Horizontal grouped bars (long class names stay readable). cat_links: click target per category label."""
    ink = ink or Ink()
    value_fmt = value_fmt or (lambda v: fmt_num(v))
    n = max(1, len(series))
    group = n * bar + (n - 1) * 2
    gap = 10
    lab_w = min(220, max([text_w(c, 11) for c in categories] + [40]) + 14)
    lg = 0
    tmp = Svg(w, 10, ink)
    if legend and n >= 2:
        lg = _legend(tmp, [(s["color"], s["name"]) for s in series], lab_w, 12, "rect", w - lab_w - 10)
    h = 14 + lg + len(categories) * (group + gap) + 30
    svg = Svg(w, h, ink, title)
    if legend and n >= 2:
        _legend(svg, [(s["color"], s["name"]) for s in series], lab_w, 12, "rect", w - lab_w - 10)
    vmax = max_value or max([v for s in series for v in s["values"] if v is not None] + [1e-9])
    ticks = nice_ticks(0, vmax, 4)
    vmax = max(vmax, ticks[-1])
    vals = [v for s in series for v in s["values"] if v is not None]
    vl = (max(text_w(value_fmt(v), 10) for v in vals) + 8) if value_labels and vals else 0
    pw = w - lab_w - 12 - vl
    top = 14 + lg
    X = lambda v: lab_w + v / vmax * pw
    for t in ticks:
        svg.line(X(t), top - 4, X(t), h - 26, "grid")
        svg.text(X(t), h - 14, value_fmt(t), 10, "muted", "middle", mono=True)
    for ci, c in enumerate(categories):
        gy = top + ci * (group + gap)
        svg.text(lab_w - 10, gy + group / 2, c if text_w(c, 11) < lab_w - 12 else c[: int((lab_w - 20) / 6)] + "…",
                 11, "ink2", "end", tip=c, link=(cat_links or [None] * len(categories))[ci])
        for si, s in enumerate(series):
            v = s["values"][ci]
            y = gy + si * (bar + 2)
            tip = (s.get("tips") or [None] * len(categories))[ci] or (
                f"{s['name']} · {c}: {value_fmt(v)}" if v is not None else f"{s['name']} · {c}: no data")
            if v is None:
                svg.text(lab_w + 4, y + bar / 2, "n/a", 10, "muted")
                continue
            link = (s.get("links") or [None] * len(categories))[ci]
            svg.bar(lab_w, y, max(0.0, X(v) - lab_w), bar, s["color"], True, tip, link=link)
            if v == 0:   # a hit area for the tooltip; no "0" label clutter (the table has it)
                svg.add(f'<rect x="{lab_w:.1f}" y="{y:.1f}" width="24" height="{bar}" style="fill:#000;fill-opacity:0" '
                        f'data-tip="{escape(tip)}"/>')
                continue
            if value_labels:
                svg.text(X(v) + 5, y + bar / 2, value_fmt(v), 10, "ink2", mono=True)
    svg.line(lab_w, top - 4, lab_w, h - 26, "axis")
    return svg.render()


def diverging_bars(categories, values, w=560, ink=None, title="", value_fmt=None, labels=("worse", "better")):
    """Signed differences (e.g. AP of B minus A per class): blue = positive, red = negative, grey zero line."""
    ink = ink or Ink()
    value_fmt = value_fmt or (lambda v: f"{v:+.3f}")
    bar, gap = 14, 8
    lab_w = min(220, max([text_w(c, 11) for c in categories] + [40]) + 14)
    h = 34 + len(categories) * (bar + gap) + 28
    svg = Svg(w, h, ink, title)
    m = max([abs(v) for v in values if v is not None] + [1e-6])
    pw = w - lab_w - 70
    cx = lab_w + 30 + pw / 2
    X = lambda v: cx + v / m * pw / 2
    svg.text(cx - 6, 14, f"← {labels[0]}", 10, "muted", "end")
    svg.text(cx + 6, 14, f"{labels[1]} →", 10, "muted", "start")
    for ci, (c, v) in enumerate(zip(categories, values)):
        y = 30 + ci * (bar + gap)
        svg.text(lab_w - 10, y + bar / 2, c, 11, "ink2", "end")
        if v is None:
            continue
        col = "pos" if v >= 0 else "neg"
        tip = f"{c}: {value_fmt(v)}"
        if v >= 0:
            svg.bar(cx, y, X(v) - cx, bar, col, True, tip)
            svg.text(X(v) + 5, y + bar / 2, value_fmt(v), 10, "ink2", mono=True)
        else:
            w_ = cx - X(v)
            r = min(4, bar / 2, w_)
            if w_ > 0:
                x = X(v)
                d = (f"M{cx:.1f},{y:.1f} H{x + r:.1f} A{r},{r} 0 0 0 {x:.1f},{y + r:.1f} V{y + bar - r:.1f} "
                     f"A{r},{r} 0 0 0 {x + r:.1f},{y + bar:.1f} H{cx:.1f} Z")
                svg.add(f'<path d="{d}" style="fill:{ink.tok(col)}" data-tip="{escape(tip)}"/>')
            svg.text(X(v) - 5, y + bar / 2, value_fmt(v), 10, "ink2", "end", mono=True)
    svg.line(cx, 24, cx, h - 20, "axis")
    return svg.render()


# ------------------------------------------------------------------ heatmap (confusion matrix)
def _seq_k(v):
    """0..1 -> sequential step index."""
    return min(len(SEQ) - 1, max(0, int(round(v * (len(SEQ) - 1)))))


def _lum(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def heatmap(matrix, row_labels, col_labels, ink=None, title="", normalize="row", cell=None, x_label="Predicted",
            y_label="True", max_w=760, link=None):
    """Confusion matrix: rows = true class, columns = predicted (background last). normalize: 'row' | None.
    Cells show the count; colour = share of the row (recall-style), so rare classes read as clearly as
    common ones."""
    ink = ink or Ink()
    import numpy as np
    m = np.asarray(matrix, float)
    n_r, n_c = m.shape
    lab_w = min(180, max([text_w(c, 11) for c in row_labels] + [40]) + 16)
    col_h = min(140, max([text_w(c, 10) for c in col_labels] + [30]) * 0.75 + 20)
    cell = cell or max(18, min(44, int((max_w - lab_w - 40) / max(1, n_c))))
    w = lab_w + n_c * cell + 30
    h = 30 + col_h + n_r * cell + 16
    svg = Svg(w, h, ink, title)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1) if normalize == "row" else m / max(m.max(), 1)
    top = 26 + col_h
    svg.text(lab_w + n_c * cell / 2, 12, x_label + " →", 11, "ink2", "middle")
    svg.add(f'<text transform="translate(12,{top + n_r * cell / 2:.1f}) rotate(-90)" text-anchor="middle" '
            f'style="font-family:{ink.font};font-size:11px;fill:{ink.tok("ink2")}">{escape(y_label)} →</text>')
    for j, c in enumerate(col_labels):
        x = lab_w + j * cell + cell / 2
        svg.add(f'<text transform="translate({x + 3:.1f},{top - 6:.1f}) rotate(-45)" text-anchor="start" '
                f'style="font-family:{ink.font};font-size:10px;fill:{ink.tok("ink2")}">{escape(c)}</text>')
    font = 10 if cell >= 30 else 8
    for i, r in enumerate(row_labels):
        y = top + i * cell
        svg.text(lab_w - 8, y + cell / 2, r, 11, "ink2", "end")
        for j in range(n_c):
            v, nv = m[i, j], norm[i, j]
            tip = f"True {r} → predicted {col_labels[j]}: {int(v)} ({nv * 100:.1f}% of {r})"
            if v == 0:
                svg.rect(lab_w + j * cell + 1, y + 1, cell - 2, cell - 2, "empty", tip, rx=2)
                continue
            col, tc = ink.seq(_seq_k(nv))
            svg.rect(lab_w + j * cell + 1, y + 1, cell - 2, cell - 2, col, tip, rx=2, link=link(i, j) if link else None)
            label = fmt_num(v) if v < 1e4 else f"{v / 1e3:.0f}K"
            if text_w(label, font) < cell - 4:
                svg.add(f'<text x="{lab_w + j * cell + cell / 2:.1f}" y="{y + cell / 2 + font * 0.36:.1f}" '
                        f'text-anchor="middle" style="font-family:{ink.font};font-size:{font}px;fill:{tc}" '
                        f'data-tip="{escape(tip)}">{label}</text>')
            if i == j:
                svg.add(f'<rect x="{lab_w + j * cell + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2}" height="{cell - 2}" '
                        f'rx="2" style="fill:none;stroke:{ink.tok("ink")};stroke-opacity:0.35;stroke-width:1"/>')
    return svg.render()


# ------------------------------------------------------------------ scatter (speed vs accuracy)
def scatter(points, x_label, y_label, w=560, h=320, ink=None, title="", x_fmt=None, y_fmt=None, better="upleft"):
    """points: [{name, color, x, y, tip}] - each point direct-labelled (identity is never colour alone)."""
    ink = ink or Ink()
    svg = Svg(w, h, ink, title)
    x_fmt = x_fmt or (lambda v: fmt_num(v))
    y_fmt = y_fmt or (lambda v: fmt_num(v))
    xs = [p["x"] for p in points if p["x"] is not None] or [0, 1]
    ys = [p["y"] for p in points if p["y"] is not None] or [0, 1]
    xt = nice_ticks(0, max(xs) * 1.15 or 1, 5)
    lo = min(ys)
    yt = nice_ticks(max(0.0, lo - (max(ys) - lo) * 0.3 - 0.02), min(1.0, max(ys) + 0.02) if max(ys) <= 1 else max(ys) * 1.1, 5)
    ml = max(text_w(y_fmt(t), 10) for t in yt) + 32
    mr, mt, mb = 20, 16, 40
    pw, ph = w - ml - mr, h - mt - mb
    X = lambda v: ml + (v - xt[0]) / ((xt[-1] - xt[0]) or 1) * pw
    Y = lambda v: mt + ph - (v - yt[0]) / ((yt[-1] - yt[0]) or 1) * ph
    for t in yt:
        svg.line(ml, Y(t), ml + pw, Y(t), "grid")
        svg.text(ml - 8, Y(t), y_fmt(t), 10, "muted", "end", mono=True)
    for t in xt:
        svg.line(X(t), mt + ph, X(t), mt + ph + 4, "axis")
        svg.text(X(t), mt + ph + 14, x_fmt(t), 10, "muted", "middle", mono=True)
    svg.line(ml, mt + ph, ml + pw, mt + ph, "axis")
    svg.text(ml + pw / 2, h - 10, x_label, 11, "ink2", "middle")
    svg.add(f'<text transform="translate(12,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
            f'style="font-family:{ink.font};font-size:11px;fill:{ink.tok("ink2")}">{escape(y_label)}</text>')
    placed = []
    for p in points:
        if p["x"] is None or p["y"] is None:
            continue
        x, y = X(p["x"]), Y(p["y"])
        svg.dot(x, y, p["color"], 5, p.get("tip") or f"{p['name']}: {x_fmt(p['x'])}, {y_fmt(p['y'])}")
        ly = y - 12
        while any(abs(ly - q) < 13 and abs(x - qx) < 120 for qx, q in placed):
            ly += 14
        placed.append((x, ly))
        anchor = "end" if x > ml + pw * 0.7 else "start"
        svg.text(x + (-9 if anchor == "end" else 9), ly, p["name"], 11, "ink", anchor, 500)
    return svg.render()


# ------------------------------------------------------------------ video timeline
def timeline(times, series, events=None, w=760, h=200, ink=None, title="", y_label="", y_max=None, y_fmt=None,
             legend=True):
    """Detections over time per model (2 px lines) + optional event rugs below the axis per model:
    events: [{t, color, label, tip}] drawn as short ticks."""
    ink = ink or Ink()
    svg = Svg(w, h, ink, title)
    y_fmt = y_fmt or (lambda v: fmt_num(v))
    lg = _legend(svg, [(s["color"], s["name"]) for s in series], 52, 12, "line", w - 60) if legend and len(series) >= 2 else 0
    rug = 10 * len({e["color"] for e in events or []})
    ymax = y_max or max([max(s["y"]) if len(s["y"]) else 0 for s in series] + [1])
    yt = nice_ticks(0, ymax, 4)
    ml = max(text_w(y_fmt(t), 10) for t in yt) + 18 + (16 if y_label else 0)
    mt, mb, mr = 14 + lg, 34 + rug, 28
    pw, ph = w - ml - mr, h - mt - mb
    t0, t1 = (times[0], times[-1]) if len(times) else (0, 1)
    X = lambda v: ml + (v - t0) / ((t1 - t0) or 1) * pw
    Y = lambda v: mt + ph - v / (yt[-1] or 1) * ph
    for t in yt:
        svg.line(ml, Y(t), ml + pw, Y(t), "grid")
        svg.text(ml - 8, Y(t), y_fmt(t), 10, "muted", "end", mono=True)
    from AIO_QC.tools.frame_extractor.core.sources import format_time
    for t in nice_ticks(t0, t1, max(2, int(pw / 110))):
        svg.line(X(t), mt + ph, X(t), mt + ph + 4, "axis")
        svg.text(X(t), mt + ph + 14 + rug, format_time(t)[:-4] if t1 - t0 > 60 else f"{t:.1f}s", 10, "muted",
                 "middle", mono=True)
    svg.line(ml, mt + ph, ml + pw, mt + ph, "axis")
    if y_label:
        svg.add(f'<text transform="translate(12,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
                f'style="font-family:{ink.font};font-size:11px;fill:{ink.tok("ink2")}">{escape(y_label)}</text>')
    for s in series:
        xs, ys = decimate(times, s["y"], int(pw))
        svg.polyline([(X(a), Y(b)) for a, b in zip(xs, ys)], s["color"], 1.5 if len(series) > 1 else 2, tip=s["name"])
    # event rugs: one tick per row per 3-px bucket (thousands of events stay a few hundred marks)
    rows, buckets = {}, {}
    for e in events or []:
        r = rows.setdefault(e["color"], len(rows))
        key = (r, int(X(e["t"]) / 3))
        b = buckets.get(key)
        if b is None:
            buckets[key] = [e, 1]
        else:
            b[1] += 1
    for (r, _), (e, n) in buckets.items():
        y = mt + ph + 6 + r * 10
        x = X(e["t"])
        svg.line(x, y, x, y + 7, e["color"], 2)
        tip = e.get("tip")
        if tip:
            tip = tip + (f" (+{n - 1} more here)" if n > 1 else "")
            svg.add(f'<rect x="{x - 4:.1f}" y="{y - 1:.1f}" width="8" height="10" '
                    f'style="fill:#000;fill-opacity:0"{_attrs(tip, e.get("link"))}/>')
    return svg.render()


def histogram_lines(edges, series, w=560, h=260, ink=None, title="", x_label="Confidence", y_label="Share of detections"):
    """Distributions as frequency polygons (lines overlay cleanly; bars of several models would not)."""
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    norm = []
    for s in series:
        tot = sum(s["counts"]) or 1
        norm.append(dict(name=s["name"], color=s["color"], x=centers, y=[c / tot for c in s["counts"]]))
    ymax = max([max(n["y"]) for n in norm] + [0.05])
    return line_chart(norm, x_label, y_label, (0, 1), (0, ymax), w, h, ink, title,
                      y_fmt=lambda v: f"{v * 100:.0f}%", end_labels=False)


def kpi(value, label, sub="", ink=None, w=180, h=76):
    """A stat tile as SVG (used in the app; the report uses HTML tiles)."""
    ink = ink or Ink()
    svg = Svg(w, h, ink, label)
    svg.text(12, 18, label, 11, "ink2")
    svg.text(12, 44, value, 24, "ink", weight=600)
    if sub:
        svg.text(12, 66, sub, 10, "muted")
    return svg.render()
