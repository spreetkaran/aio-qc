# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
AIO QC icon pipeline.

UI icons are duotone SVGs (Phosphor style: 256 grid, 16-unit strokes, 20 % tint fill, round caps/joins,
`currentColor` ink that the app replaces with the theme's text colour). Colour is reserved for meaning:

    ink (currentColor)  navigation / view / neutral
    BLUE   #1E88E5      files & data            GREEN  #43A047   OK / confirm / run
    ORANGE #FB8C00      flagged / warning       RED    #E53935   delete / error / stop
    PURPLE #8E24AA      model / AI

Usage
    python generate_icons.py icons                          # drawn SVGs + app icon rasters
    python generate_icons.py icons --sources <phosphor_dir> # also (re)normalise the supplied SVGs
"""
import argparse
import re
import sys
from pathlib import Path

INK = "currentColor"
BLUE, GREEN, ORANGE, RED, PURPLE = "#1E88E5", "#43A047", "#FB8C00", "#E53935", "#8E24AA"
SW = 16          # stroke width on the 256 grid


# =============================================================================
# 1 · Normalising supplied SVGs (Phosphor + custom) to one grid and currentColor
# =============================================================================
SUPPLIED = {     # app icon name <- source file (without .svg)
    "open": "folder-open", "prev": "arrow-fat-left", "next": "arrow-fat-right", "undo": "arrow-u-up-left",
    "zoom-in": "magnifying-glass-plus", "zoom-out": "magnifying-glass-minus", "fit-window": "frame-corners",
    "fit-width": "frame-arrows-horizontal", "eye": "eye", "delete": "trash", "edit": "note-pencil",
    "fill": "paint-bucket", "brightness-contrast": "brightness-contrast", "info": "info", "question": "question",
    "copy": "copy", "folders": "folders", "save": "floppy-disk", "image": "image-square", "layout": "layout-duotone",
    "review-clear": "x-circle", "ai-box": "ai-box", "ai-points": "ai-points", "file-x": "file-x",
    "shape-rectangle": "rectangle", "shape-polygon": "polygon", "shape-line": "line-segment",
    "shape-linestrip": "line-segments", "shape-circle": "circle", "circles-four": "circles-four",
    # reuse under tool / group names
    "tool-empty": "file-x", "tool-ambiguity": "question", "tool-dup-shapes": "copy",
    "tool-crop-gallery": "circles-four", "group-dataset": "folders", "group-shapes": "polygon",
}
NEUTRAL_INK = ("#333333", "#333", "rgb(51,51,51)", "#000000", "#000")
FORCE_INK = {"tool-empty", "file-x"}          # red in the source, but "empty" is not destructive
ACCENT = {"tool-model-check": PURPLE, "ai-box": PURPLE, "ai-points": PURPLE}   # sparkle / AI parts


def normalize_svg(text: str, force_ink=False) -> str:
    """Any grid -> 256, neutral greys/black -> currentColor, drop XML prolog / editor metadata."""
    text = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", text).strip()
    text = re.sub(r'\s(xmlns:(serif|xlink)|xml:space|serif:\w+|version)="[^"]*"', "", text)
    m = re.search(r'viewBox="([\d.\s-]+)"', text)
    vb = [float(v) for v in m.group(1).split()] if m else [0, 0, 256, 256]
    for c in NEUTRAL_INK:
        text = re.sub(re.escape(c) + r"(?![0-9A-Fa-f])", INK, text, flags=re.I)
    if force_ink:
        text = re.sub(r"#[0-9A-Fa-f]{6}|rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", INK, text)
    open_tag = re.search(r"<svg[^>]*>", text).group(0)
    inner = text[len(open_tag):text.rfind("</svg>")]
    fill = re.search(r'\sfill="([^"]+)"', open_tag)
    style = re.search(r'\sstyle="([^"]+)"', open_tag)
    scale = 256.0 / vb[2]
    body = inner if abs(scale - 1) < 1e-9 and vb[0] == 0 and vb[1] == 0 else \
        f'<g transform="scale({scale:g}) translate({-vb[0]:g} {-vb[1]:g})">{inner}</g>'
    attrs = f' fill="{fill.group(1)}"' if fill else ""
    if style:
        attrs += f' style="{style.group(1)}"'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"{attrs}>{body}</svg>\n'


def accent_sparkle(text: str, color: str) -> str:
    """Colour the sparkle (the 8-point star path) of ai-box / ai-points."""
    return re.sub(r'(<path d="M(?:128 84|196 32)[^"]*")', rf'\1 fill="{color}"', text)


# =============================================================================
# 2 · Drawn icons (same recipe)
# =============================================================================
def svg(*parts) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="none">'
            + "".join(parts) + "</svg>\n")


def duo(el: str, color=INK, tint=True, sw=SW) -> str:
    """el = '<rect .../>' etc. without paint attributes -> tint fill + outline."""
    head = el.rstrip()[:-2]
    out = f'{head} fill="{color}" fill-opacity="0.2" stroke="none"/>' if tint else ""   # fill-opacity: SVG Tiny safe
    return out + (f'{head} fill="none" stroke="{color}" stroke-width="{sw}" '
                  f'stroke-linecap="round" stroke-linejoin="round"/>')


def ln(d: str, color=INK, sw=SW) -> str:
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def dot(cx, cy, r, color=INK) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'


def P(d):  # path element helper for duo()
    return f'<path d="{d}"/>'


# ---- corner markers (always top-right, same size) -------------------------
def mk_sparkle(color=PURPLE):
    return f'<path d="M204 20 L214 46 L240 56 L214 66 L204 92 L194 66 L168 56 L194 46 Z" fill="{color}"/>'


def mk_x(color=RED):
    return ln("M180 32 L228 80 M228 32 L180 80", color, 18)


def mk_plus(color=GREEN):
    return ln("M204 28 V84 M176 56 H232", color, 18)


def mk_check(color=GREEN):
    return ln("M178 58 L198 78 L234 38", color, 18)


# ---- shapes -------------------------------------------------------------------
TAG = "M40 48 H120 L216 144 L136 224 L40 128 Z"


def tag(tx=0, ty=0, s=1.0, color=INK):
    g = duo(P(TAG), color) + dot(84, 92, 14, color)
    return f'<g transform="translate({tx} {ty}) scale({s})">{g}</g>' if (tx or ty or s != 1) else g


def paper(x1=48, y1=24, x2=208, y2=232, fold=48, color=INK, tint=True):
    d = f"M{x2 - fold} {y1} H{x1 + 8} a8 8 0 0 0 -8 8 V{y2 - 8} a8 8 0 0 0 8 8 H{x2 - 8} a8 8 0 0 0 8 -8 V{y1 + fold} Z"
    return duo(P(d), color, tint) + ln(f"M{x2 - fold} {y1} V{y1 + fold} H{x2}", color)


HAND = ("M72 148 V92 a16 16 0 0 1 32 0 V68 a16 16 0 0 1 32 0 V76 a16 16 0 0 1 32 0 V100 "
        "a16 16 0 0 1 32 0 V156 a72 72 0 0 1 -72 72 H126 a64 64 0 0 1 -52 -26 L40 156 "
        "a16 16 0 0 1 26 -18 Z")

DRAWN = {
    # --- modes
    "hand": svg(duo(P(HAND)), ln("M104 92 V132 M136 76 V132 M168 100 V132")),
    "cursor": svg(duo(P("M64 32 V200 L108 160 L136 224 L168 210 L140 146 L200 146 Z"))),
    # --- files / general
    "list": svg(*(dot(52, y, 14) for y in (64, 128, 192)), ln("M96 64 H216 M96 128 H216 M96 192 H216")),
    "logs": svg(paper(), ln("M88 112 H168 M88 144 H168 M88 176 H136"), ln("M88 208 H152", ORANGE)),
    "quit": svg(duo('<rect x="40" y="40" width="120" height="176" rx="8"/>'), ln("M136 128 H224 M192 96 L224 128 L192 160", RED)),
    "zoom": svg(duo('<rect x="32" y="48" width="192" height="160" rx="12"/>'),
                ln("M84 104 L100 92 V164 M156 104 L172 92 V164"), dot(128, 112, 9), dot(128, 148, 9)),
    "new": svg(duo('<rect x="40" y="40" width="176" height="176" rx="12"/>'), ln("M128 88 V168 M88 128 H168")),
    "export": svg(duo(P("M40 144 V200 a8 8 0 0 0 8 8 H208 a8 8 0 0 0 8 -8 V144"), BLUE, tint=False),
                  duo('<rect x="40" y="144" width="176" height="64" rx="8"/>', BLUE, sw=0).replace('stroke-width="0"', 'stroke-width="0" opacity="0"'),
                  ln("M128 40 V144 M88 80 L128 40 L168 80", BLUE)),
    "tag": svg(tag()),
    # --- review
    "review-ok": svg(duo('<circle cx="128" cy="128" r="96"/>', GREEN), ln("M88 132 L116 160 L172 100", GREEN)),
    "review-flag": svg(duo(P("M56 168 C104 128 152 208 216 168 V56 C152 96 104 16 56 56 Z"), ORANGE),
                       ln("M56 232 V40", ORANGE)),
    "review-note": svg(paper(40, 24, 196, 232, 44, ORANGE), ln("M76 112 H148 M76 144 H148 M76 176 H120", ORANGE)),
    "review-fixed": svg(duo('<circle cx="128" cy="128" r="96"/>', BLUE),
                        duo(P("M168 64 a44 44 0 0 0 -54 56 L68 166 a20 20 0 0 0 28 28 L142 148 "
                              "a44 44 0 0 0 56 -54 L172 120 L148 108 L136 84 Z"), BLUE, sw=14)),
    "restore-window": svg(duo('<rect x="140" y="140" width="84" height="84" rx="14"/>', BLUE, sw=16),
                          ln("M118 118 L40 40"), ln("M40 124 V40 H124")),
    "pip": svg(duo('<rect x="24" y="40" width="208" height="160" rx="14"/>'),
               duo('<rect x="124" y="120" width="88" height="62" rx="8"/>', BLUE, sw=14)),
    "tool-inbox": svg(duo(P("M32 136 L64 48 H192 L224 136 V200 a16 16 0 0 1 -16 16 H48 a16 16 0 0 1 -16 -16 Z")),
                      ln("M32 136 H88 l16 32 h48 l16 -32 H224"),
                      duo('<circle cx="196" cy="62" r="36"/>', RED, sw=12), ln("M196 42 V66", RED, 14), dot(196, 84, 8, RED)),
    "tool-rules": svg(duo('<rect x="28" y="36" width="96" height="72" rx="10"/>'),
                      duo('<rect x="132" y="148" width="96" height="72" rx="10"/>', BLUE),
                      ln("M76 108 V184 H132", BLUE), ln("M120 184 l12 0 M118 172 l14 12 l-14 12", BLUE)),
    "tool-quality": svg(duo('<rect x="24" y="40" width="208" height="160" rx="12"/>'),
                        ln("M40 132 H84 L104 84 L132 176 L156 112 L170 132 H216", RED, 16)),
    "tool-export": svg(duo(P("M32 80 a16 16 0 0 1 16 -16 H100 l20 24 H208 a16 16 0 0 1 16 16 V192 a16 16 0 0 1 -16 16 "
                             "H48 a16 16 0 0 1 -16 -16 Z")),
                       ln("M128 184 V112 M100 138 L128 108 L156 138", BLUE, 18)),
    "tool-class": svg(duo(P("M40 52 V124 L140 224 L224 140 L124 40 H52 a12 12 0 0 0 -12 12 Z"), BLUE),
                      dot(88, 88, 16, BLUE)),
    "tool-ocr": svg(ln("M40 88 V56 a16 16 0 0 1 16 -16 H88 M168 40 H200 a16 16 0 0 1 16 16 V88 "
                       "M216 168 V200 a16 16 0 0 1 -16 16 H168 M88 216 H56 a16 16 0 0 1 -16 -16 V168"),
                    duo('<rect x="72" y="96" width="112" height="28" rx="6"/>', BLUE, sw=12),
                    ln("M80 156 H176 M80 184 H144")),
    "tool-editor": svg(duo('<rect x="24" y="48" width="160" height="144" rx="10"/>'),
                       ln("M24 160 L76 108 L116 148 L140 124 L184 168"), dot(142, 90, 14),
                       duo(P("M226 110 L148 188 L120 196 L128 168 L206 90 Z"), BLUE, sw=14),
                       ln("M190 106 L210 126", BLUE, 14)),
    "review-next": svg(duo('<circle cx="128" cy="128" r="96"/>'), ln("M100 88 L148 128 L100 168 M172 88 V168")),
    "run": svg(duo('<circle cx="128" cy="128" r="96"/>', GREEN), duo(P("M104 84 L172 128 L104 172 Z"), GREEN)),
    "stop": svg(duo('<circle cx="128" cy="128" r="96"/>', RED), duo('<rect x="96" y="96" width="64" height="64" rx="6"/>', RED)),
    # --- tools
    "tool-stats": svg(duo('<rect x="48" y="120" width="40" height="88" rx="4"/>'),
                      duo('<rect x="108" y="56" width="40" height="152" rx="4"/>', BLUE),
                      duo('<rect x="168" y="92" width="40" height="116" rx="4"/>'), ln("M32 216 H224")),
    "tool-validation": svg(duo(P("M40 64 V120 c0 72 64 104 88 112 c24 -8 88 -40 88 -112 V64 a8 8 0 0 0 -8 -8 H48 a8 8 0 0 0 -8 8 Z")),
                           ln("M92 132 L116 156 L164 108", GREEN)),
    "tool-image-dups": svg(ln("M72 40 H208 a8 8 0 0 1 8 8 V184"),
                           duo('<rect x="40" y="72" width="144" height="144" rx="8"/>', BLUE),
                           ln("M40 180 L88 132 L128 172 L152 148 L184 180", BLUE), dot(144, 112, 12, BLUE)),
    "tool-label-map": svg(tag(0, 0, 0.62), ln("M120 176 H216 M184 144 L216 176 L184 208"),
                          ln("M40 200 H136 M72 168 L40 200 L72 232".replace("200 H136", "200 H96"), INK)),
    "tool-remove-class": svg(tag(0, 24, 0.82), mk_x()),
    "tool-outliers": svg(ln("M40 40 V216 H216"), *(dot(x, y, 12) for x, y in ((80, 176), (112, 144), (120, 188), (152, 164), (84, 128))),
                         duo('<circle cx="196" cy="68" r="20"/>', ORANGE)),
    "tool-dim-area": svg('<g transform="rotate(-45 128 128)">' + duo('<rect x="24" y="92" width="208" height="72" rx="8"/>')
                         + ln("M56 92 V124 M88 92 V112 M120 92 V124 M152 92 V112 M184 92 V124") + "</g>"),
    "tool-model-setup": svg(ln("M40 72 H216 M40 128 H216 M40 184 H216"),
                            duo('<circle cx="96" cy="72" r="20"/>'), duo('<circle cx="168" cy="128" r="20"/>', PURPLE),
                            duo('<circle cx="120" cy="184" r="20"/>')),
    "tool-sampling": svg(duo('<circle cx="128" cy="128" r="96"/>'), duo('<circle cx="128" cy="128" r="56"/>', tint=False),
                         dot(128, 128, 16, ORANGE)),
    "tool-smart-filter": svg(duo(P("M40 56 a8 8 0 0 1 8 -8 H208 a8 8 0 0 1 8 8 L152 136 V208 L104 232 V136 Z"))),
    "tool-coverage": svg(*(duo(f'<rect x="{x}" y="{y}" width="56" height="56" rx="8"/>',
                               GREEN if (i, j) in ((0, 0), (1, 0), (0, 1), (2, 0)) else (ORANGE if (i, j) == (1, 1) else INK),
                               tint=(i, j) != (2, 2))
                           for i, x in enumerate((36, 100, 164)) for j, y in enumerate((36, 100, 164)))),
    # --- groups
    "group-labels": svg(tag()),
    "group-models": svg(duo('<rect x="56" y="56" width="144" height="144" rx="16"/>', PURPLE),
                        duo('<rect x="96" y="96" width="64" height="64" rx="6"/>', PURPLE, tint=False),
                        ln("M96 24 V56 M160 24 V56 M96 200 V232 M160 200 V232 M24 96 H56 M24 160 H56 M200 96 H232 M200 160 H232", PURPLE)),
    "group-review": svg(duo('<rect x="48" y="40" width="160" height="192" rx="8"/>'),
                        duo('<rect x="88" y="24" width="80" height="40" rx="8"/>', tint=False),
                        ln("M88 136 L112 160 L168 104", GREEN)),
}
# the model-check icon: supplied ai-box with a purple sparkle (built from the source when available)


# =============================================================================
# 3 · App icon (raster): annotated image tile + QC magnifier
# =============================================================================
def app_icon(size: int, margin_frac: float):
    from PIL import Image, ImageDraw, ImageFilter
    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    m = int(S * margin_frac)
    tile = (m, m, S - m, S - m)
    rad = int((S - 2 * m) * 0.2)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(tile, rad, fill=255)
    # scene: dusk gradient sky, hills, sun  (flat, duotone-friendly colours)
    scene = Image.new("RGBA", (S, S))
    px = scene.load()
    top, bot = (44, 62, 110), (120, 150, 196)          # dusk sky
    for y in range(S):
        t = y / S
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)) + (255,)
        for x in range(S):
            px[x, y] = c
    d = ImageDraw.Draw(scene)
    d.ellipse((690, 170, 830, 310), fill=(255, 202, 110, 255))
    d.polygon([(0, 700), (230, 540), (430, 660), (640, 500), (1024, 720), (1024, 1024), (0, 1024)], fill=(72, 88, 128, 255))
    d.polygon([(0, 830), (300, 720), (560, 810), (1024, 700), (1024, 1024), (0, 1024)], fill=(52, 64, 98, 255))
    img.paste(scene, (0, 0), mask)
    # annotation polygon (labelme-like): translucent green fill, white outline, white vertices
    detail = size >= 48
    poly = [(200, 330), (430, 220), (610, 360), (590, 590), (350, 660), (170, 520)]
    ov = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.polygon(poly, fill=(76, 196, 80, 175))
    od.line(poly + [poly[0]], fill=(255, 255, 255, 255), width=18 if detail else 34, joint="curve")
    if detail:
        for x, y in poly:
            od.ellipse((x - 22, y - 22, x + 22, y + 22), fill=(255, 255, 255, 255), outline=(46, 125, 50, 255), width=8)
    img.alpha_composite(Image.composite(ov, Image.new("RGBA", (S, S), (0, 0, 0, 0)), mask))
    # QC magnifier with check (bottom-right), white with soft shadow
    mag = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    md = ImageDraw.Draw(mag)
    cx, cy, r = 720, 720, 128
    md.line((cx + r * 0.66, cy + r * 0.66, S - m - 80, S - m - 80), fill=(255, 255, 255, 255), width=62)
    md.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 70), outline=(255, 255, 255, 255), width=40)
    md.line((cx - 60, cy + 4, cx - 16, cy + 50, cx + 66, cy - 44), fill=(67, 160, 71, 255), width=40, joint="curve")
    shadow = mag.split()[3].filter(ImageFilter.GaussianBlur(14))
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sh.putalpha(shadow.point(lambda v: int(v * 0.45)))
    img.alpha_composite(Image.composite(sh, Image.new("RGBA", (S, S), (0, 0, 0, 0)), mask))
    img.alpha_composite(Image.composite(mag, Image.new("RGBA", (S, S), (0, 0, 0, 0)), mask))
    # thin inner rim so the tile edge reads on any background
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(rim).rounded_rectangle(tile, rad, outline=(255, 255, 255, 70), width=10)
    img.alpha_composite(rim)
    return img.resize((size, size), Image.LANCZOS)


def build_app_icons(out: Path):
    for s in (16, 32, 64, 128, 256):
        app_icon(s, 0.02 if s <= 32 else 0.03).save(out / f"app_{s}.png")
    app_icon(256, 0.03).save(out / "app.png")                     # canonical copy
    app_icon(200, 0.05).save(out / "app_200_border.png")          # transparent border (labelme style)
    app_icon(128, 0.03).save(out / "app.ico", sizes=[(128, 128)])


# =============================================================================
def main(out: Path, sources: Path | None):
    out.mkdir(parents=True, exist_ok=True)
    if sources is not None:
        for name, src in SUPPLIED.items():
            p = sources / f"{src}.svg"
            if not p.is_file():
                print(f"  missing source {p}", file=sys.stderr)
                continue
            text = normalize_svg(p.read_text(encoding="utf-8"), force_ink=name in FORCE_INK)
            if name in ACCENT:
                text = accent_sparkle(text, ACCENT[name])
            (out / f"{name}.svg").write_text(text, encoding="utf-8")
        ai = sources / "ai-box.svg"
        if ai.is_file():
            (out / "tool-model-check.svg").write_text(
                accent_sparkle(normalize_svg(ai.read_text(encoding="utf-8")), PURPLE), encoding="utf-8")
    for name, text in DRAWN.items():
        (out / f"{name}.svg").write_text(text, encoding="utf-8")
    build_app_icons(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(Path(__file__).parent))
    ap.add_argument("--sources", default=None, help="folder with the supplied Phosphor / custom SVGs")
    a = ap.parse_args()
    main(Path(a.out), Path(a.sources) if a.sources else None)
