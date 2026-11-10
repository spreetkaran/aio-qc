# Icons

**UI icons: duotone SVG** (Phosphor style) — 256 grid, 16-unit strokes, 20 % tint fills, round caps / joins.
`currentColor` is replaced at paint time with the theme's text colour by `utils.qt_utils.SvgIconEngine`, so one
file serves light and dark themes and renders crisply at any size / screen scale. No PNG, @2x or dark copies.

Colour is reserved for meaning:

| Colour | Meaning | Examples |
|---|---|---|
| ink (`currentColor`) | navigation, view, neutral | prev / next, zoom, hand, cursor, most tools |
| blue `#1E88E5` | files & data | open, dataset tab, export, image duplicates |
| green `#43A047` | OK / confirm / run | review-ok, run, validation check |
| orange `#FB8C00` | flagged / warning | review-flag, review-note, outlier dot |
| red `#E53935` | delete / error / stop | delete, stop, remove-class marker, quit arrow |
| purple `#8E24AA` | model / AI | model check sparkle, models tab |

Corner markers (top-right, same size everywhere): ✦ model · × remove · + add · ✓ verified.

**App icon: raster** — `app_16/32/64/128/256.png`, `app.png` (256), `app.ico` (128), and `app_200_border.png`
(200 px with a transparent border, labelme style).

## Files
- `*.svg` — UI icons (65). `shape-*.svg` are also used, tinted in the label colour, in the Annotations dock.
- Several icons are based on Phosphor Icons (MIT); the licence is in `licenses/Phosphor-Icons-MIT.txt` at the top of the repository.

In code: `utils.newIcon("name")`, `icon="name"` in `newAction` / `newButton`, `utils.shapeIcon(type, color)`.
