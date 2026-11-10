# Icons

**UI icons: duotone SVG**, same recipe as AIO QC (Phosphor style): 256 grid, 16-unit strokes, 20 % tint fills,
round caps / joins. `currentColor` is replaced at paint time with the theme's text colour by
`utils.qt_utils.SvgIconEngine`, so one file serves light and dark themes and renders crisply at any size / DPI.

| Colour | Meaning | Examples |
|---|---|---|
| ink (`currentColor`) | transport, navigation, view | play, frame step, zoom, ROI, track, help |
| blue `#1E88E5` | files & data | open, save, output folder, gallery, batch, extract |
| green `#43A047` | run / confirm / extraction markers | run, marker, snap target |
| orange `#FB8C00` | marks & flags | bookmark, in / out, flagged samples |
| red `#E53935` | delete / stop / quit / clear | delete, stop, quit arrow, clear-ROI × |
| purple `#8E24AA` | models / AI | detect, models, environment, suggestions |

Corner markers (top-right): ✦ model/AI · × remove · + add · ✓ verified.

- Several icons are based on Phosphor Icons (MIT); the licence is in `licenses/Phosphor-Icons-MIT.txt` at the top of the repository.
- `icon.png` / `icon.ico` — the raster app icon.

In code: `utils.newIcon("name")` or `icon="name"` in `newAction`.
