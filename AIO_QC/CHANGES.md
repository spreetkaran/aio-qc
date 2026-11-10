# Changelog

## 2.2.0 — AIO QC App: Paint-style status bar and zoom, notifications that open in place, distinct label colours

Changes since 2.1.0.

### What's new

- **The app is now called AIO QC App** (window title, About, command line).
- **Status bar** (Image QC): pointer position on the image (`549, 353px`), the size of the selected shape /
  the box being drawn / the ROI being dragged (`52 × 20px`, exact bounds in its tooltip), the image size, the file type
  (`JPG`) and the file size (`139.8 KB`, exact bytes in the tooltip) — each with its icon
  in a fixed slot; messages follow them and no longer hide them.
- **Zoom:**
  - zoom in / out buttons: a click is one ×1.25 step; press and hold zooms continuously, easing in to a steady 2.5× per
    second (computed from the elapsed time on every display frame) and easing out on release; the centre stays put;
  - zoom levels in 10-point steps to 200 %, 25-point steps to 600 %, 50-point steps above, snapping to that grid;
  - zoomed-in images are no longer blocky above 300 %: while the view moves the image is smooth-filtered; once it rests
    the visible part is re-rendered at the screen's real resolution (bicubic enlarged, area-averaged reduced) and drawn
    1:1 on device pixels;
  - the zoom box and slider follow every step live; three quick clicks are three steps.
- **Label colours that stay apart:** a fixed palette built by farthest-point sampling in CIELAB over vivid colours that
  read on photos (the first 12 at least 30 ΔE apart); each new label takes the most distinct unused colour (class list
  first, then the most frequent labels); saved per dataset (`.aio_qc/label_colors.json`), so colours never change
  between sessions.
- **Notification details open inside the notification:** *Details* grows it in place (taller, and wider when needed)
  into a scrollable panel, anchored bottom-right; *Hide details* folds it back; its countdown waits while it is open.
  Used by "Updated from disk", the dataset-layout warning, broken images, storage housekeeping and exports.

### Added

- **Help ▸ About in every mode** (Image QC, Frame Extractor, Data Studio), laid out like LabelMe's: icon, name, a
  one-line description, *Version*, **Author: Preetkaran Singh** and a *Source : GitHub* link (the author is taken from the installed package's metadata; a
  test fails if the author in About, pyproject.toml, the README or the notices ever differs).
- An SPDX header (`SPDX-FileCopyrightText` / `SPDX-License-Identifier: GPL-3.0-only`) in every source file — bundled
  third-party code keeps its own notices; `packaging/add_spdx.py` adds missing headers (`--check` only reports; a test
  runs it).
- *Settings ▸ Viewer ▸ Show pixels when zoomed in* brings back hard pixel squares for pixel-level work.
- A logarithmic zoom slider in the toolbar (every step the same ratio; zooms live while dragged).
- App icons ship 16–512 px; colour dots, file-list badges, Testing's icons and the eyedropper cursor are drawn at the
  screen's real resolution (sharp at 125 %, 150 %, 4K, 8K).

### Changed

- **Window titles follow one pattern:** *Image QC — dataset — AIO QC App*, *Frame Extractor — video — AIO QC App*,
  *Data Studio — workspace — AIO QC App* (Help and Get Started likewise); no version in any title (it is in About).
- The Frame Extractor icon is redrawn as a vector (SVG): sharp at every size and scaling.
- The mode switch shows the AIO QC App icon for Image QC (the Frame Extractor keeps its own), with a little space
  between icon and name.
- Zoom controls in the toolbar in Paint's order: after Drag / Select — zoom box, Fit Window, 1:1, zoom out, slider,
  zoom in.
- **Everything on the image keeps its screen size together:** outlines (Display ▸ Outline still scales them), point
  dots, keypoints of the predictions layer, label tags and their dots — zooming in never buries pixels under a thick
  line, and tags stay attached to thin outlines at every zoom.
- **Selecting a shape whose label is hidden** (greyed row, or a QC finding pointing at it) shows that shape while it
  is selected; clicking elsewhere — or the same row — hides it again. The dock and the image always agree.
- QC panel tabs look selected only while their tools are shown (drawer open or pinned) and hovered only while the mouse
  is on them.
- **Faster with many shapes** (same pictures, same tag placement): tags are rendered once and blitted while panning /
  selecting (≈ 2× faster frames with 1 500 shapes), tag placement ≈ 40 % faster, point shapes no longer rebuild a path
  for every bounds query, tag text is measured once per name.
- Get Started, notifications and the editor mini-window paint their icons instead of stretching small bitmaps.

### Fixed

- Selecting a shape whose label is hidden used to leave its row selected for good; hiding the label of the selected
  shape (or a shape that cannot be drawn) now clears the selection everywhere.
- Label colours often landed a few degrees apart (they came from a hash of the name).
- The old 1200 px Frame Extractor picture turned to mush at 16–96 px.
