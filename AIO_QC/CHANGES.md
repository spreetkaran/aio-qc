# Changelog

## 1.0.1 — Text Labels and Training Size; annotation import and image picking in Data Prep

Changes since 1.0.0.

### What's new

- **Text Labels** (Labels tab, Ctrl+Alt+T): checks the text **stored** in your annotations (LabelMe `description`
  or another shape field) without an OCR model and without assuming any kind of text.
  - Rules: the validators you choose (optional; the same library as OCR Check, plus your own), and what the app
    learns from the open dataset, per class — the formats most texts share (only with ≥ 10 different texts, ≥ 80 %
    of them fitting), the texts that recur, the characters used (any script) and the characters its frames confuse.
  - Text classes are found in the data. Each finding (no text, rule failed, unusual format, look-alike of a frequent
    text, differs from other frames, not normalised) comes with a correction, cheapest first: what the other frames
    agree on · a frequent look-alike text · one look-alike character · a known text one or two characters away ·
    any one character.
  - Sure corrections are applied in one click; otherwise you pick. Writes are re-checked against the file, backed up
    and undoable.
- **Training Size** (Shapes tab, Ctrl+Alt+Z): every object measured at the training size (longer side = imgsz, as in
  Ultralytics) and after scale augmentation. Lists the objects too small to learn and those at risk once augmentation
  shrinks them; per class, the imgsz that keeps 95 % of the class above the limit, with advice (train larger,
  crop / tile, SAHI). The imgsz defaults to your model's.

### Added

- **Model Check ▸ Accept by confidence…**: accepts every missing-box, class-mismatch or poor-fit finding at or above a
  confidence you choose, in one go, with a live count per kind. Rejected findings and deletions are never included;
  the change is backed up and undoable.
- **Fix EXIF rotation** (Image Quality ▸ ⟳ Fix EXIF rotation, and Data Prep): the rotation tag is baked into the
  pixels (other EXIF data and the colour profile kept) and LabelMe / YOLO labels are brought into the same upright
  frame. Per file, the label size tells whether the labels were drawn on the raw or the upright picture. Every file is
  kept first; Undo restores it byte for byte.
- **Data Prep ▸ Import annotations**: COCO (boxes, polygons, RLE masks), Pascal VOC, CVAT for images / video (rotated
  boxes, ellipses, masks, attributes) and Label Studio (rotated boxes, polygons, points, ellipses; text areas become
  the description), into a new LabelMe or YOLO detection / segmentation folder. Images are matched by path, by name,
  or by name without Label Studio's upload prefix, and hard-linked or copied.
- **Data Prep ▸ Pick images to label**: picks the most varied set from an unlabelled pool (greedy k-center, so
  near-identical frames are not picked twice). Optionally prefers images unlike the open dataset's and those where its
  model is unsure. Images are compared by built-in features (no model needed) or the Visual Similarity model; the
  picks are copied or moved with a ranked `picked.csv`.
- Data Prep runs can undo in-place rewrites (used by the rotation fix) — restored only if the file is unchanged since.

### Changed

- Package metadata follows PEP 639: `license = "MIT"` (SPDX) and `license-files`; the deprecated
  `License ::` classifier is gone, so building for PyPI gives no deprecation warnings. Requires setuptools ≥ 77 to build.
- Clearer PyPI / GitHub description; project links point to github.com/spreetkaran/aio-qc.
- Every source file carries an SPDX header (`SPDX-FileCopyrightText` / `SPDX-License-Identifier: MIT`); the full texts
  of bundled third-party licenses are in `licenses/`.

### Fixed

- The PyPI page now shows the README's pictures (absolute image links).
- README: `aio-qc --version` (was `AIO_QC --version`) and the PyInstaller command's `${AIO_QC_PATH}`.
