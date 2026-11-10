# Changelog

## 2.0.0 — Two modes: Image QC and Data Studio; now GPL-3.0

Changes since 1.0.1.

### What's new

- **Two modes in one app: Image QC and Data Studio.** A switch in the menu-bar corner of both windows (and
  Ctrl+Alt+M) changes mode. Each mode is its own window with its own menus, panels, recent files and settings.
  - The app starts in the mode used last; `aio-qc --mode data` or `aio-qc file.csv` starts in Data Studio.
  - Data Studio never loads the image machinery unless you switch; closing either window quits the app, and both
    get to save first.
- **Data Studio** — table analysis, cleaning and feature engineering:
  - **Open:** CSV / TSV (separator and encoding sniffed), Excel (sheet), Parquet, Feather, JSON, JSON lines and
    SQLite. Files above 400 MB open as a sample; exports and Spark use the whole file. Drag and drop and recent files.
  - **Recipe:** every change is a step — switch off, edit, reorder, undo, click to see the data right after it.
    Results are cached per step.
  - **Steps:** columns (keep, remove, rename, change type, new column from a safe formula language, split) · rows
    (filter with 14 conditions, sort, sample, first rows) · clean (fill missing, remove missing, remove duplicates,
    replace, clean text, outliers) · reshape (group & summarise with 11 summaries, join) · features (bin, one-hot,
    scale, date parts).
  - **Profile:** per column type, missing, distinct, statistics, a mini chart and warnings (constant, mostly missing,
    IDs, numbers / dates stored as text, padded text, outliers, skew), each with a one-click fix; strong correlations
    and a quality score.
  - **Charts:** histogram, bar, line, scatter and box, colour by category, hover values, PNG export.
  - **Code:** the recipe as runnable pandas or PySpark code — the app runs exactly the pandas code it shows.
  - **Run on Spark:** the recipe on the full data in a Python environment with pyspark — locally, on a cluster
    (`--master`) or via Spark Connect. The app never imports Spark.
  - **Output:** CSV / Parquet / Excel / JSON, an HTML profile report, workspaces saved as small `.aioqc-data.json`
    recipes.

### Added

- *Tools ▸ Annotations → Data Studio*: the open dataset as a table, one row per shape.
- `pip install "aio-qc[data]"` for Data Studio (pandas, pyarrow, openpyxl); `[full]` includes it.
- A double-click on the empty band beside an image zooms into the nearest image edge.

### Changed

- **License: GNU GPL v3.0 (GPL-3.0-only)** from this version on (1.x was MIT). `LICENSE` holds the GPL text; the
  package metadata uses `license = "GPL-3.0-only"` and `license-files` (PEP 639, no deprecated `License ::`
  classifier); the third-party notices and the conda recipe say so too.
- **Class labels sit on their shape at every zoom:** a tag is attached to a point on the outline itself (top-most
  vertex first, then left-, right- or bottom-most; corners for rectangles, the rim for circles), its corner touching
  that point, marked by a dot in the class colour. When it has to move out to stay clear, a thin leader line runs back
  to the dot; no tag ever covers another shape's dot.
- **Zoom glide redone (double-click, Ctrl+wheel, fit, 100 %), Windows Photos style:** the end of the zoom is decided
  first and kept inside the image; the glide is a pure zoom around one point (every point moves in a straight line);
  the image stays smoothly filtered while gliding (very large images use the fast filter); with many tags on screen
  they settle once the glide lands.

### Fixed

- The zoom no longer corrects itself at the end, no longer bends towards an edge first when clicking near one (also
  on low-resolution images), and frames are no longer snapped back by Qt mid-glide.
