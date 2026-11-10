# Changelog

## 2.3.0 — Recommended checks, screenshots, Keep previous zoom and brightness / contrast, Relationship Rules reworked

Changes since 2.2.0.

### What's new

- **Recommended checks.** When a dataset opens, the app works out which checks matter most for *this* dataset and
  says where to start ("Start here: Label Ambiguities — 'Car ' / 'car' look like one class"). Reasons come from the
  data: unreadable annotation files, broken images, label names that differ only by case / spaces or are missing from
  the class list, empty files, splits and rare classes, video-frame names (→ Duplicate Images, Sequence Check), classes
  that almost always lie inside another (→ Relationship Rules), texts on shapes, models set up, flagged files. A small
  background sample (≤ 300 files) keeps opening instant. A **Next check** button in the status bar runs the next one;
  every finished check suggests the one after it; *Review ▸ Recommended Checks…* lists them all (kept per dataset in
  `.aio_qc/guide.json`).
- **Screenshots** (Image QC): **E** saves the image exactly as you see it — the part on screen, brightness / contrast,
  the labels you ticked, the predictions layer, a sketch — at the screen's real resolution, cropped to the image, into
  the screenshots folder as `<dataset>/<sub-folders>/<file name>.png` (never overwriting). **Shift+E** asks for a name;
  also a toolbar button, *File ▸ Screenshot* and a right-click menu on the image (save, save as, copy, open folder).
- **Screenshots of many files** (*Review ▸ Flagged Screenshots*, Ctrl+Shift+Alt+E; *File list ▸ Save screenshots of N
  files*): all flagged / flagged in the current list / the selected files, each drawn on a hidden canvas exactly as
  Image QC shows it, in the background with progress and Cancel. View: **the same spot as now** or **the whole image**;
  a **caption strip** (file, review mark and time, the flag note) and each file's **sketch**; existing files: keep both /
  overwrite / skip.
- **View ▸ Keep Previous Zoom**: the next image opens on the same spot at the same magnification — same-size images
  keep the exact zoom and pixel, other sizes show the same part of the image; fit stays fit, home stays home.
- **View ▸ Keep Previous Brightness/Contrast** (also *Keep for next images* in the Display tab), labelme's rule: an
  image opens with its own adjustment from earlier in the session, else the previous image's, else unadjusted
  (brightness, contrast, gamma, auto contrast; rendered once, already adjusted).
- **Relationship Rules reworked:** one **Parent / part** rule per relationship (the part lies inside a parent, is not
  larger than it, min–max parts per parent, parts of one parent do not overlap; a part drawn bigger than its parent is
  reported with how much bigger); real geometry (actual outlines for convex shapes, sizes from the minimum-area
  rectangle); every rule takes any class or `*`; **✔ Apply proposed fix** (fit a part inside its parent, delete a stray
  part or the smaller of two duplicates, polygon ↔ rectangle — one backup, Undo, files marked Fixed).

### Added

- **Settings ▸ General ▸ Saving** in every mode: a default folder plus one each for Screenshots, Exports & lists,
  Reports, Images and Dataset outputs; *Start where I saved last*; *Forget the last folders*. Every Save / Export
  dialog follows these rules.
- **Fix in Editor ▸ Labels: "Show only selected labels in the editor"**: the editor opens with only those shapes; the
  others are written back exactly as they were, in their place (LabelMe and YOLO alike).
- New rules: *Must touch* (every A overlaps a B), *Not more than* (no more A than B per image), *Tilt limit*; Parent /
  part can require a **minimum size** and the **same orientation**; points / lines can be parts.
- Rule choices detected from the dataset (in the background, up to 5,000 files): under each field what the data does,
  with one click to *use* it (or *Use all suggestions*); a pair that does not go together is called out.
- Rule cards draw the shape at fault, the related shapes dashed and the **proposed fix** in green, with the measured
  values as tags (*wheel #1 · 107 % size · 87 % inside*); a *Proposed fix* column; filter results by rule (optional
  short **Name** per rule).
- *⚑ Flag files* with nothing ticked offers to flag every file found; flag notes name the rule and the exact problem;
  *Saved Views* get *⚑ Flag files* too.
- Status bar **⚑ N flagged** for the whole dataset — click it to list only the flagged files.
- One guideline **per relationship rule** (shown above that rule's results).
- *File list ▸ Copy file name / Copy file path*.
- Settings ▸ Interface ▸ *Recommend checks when a dataset opens* (on by default).

### Changed

- **File list in natural order, per folder and split:** `frame_2` before `frame_10`, case-insensitive; each folder lists
  its files, then its sub-folders; splits as train → val → test.
- *Never together* takes a list of classes; saved rules are converted automatically and a rule is never added twice;
  *Suggest from data* proposes one Parent / part rule per pair (also *Must touch* and *Not more than*) and leaves out
  *Requires* when a stronger rule implies it.
- Faster scans: files with none of the rules' classes are skipped unopened.
- **Faster adjustments on huge images:** brightness / contrast is one multi-threaded tone-curve pass (≈ 8×: 2.9 s →
  0.37 s on a 108 MP image); auto contrast is computed once per image; the high-quality zoomed view reads the
  pixmap's own pixels (0.37 s → 0.03 s per new 108 MP image).
- Guidelines list only checks that find labelling mistakes; example crops outline the shape itself.
- Screenshots saved with E go to the screenshots folder (default *Pictures/AIO QC Screenshots*).

### Removed

- The rules *Must be inside*, *Must contain*, *Children apart* (now one *Parent / part* rule), *Never together* for two
  classes and *Classes exclusive* (now one *Never together* list) — saved rules are converted automatically.
- Same-class duplicates as a rule (Duplicate Shapes finds them).

### Fixed

- Drawing a box over a shape kept that shape highlighted while dragging.
- Frame Extractor: a second Settings gear sat over the *File* menu.
- Windows in the background went stale: the background-task indicator kept showing a finished job; notifications
  posted while another mode was in front never started their countdown; a panel hidden while its window was in the
  background came back.
- Labels dock: turning a label off / on, *Show all labels* or *Show only this label* left the row highlighted.
