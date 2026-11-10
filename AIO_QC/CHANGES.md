# Changelog

## 2.4.0 — Background Tasks window and loading ring; a steadier status bar; less memory

Changes since 2.3.0.

### What's new

- **Background Tasks** (new toolbar button, *Tools ▸ Background Tasks…*, or a click on the status bar's ring): every
  job of the session — QC scans, screenshots, exports, Data Prep, Testing, Frame Extractor batches, libmpv downloads —
  running ones first with live progress, details, start time, duration and Cancel, then the finished ones.
  *Clear finished* empties the list.
- **Loading animation in the status bar:** a turning blue → cyan ring with a soft glow, the job's text and progress;
  when the last job ends it turns into a green tick ("Done · …") for a moment. It sits first among the status-bar items,
  so its changing text never moves the others.
- **Less memory, same quality** (40 × 12 MP images browsed, zoomed, adjusted, four tools run: 1,039 MB → 707 MB, flat
  over repeated tool open / close): the decoded-image cache is capped at 512 MB and ~10 % of RAM; one shared budget for
  every image cache (≤ ~25 % of RAM, ≤ 2 GB); a mode in the background keeps only the image it shows; a leaner label
  index; no pre-loading below 700 MB free memory.

### Changed

- **Steady status bar:** the review mark is one short word in a fixed width — *Pending* (was "unreviewed"), *OK*,
  *Flagged*, *Fixed* — with sharp icons; a flag note shows as a note icon (text in the tooltip); the file position is
  as wide as the largest number the list can show. Nothing to their left moves from one image to the next.
- The review mark is painted at once when you go back to a file (it showed *unreviewed* for a moment).
- **Sharp icons instead of emoji** for the status marks beside *Open Dir* and *Load Classes*, the dataset name in the
  toolbar and Drag / Select in the status bar.
- Buttons that open a menu and already show ▾ get no second arrow; pressed toolbar buttons no longer shift their icon.
- Settings: a proper *back to default* icon replaces the ↺ character.
- Notifications: hiding *Details* shrinks the notification back to its own width.

### Removed

- The file count under the file list (the status bar shows *position / count*; with a filter "3 / 12 of 300").

### Fixed

- A background task whose window was closed mid-job could stay "running" forever.
- **Crashes when memory runs low** (`'NoneType' object has no attribute 'setsize'`, OpenCV "Insufficient memory" in
  the high-quality zoom view): the image stays on screen in the fast view, the high-quality render pauses for 15 s,
  cached neighbour images are freed and one notice says so; an image that fails to open for lack of memory says that
  instead of "Image not found".
- **Closed tool windows were never freed** — memory grew all session; they are now deleted when closed, and so are
  modal dialogs once they finish.
- Out of memory anywhere (also errors that only *wrap* one, like OpenCV's around numpy's "Unable to allocate") frees
  the caches and shows one notice instead of an unhandled-error report.
