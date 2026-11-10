# Changelog

## 2.1.0 — The Frame Extractor joins as a third mode; Testing in the Models tab

Changes since 2.0.0.

### What's new

- **FrameExtractor 1.0 is now part of AIO QC — the Frame Extractor mode.** The mode switch has three positions:
  Frame Extractor | Image QC | Data Studio (Ctrl+Alt+M cycles; the app starts in Image QC). One app: one theme, one
  settings store, one log, one crash handler, one animation setting, one model registry, one Help Center.
  - Every FrameExtractor 1.0 feature: player (GPU canvas, real-time pipeline, optional libmpv), markers and bookmarks,
    snap to sharpest, timeline analysis and suggestions, ROI and tracking, detections, batch queue (multi-process,
    resumable, mining, pre-annotations, cross-video de-duplication), review gallery, frame search, clip and
    annotated-video export, command palette, customisable shortcuts, Get Started and its guide chapters.
  - Loads on first use only: start-up time and memory are unchanged.
- **Testing** (Image QC ▸ Models, Ctrl+Alt+L) — FrameExtractor's Model Lab: test sets from videos, labelled datasets
  or videos only, reference model, leaderboard, charts, confusion, per class, failures, plots, A / B, videos, label
  fixes, projects, reports, annotated export. Works on **this dataset** by default (its files, labels, classes, and
  the models and class maps from Models & Environment); one prediction cache per dataset that **Model Check reuses**.
- **Video review — models on whole videos, in the player** (Testing ▸ 6 · Review videos): add videos (optional in /
  out points), see each model's details, and *Review in player* plays them one after another with every model's boxes.
  - Playback modes: **Balanced** (every frame live when the models keep up, otherwise an adaptive step with
    interpolation and backfill), **Best quality** (every frame; playback waits), **Fixed step**. Frames are decoded on
    the CPU with the player's decoder, so boxes belong to the exact frame on screen.
  - Review playlist panel: each video's state, Previous / Next, Loop, a switch per model, a live confidence slider; a
    paused frame always gets its own detections; at the end the video that was open before comes back.
  - A strip under the seek bar shows each model's detections per frame and the events (models disagree, flicker,
    uncertain); event list and Alt+[ / Alt+] jump between them.

### Added

- **libmpv installed automatically when missing**, in the background while the built-in engine plays (Windows: the
  LGPL `libmpv-2.dll`, ~30 MB; macOS: mpv from conda-forge; Linux: the distribution's package after asking once),
  with progress and Cancel; libmpv then takes over the open video. *Settings ▸ Playback ▸ libmpv…* shows the library
  in use or exactly why none can be used, with *Get libmpv automatically* and the manual ways.
- Command line: `aio-qc extract | probe | testset | eval | clip` (Qt-free);
  `aio-qc video.mp4` opens a video; extras `[video]` (PyAV, ffmpeg) and `[models]` (ONNX Runtime).
- **Open in Image QC** (File, gallery, batch queue): an extraction with its pre-labels becomes the dataset. Frames
  remember their video and frame: **Show Source Video in Frame Extractor** from Image QC.
- Testing: **Send to review** from Failures and Compare A / B flags the images in Image QC and filters the file list;
  label fixes go through AIO QC (backed up, in the file's history, flagged → fixed); "leave out of scoring" and the
  Testing project are stored with the dataset (`.aio_qc/`); old `.fxlab` projects still open.
- Video review: saved detections in the prediction cache (reused by Testing and the annotated export), class filter,
  **CSV** of every detection, **Send hard frames to Image QC** (frames where models disagree, flicker or hesitate,
  with LabelMe pre-labels), Video review defaults in Settings.
- **One Settings window per mode**, laid out like Labelme 7.8's: sections with icons and **Search settings**
  (Ctrl+F), one scrolling page, changes apply at once; ↺ per setting, reset per section, **Reset all**,
  **Export… / Import…** (JSON). Open it with the gear next to the mode switch, *File ▸ Settings…* (Ctrl+,) or the
  command palette.
- New settings — Data Studio: large-file sampling, profile sample, histogram bars, chart points, memory for
  intermediate results, Spark defaults; Image QC: storage retention, Get Started at start, temporary model environment.
- One **model registry** and one Python environment for .pt / .engine models (Image QC, Frame Extractor, Testing);
  one **background-task indicator** in every mode's status bar (with Cancel); one **Help Center** (F1 in every mode)
  with one shortcut page for all modes.
- Settings and data of the standalone FrameExtractor are taken over once (`~/.frame_extractor` → AIO QC's data folder).
- A real-Qt test suite (72 tests, offscreen).

### Changed

- **Menus decluttered:** they keep commands; preferences moved to Settings (every moved item keeps its shortcut and
  behaviour). Frame Extractor: the Settings menu, View ▸ Renderer / Animations, output format and naming, keyboard
  customisation. Image QC: compact toolbar, watch dataset, guideline cards, animations, auto-advance (now remembered),
  dataset cache, storage retention.
- Models & Environment is now **Ctrl+Alt+N** (Ctrl+Alt+M is the mode switch).
- The Frame Extractor keeps its own app icon and the GPU canvas by default; switching modes never fades its OpenGL
  window (that made it a layered window on Windows and forced CPU); an install that already fell back gets one fresh
  GPU attempt.
- The PyInstaller build includes pandas (Data Studio) and ONNX Runtime (in-app ONNX models).

### Removed

- Testing from the Frame Extractor's menus (it lives in Image QC ▸ Models).
- Duplicate menu items: "Switch to Image QC" in Data Studio's Help, "Command palette" in two menus, Storage… in two
  menus, the disabled hint line in Image QC's Edit menu.

### Fixed

- **libmpv plays** (it never did, also in FrameExtractor 1.0: "mpv could not play the file"): its render context is
  created before the file is opened; a one-time picture check falls back to mpv's basic renderer or to the built-in
  engine; a speed guard switches mpv to its own clock when the display's refresh can't be followed.
- .pt / .engine models failed to load with "cannot import name 'YOLO' from 'ultralytics'": the model server's own
  folder came first on the import path and AIO QC's `core/profile.py` shadowed Python's `profile` (needed by torch).
  Every model server now takes its folder off the path first. Models & Environment no longer reports "✅ 0/1 ready":
  failures show with their reason.
- A page change in Testing could raise "wrapped C/C++ object of type Transition has been deleted".
- The decoder benchmark could fail ("NoneType has no len") when a new video opened while the previous one was indexed.
- Frame Extractor menu items no longer contain line breaks ("Load⏎Video", "Book-⏎mark").
- Ctrl+Alt+M was bound to both Models & Environment and the mode switch (neither fired).
- Delayed calls scheduled twice with the same method fired only once and kept their windows alive.
- A random crash: the animation helper ran Python code from Qt's `destroyed` signal during teardown.
- An abort when a QC scan finished (its thread could be collected while still running).
- Data Prep pages read Qt widgets from the worker thread (they now get a snapshot).
- From FrameExtractor: a 3.12-only f-string broke Testing reports on Python 3.10 / 3.11; a finished Lab job could touch
  a closed window; the Lab's view thread leaked per window; Run / Build could cancel an unrelated task; video samples
  could exceed `MAX_SAMPLES` by one; two threads could write the same test-set frame; `eval -o folder` wrote a file
  without extension; batch-worker decoding is now explicitly CPU-only.
