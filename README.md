<p align="center">
  <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/logo/.readme/fx_icon_200x200.png" width="200" height="200" align="middle">
  &nbsp;&nbsp;<b>|</b>&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/logo/.readme/app_200_border.png" width="200" height="200" align="middle">
  &nbsp;&nbsp;<b>|</b>&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/logo/.readme/studio_icon_200x200.png" width="170" height="170" align="middle">
</p>

<h1 align="center">AIO QC</h1>

<h4 align="center">
  Find and fix annotation errors in LabelMe and YOLO datasets before you train — with a Frame Extractor for videos and Data Studio for tables, in one desktop app.
</h4>

<div align="center">
  <a href="https://pypi.org/project/aio-qc/3.0.0/"><img src="https://img.shields.io/badge/pypi-3.0.0-blue.svg"></a>
  <a href="https://pypi.org/project/aio-qc/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg"></a>
  <a href="https://github.com/spreetkaran/aio-qc/blob/v3.0.0/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg"></a>
  <a href="https://github.com/spreetkaran/aio-qc/releases/tag/v3.0.0"><img src="https://img.shields.io/badge/release-v3.0.0-informational.svg"></a>
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a>
  | <a href="#usage"><b>Usage</b></a>
  | <a href="#features"><b>Features</b></a>
  | <a href="#keyboard-shortcuts"><b>Shortcuts</b></a>
  | <a href="#faq"><b>FAQ</b></a>
  | <a href="https://github.com/spreetkaran/aio-qc/releases"><b>Download ↗</b></a>
</div>

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/main_app/.readme/main_light.png" width="80%">
</div>

## Description

AIO QC App is a desktop app for checking annotated image datasets **before you train**: broken files, wrong or
inconsistent label names, sloppy or duplicate shapes, split leakage and probable mislabels.\
It shows every problem on the image, lets you fix it in the same window, and keeps the original of every file it
changes, so any change can be undone. It also prepares datasets, trains and checks models, and records what made every annotation. Two more modes share the app: the **Frame Extractor** turns videos into frames worth annotating, and **Data Studio** analyses and cleans tables. It is written in Python and uses Qt for its graphical interface.

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/qc_drawer/.readme/qc_drawer.gif" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/review/.readme/review.gif" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/zoom/.readme/zoom.gif" width="32%" />\
<i>The QC drawer (Dataset, Labels, Shapes, Models and Review tabs), reviewing with auto-advance, and zoom.</i>

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/validation/.readme/validation.png" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/crop_gallery/.readme/crop_gallery.png" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/review_grid/.readme/review_grid.png" width="32%" />\
<i>Validation, Crop Gallery and Review Grid.</i>

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/convert_split/.readme/convert_split.png" width="49%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/preparation/.readme/preparation.png" width="49%" />\
<i>Convert &amp; Split planning a balanced train / val / test split, and Preparation adding a folder of new data to a dataset (matched by image content, previewed before anything changes).</i>

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/training/.readme/training.png" width="49%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/finetuning/.readme/finetuning.png" width="49%" />\
<i>Train Model: a YOLO run with live loss and val scores, then fine-tuning from its best.pt after new images were added (Fine-tuning preset: AdamW, small learning rate, backbone frozen).</i>

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/frame_extractor/.readme/fx_markers.gif" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/frame_extractor/.readme/fx_autopick.gif" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/frame_extractor/.readme/fx_review.png" width="32%" />\
<i>Frame Extractor: mark and extract frames, auto-pick the sharpest ones, review what was saved.</i>

<img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/stats/.readme/stats.png" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/modes/.readme/modes.gif" width="32%" /> <img src="https://raw.githubusercontent.com/spreetkaran/aio-qc/main/samples/main_app/.readme/main_dark.png" width="32%" />\
<i>Dataset Stats, the three modes, and the dark appearance.</i>

<sub>Images and the sample video are made from <a href="https://cocodataset.org">COCO</a> (COCO128 subset); annotations CC BY 4.0, images under their Flickr licenses.</sub>

## Features

### Image QC

- [x] LabelMe (v5 and v6, every shape type), YOLO (detection, segmentation, OBB, pose; flat or split) and COCO JSON (edited in place), plus image classification (class folders or LabelMe flags) — the layout is detected automatically
- [x] Ranked checks, worst first, each jumping straight to the problem on the image (`F8` for the next one); a **Health check** runs the recommended ones in one go:
  - **Dataset** — Preparation · Dataset Stats · Validation · Image–Label Pairs · Name Clashes · Empty Annotations · Duplicate Images · Image Quality · Split Health
  - **Labels** — Label Ambiguities · Label Mapping · Classification Check · Class Gallery · Guidelines · Text Labels · Remove / Re-order Classes
  - **Shapes** — Crop Gallery · Duplicate & Overlapping Shapes · Outliers · Sequence Check · Dim / Area · Training Size · Relationship Rules
  - **Models** — Model Check · OCR Check · Visual Similarity · Search in Words · Claude Review · Train Model · Improve Model · Image Augmentation · Testing
  - **Review** — Issues Inbox · Convert & Split · QC Sampling · Review Grid · Coverage Map · Smart Filter · Saved Views · QC Presets · QC Report · Hand-off Batches
- [x] Fix on the image: draw a bbox (`Ctrl+R`) or polygon (`Ctrl+N`), move, reshape, relabel, delete — with exact undo / redo; or hand files to LabelMe / labelImg with *Fix in Editor*
- [x] **Model Assist**: your model suggests labels — click to accept, right-click to drop; boxes can become outlines with SAM
- [x] **Search in Words**: "truck at night in rain" finds the matching images or objects with a local SigLIP 2 / CLIP model
- [x] **Claude Review**: Claude reads the image with its labels and says what is wrong, in words — Apply / Undo / Dismiss (an API key from an environment variable, or your signed-in Claude Code)
- [x] **Classification**: Class Gallery (most suspicious first, keyboard and drag-and-drop re-classing), Classification Check, classifier findings in Model Check
- [x] **QC Presets**: the same checks and limits on every delivery — PASS / FAIL per file (HTML + CSV)
- [x] Review marks (OK / flag / note) with auto-advance, Review Grid, screenshots, hand-off batches, labelling guidelines and file history
- [x] **Provenance**: every annotation, output folder, image and model AIO QC writes records what made it (manual edit, model, import, conversion, rule) — never a person; hover a shape to read it
- [x] Built for big datasets: cached indexing, background work, virtualised lists — fluid at millions of files and shapes

### Dataset preparation

- [x] **Preparation** (QC drawer ▸ Dataset, no dataset needed): unzip, import COCO / Pascal VOC / CVAT / Label Studio and more, combine & flatten, combine YOLO datasets, orphans, integrity, exact duplicates, EXIF rotation, resize, pick the images worth labelling — each previewed first and undoable
- [x] **Add new data / Add classes by image**: new data into an existing dataset, matched by image content (not file names), converted to its format, with a balanced split, a report and Undo
- [x] **Name Clashes**: files that would collide when flattened or copied, renamed in place with one Undo
- [x] **Convert & Split**: one window with the steps in a side pane — YOLO, LabelMe, COCO, masks, Pascal VOC, CVAT, DOTA, crops, class folders or CSV; the dataset's splits or a balanced train / val / test split that keeps duplicates and video frames together

### Models and training

- [x] **Train Model**: Ultralytics YOLO and classifiers (ResNet, EfficientNet, ConvNeXt, ViT, Swin, DINOv3 / DINOv2 / SigLIP 2 with a linear probe, any timm / torchvision model) — fine-tuning presets, live charts, stop / pause / resume
- [x] **Improve Model**: train → check → label more → retrain, round by round, with a history of what got better
- [x] **Extend Model**: teach a trained model more images or new classes without forgetting the old ones, with a before / after score per class
- [x] **Image Augmentation**: 74 effects with a live preview on your images; train with a recipe in memory or write an augmented copy
- [x] **Model Check** and **Testing**: report cards, confusion matrices, label-error estimates, leaderboards, A / B and video review
- [x] Models run in **their own Python environment** — the app itself never needs PyTorch or a GPU

### Frame Extractor

- [x] Video player made for picking frames: markers, snap to the sharpest frame, auto-pick, timeline analysis, ROI and tracking
- [x] Batch extraction (resumable, de-duplicated), review gallery, clip and annotated-video export
- [x] **Auto-Annotate** with your model, or **AI annotate** with a vision model from plain-word class descriptions
- [x] Open an extraction in Image QC as a dataset; jump from any extracted image back to its video and frame

### Data Studio

- [x] CSV / TSV, Excel, Parquet, Feather, JSON and SQLite; the open image dataset's annotations as a table
- [x] A recipe of undoable steps, shown as pandas / PySpark code and runnable on Spark
- [x] Column profile with one-click fixes, charts, an HTML profile report and exports

### The whole app

- [x] Three modes in one app (`Ctrl+Alt+M`), light and dark appearance, one Help Center (`F1`) and a Quick Tour
- [x] Every action searchable in the command palette (`Ctrl+K`)
- [x] Long jobs with time left, Stop and Undo; errors that say what happened and offer the fix

## Installation

There are 2 options to install AIO QC App:

### Option 1: Using pip

```bash
pip install aio-qc==3.0.0

# with every optional part (Data Studio, Frame Extractor video engine, ONNX models, faster JSON …):
pip install "aio-qc[full]"
# or only what you need: "aio-qc[data]", "aio-qc[video]", "aio-qc[models]", "aio-qc[augment]"
```

### Option 2: Using standalone executable (easiest)

Download the **standalone** (one program file) or **portable** (one folder, leaves nothing on the PC) zip for your system from the [v3.0.0 release](https://github.com/spreetkaran/aio-qc/releases/tag/v3.0.0) page — no Python needed.

### Supported Python and platforms

| | |
| ------ | ------------------------------ |
| Python | 3.10 – 3.13 |
| Qt | Qt6 (PyQt6) |
| OS | Windows / macOS / Linux |
| Required | PyQt6, NumPy, OpenCV, loguru |
| Optional | `[data]` pandas, pyarrow, openpyxl · `[video]` PyAV, imageio-ffmpeg, psutil · `[models]` ONNX Runtime · `[augment]` albumentations, imgaug · natsort, Pillow, PyYAML, orjson |

Model checks, training and Testing run your models in **their own Python environment** that you choose, so the app itself never needs PyTorch or a GPU.

## Usage

```bash
aio-qc                                # open the app
aio-qc-console                        # the same, with the log in a terminal
aio-qc --mode video                   # start in the Frame Extractor (or: --mode data)
aio-qc video.mp4                      # open a video; a .csv / .xlsx opens in Data Studio
aio-qc extract -h                     # the Frame Extractor's command line (extract | probe | testset | eval | clip | autoannotate)
aio-qc add-data -h                    # add new data to a dataset, matched by image content (or: add-classes)

aio-qc-console --version                      # print the version
aio-qc-console --logger-level debug           # verbose log in aio-qc.log
```

Click **Open Directory** (`Ctrl+O`) or drop a dataset folder on the window — or try **Help ▸ Open a Sample Dataset**. The recommended checks are suggested as soon as it loads. The switch in the menu-bar corner (`Ctrl+Alt+M`) changes mode; **`F1`** opens the user guide.

### Supported datasets

| Layout | Detected when | Notes |
| --- | --- | --- |
| LabelMe | `*.json` anywhere under the folder | v5 and v6, all shape types; sub-folders become filters; `imageData` optional |
| YOLO | `images/` + `labels/` | detection, segmentation, OBB, pose; `data.yaml` / `classes.txt` |
| YOLO split | `<split>/images` + `<split>/labels` | train / val / test, and any split name |
| COCO | a COCO annotation JSON | opened and edited in place |
| Classification: class folders | `<class>/<image>`, optionally under splits | changing a class moves the file |
| Classification: LabelMe flags | flags in LabelMe JSONs | multi-label flags supported |

Other formats (Pascal VOC, CVAT, Label Studio, KITTI, VIA, Supervisely, CreateML, Open Images, Datumaro, segmentation masks) are imported with **Preparation**.

### Keyboard shortcuts

Everything has a key; press **`?`** in the app for the live key map (type to filter, click to run).

| Keys | Action | Keys | Action |
| --- | --- | --- | --- |
| `D` / `A` | next / previous image | `S` / `X` / `U` | mark OK / flag / clear |
| `F8` / `Shift+F8` | next / previous finding | `Ctrl+K` | command palette |
| `Ctrl+R` / `Ctrl+N` | new bbox / polygon | `Ctrl+E` | rename / merge a label |
| `Ctrl+F` / `Ctrl+1` | fit / 100 % | double-click | zoom in ↔ back to fit |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo an edit | `Ctrl+H` | file history |
| `Alt+1…6` | open a QC tab | `F6` | next panel |
| `P` / `O` | sketch / ROI capture | `F11` | focus mode (image only) |
| `Ctrl+Alt+M` | next mode | `E` | screenshot of the view |

### Your data is safe

- **Nothing changes without a confirmation.** Views, statistics, label mappings and display adjustments never touch files.
- **Atomic writes:** a crash can never leave a half-written annotation.
- **Originals are kept** in `<dataset>/.aio_qc/backups.sqlite` before every rewrite (compressed, deduplicated) and restored byte for byte.
- **Deletes are moves** into `<dataset>/.aio_qc_trash/`.
- **Changes made by other programs are noticed** and re-read; the app never overwrites them.
- Everything the app stores for a dataset lives in `<dataset>/.aio_qc/` — delete that folder to reset it; your annotations are not in it.

### FAQ

- **Does it change my annotation files?** Only when you apply a fix, edit on the image, merge a batch or move files, and always after a confirmation. The original is stored first, so every change can be undone.
- **Do I need a GPU or PyTorch?** No. The core app needs neither. Model checks and training run in a separate Python environment that you point the app to (the app can create one), using a GPU when there is one.
- **My LabelMe JSONs embed the image (`imageData`). Is that supported?** Yes. Images are read from `imageData` when there is no image file, and *Preparation ▸ Combine & flatten* can restore the image files.
- **Where are the logs?** `aio-qc.log` in the folder you started the app from (the standalone app keeps it in its data folder's `logs/`); *Tools ▸ Logs* shows this session's errors. A native crash leaves `aio-qc-crash.log` there.

## How to build standalone executable

PyInstaller cannot cross-compile: build on the system you are building for.

```bash
AIO_QC_PATH=./AIO_QC
pyinstaller AIO_QC/__main__.py \
  --name=AIO-QC \
  --windowed \
  --noconfirm \
  --specpath=build \
  --add-data="${AIO_QC_PATH}/icons:AIO_QC/icons" \
  --add-data="${AIO_QC_PATH}/fonts:AIO_QC/fonts" \
  --add-data="${AIO_QC_PATH}/editors:AIO_QC/editors" \
  --add-data="${AIO_QC_PATH}/validators:AIO_QC/validators" \
  --onedir
```

## Development

With [uv](https://docs.astral.sh/uv/) (recommended — it installs the right Python and the exact versions in `uv.lock`):

```bash
git clone https://github.com/spreetkaran/aio-qc.git
cd aio-qc
git checkout v3.0.0
uv sync --all-extras                    # .venv with the app, every optional part and the dev tools
uv run aio-qc-console --logger-level debug
uv run pytest tests                     # the test suite (offscreen Qt); --cov for coverage
uv run ruff check                       # lint
```

With pip instead: `pip install -e ".[full]"`, then `pip install -e . --group dev` (pip 25.1+) for the test tools.

Issues and pull requests are welcome. Please include the app version (`aio-qc-console --version`), your OS and,
for bugs, the relevant part of `aio-qc.log`. Logic belongs in `core/` (Qt-free, testable); UI in `widgets/`.

## Acknowledgement

- [labelme](https://github.com/wkentaro/labelme) and [labelImg](https://github.com/HumanSignal/labelImg) — bundled
  (modified) as the external editors of *Fix in Editor*.
- [Phosphor Icons](https://phosphoricons.com) and [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons) — the basis of many of the app's icons.
- [Inter](https://github.com/rsms/inter), [Geist](https://github.com/vercel/geist-font),
  [Source Code Pro](https://github.com/adobe-fonts/source-code-pro) and AV Hershey Simplex — the bundled fonts.
- [COCO](https://cocodataset.org) — the sample images.

## License

AIO QC App is free software, released under the [GNU General Public License v3.0](LICENSE) (GPL-3.0-only).
Copyright (c) 2026 Preetkaran Singh.

It bundles third-party components that keep their own licenses. See
[THIRD_PARTY_NOTICES.md](https://github.com/spreetkaran/aio-qc/blob/v3.0.0/THIRD_PARTY_NOTICES.md); the full license texts
are in [`licenses/`](https://github.com/spreetkaran/aio-qc/tree/v3.0.0/licenses).
