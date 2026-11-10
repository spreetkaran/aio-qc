# Third-party components bundled with AIO QC

AIO QC's own code is Copyright (c) 2026 Preetkaran Singh, under the **GNU General Public License v3.0** (GPL-3.0-only) in [`LICENSE`](LICENSE). The files below are **bundled** with it (in the
`AIO_QC/` package) and keep **their own licenses**; every full license text is in [`licenses/`](licenses/).

| Component | Where | License | Notes |
|---|---|---|---|
| **labelme** 6.3.1 and 5.11.4 (Kentaro Wada and contributors), <https://github.com/wkentaro/labelme> | `AIO_QC/editors/sources/labelme-*` | **GPL-3.0-only**, full text in each folder's `LICENSE` | Modified copies; the changes are listed in each folder's `MODIFICATIONS.md`. Used by *Fix in Editor*, run as a **separate program** (own process and environment); AIO QC exchanges files with it and does not link to it. |
| **labelImg** 1.8.6 (Tzutalin and contributors), <https://github.com/HumanSignal/labelImg> | `AIO_QC/editors/sources/labelimg-1.8.6` | **MIT**, in its `LICENSE` | Modified copy, see `MODIFICATIONS.md`. Run as a separate program. |
| **Phosphor Icons** (Phosphor Icons), <https://phosphoricons.com> | basis of several app icons (`AIO_QC/icons`, `AIO_QC/tools/frame_extractor/icons`) | **MIT**, `licenses/Phosphor-Icons-MIT.txt` | |
| **Inter** (The Inter Project Authors) | `AIO_QC/fonts/Inter_24pt-SemiBold.ttf`, `Inter-*.otf`, `inter-*-web.woff` | **SIL OFL 1.1**, full text in `licenses/Inter-OFL-1.1.txt` and `AIO_QC/fonts/Inter-OFL.txt` | Used only for chart and plotted-image labels and HTML reports (Testing, Frame Extractor); the UI uses the system font. |
| **JetBrains Mono** (The JetBrains Mono Project Authors) | `AIO_QC/fonts/JetBrainsMono-Regular.ttf` | **SIL OFL 1.1** | `licenses/JetBrainsMono-OFL-1.1.txt` |
| **Source Code Pro** (Adobe; Reserved Font Name “Source”) | `AIO_QC/fonts/SourceCodePro-Regular.ttf` | **SIL OFL 1.1** | `licenses/SourceCodePro-OFL-1.1.txt`; unmodified, so the Reserved Font Name is kept |
| **AV Hershey Simplex** (Stewart C. Russell, from A. V. Hershey's stroke data) | `AIO_QC/fonts/AVHersheySimplexMedium.ttf` | **CC0 / WTFPL** (dual) | `licenses/AVHersheySimplex-CC0-WTFPL.txt` |

## Runtime dependencies (installed by pip, not bundled)

PyQt6 (GPL-3.0 / commercial, Riverbank Computing) · NumPy (BSD-3-Clause) · OpenCV (Apache-2.0) · loguru (MIT) ·
optional: natsort (MIT), Pillow (MIT-CMU), PyYAML (MIT), orjson (Apache-2.0 / MIT); Data Studio: pandas (BSD-3-Clause),
pyarrow (Apache-2.0), openpyxl (MIT); Frame Extractor `[video]`: PyAV (BSD-3-Clause; its FFmpeg libraries LGPL /
GPL depending on the build), imageio-ffmpeg (BSD-2-Clause; the bundled ffmpeg binary is GPL / LGPL), psutil
(BSD-3-Clause); `[models]`: ONNX Runtime (MIT). **libmpv** (optional playback engine of the Frame Extractor) is not
in the source package. When it is missing, the app downloads it into the user's data folder: Windows - the
LGPL-2.1+ mpv-dev-lgpl build of zhongfly/mpv-winbuild; macOS - conda-forge's mpv (GPL-2.0+) with its libraries; Linux
- the distribution's own package. `SOURCE.txt` in that folder names the origin and licence; py7zr (LGPL-2.1+) unpacks
the Windows download. App builds may carry it in `AIO_QC/tools/frame_extractor/lib/` when built with
`packaging/fetch_libmpv.py`: Windows - the LGPL-2.1+ mpv-dev-lgpl build of zhongfly/mpv-winbuild; macOS / Linux (only
with BUNDLE_LIBMPV=all) - GPL-2.0+ builds with their dependencies. `lib/LIBMPV_SOURCE.txt` in such a build names the
exact origin, version, checksum and licence; sources: https://github.com/mpv-player/mpv.

Model-based checks run your models in a separate environment that you choose; their libraries (for example PyTorch,
Ultralytics, PaddleOCR, transformers) are installed there under their own licenses and are not part of AIO QC.
