# Third-party components bundled with AIO QC

AIO QC's own code is Copyright (c) 2026 Preetkaran Singh, under the **GNU General Public License v3.0** (GPL-3.0-only) in [`LICENSE`](LICENSE). The files below are **bundled** with it (in the
`AIO_QC/` package) and keep **their own licenses**; every full license text is in [`licenses/`](licenses/).

| Component | Where | License | Notes |
|---|---|---|---|
| **labelme** 6.3.1 and 5.11.4 (Kentaro Wada and contributors), <https://github.com/wkentaro/labelme> | `AIO_QC/editors/sources/labelme-*` | **GPL-3.0-only**, full text in each folder's `LICENSE` | Modified copies; the changes are listed in each folder's `MODIFICATIONS.md`. Used by *Fix in Editor*, run as a **separate program** (own process and environment); AIO QC exchanges files with it and does not link to it. |
| **labelImg** 1.8.6 (Tzutalin and contributors), <https://github.com/HumanSignal/labelImg> | `AIO_QC/editors/sources/labelimg-1.8.6` | **MIT**, in its `LICENSE` | Modified copy, see `MODIFICATIONS.md`. Run as a separate program. |
| **Phosphor Icons** (Phosphor Icons), <https://phosphoricons.com> | basis of several app icons (`AIO_QC/icons`) | **MIT**, `licenses/Phosphor-Icons-MIT.txt` | |
| **Inter** (The Inter Project Authors) | `AIO_QC/fonts/Inter_24pt-SemiBold.ttf` | **SIL OFL 1.1** | `licenses/Inter-OFL-1.1.txt` |
| **JetBrains Mono** (The JetBrains Mono Project Authors) | `AIO_QC/fonts/JetBrainsMono-Regular.ttf` | **SIL OFL 1.1** | `licenses/JetBrainsMono-OFL-1.1.txt` |
| **Source Code Pro** (Adobe; Reserved Font Name “Source”) | `AIO_QC/fonts/SourceCodePro-Regular.ttf` | **SIL OFL 1.1** | `licenses/SourceCodePro-OFL-1.1.txt`; unmodified, so the Reserved Font Name is kept |
| **AV Hershey Simplex** (Stewart C. Russell, from A. V. Hershey's stroke data) | `AIO_QC/fonts/AVHersheySimplexMedium.ttf` | **CC0 / WTFPL** (dual) | `licenses/AVHersheySimplex-CC0-WTFPL.txt` |

## Runtime dependencies (installed by pip, not bundled)

PyQt6 (GPL-3.0 / commercial, Riverbank Computing) · NumPy (BSD-3-Clause) · OpenCV (Apache-2.0) · loguru (MIT) ·
optional: natsort (MIT), Pillow (MIT-CMU), PyYAML (MIT), orjson (Apache-2.0 / MIT).

Model-based checks run your models in a separate environment that you choose; their libraries (for example PyTorch,
Ultralytics, PaddleOCR, transformers) are installed there under their own licenses and are not part of AIO QC.
