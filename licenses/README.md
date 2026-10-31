# Third-party licenses

AIO QC's own code is under the MIT License in [`../LICENSE`](../LICENSE). The components below are **bundled** with
it and keep **their own licenses**; their full texts are in this folder. (The labelme, labelImg and Phosphor
texts also stay next to those components, as their licenses require.)

| Component | Bundled in | License | Text |
|---|---|---|---|
| labelme 6.3.1 and 5.11.4 (Kentaro Wada and contributors) — modified, used by *Fix in Editor* as a **separate program** | `AIO_QC/editors/sources/labelme-*` | GPL-3.0-only | [labelme-GPL-3.0.txt](labelme-GPL-3.0.txt) |
| labelImg 1.8.6 (Tzutalin and contributors) — modified, run as a separate program | `AIO_QC/editors/sources/labelimg-1.8.6` | MIT | [labelImg-MIT.txt](labelImg-MIT.txt) |
| Phosphor Icons (basis of several app icons) | `AIO_QC/icons` | MIT | [Phosphor-Icons-MIT.txt](Phosphor-Icons-MIT.txt) |
| Inter (The Inter Project Authors) | `AIO_QC/fonts/Inter_24pt-SemiBold.ttf` | SIL OFL 1.1 | [Inter-OFL-1.1.txt](Inter-OFL-1.1.txt) |
| JetBrains Mono (The JetBrains Mono Project Authors) | `AIO_QC/fonts/JetBrainsMono-Regular.ttf` | SIL OFL 1.1 | [JetBrainsMono-OFL-1.1.txt](JetBrainsMono-OFL-1.1.txt) |
| Source Code Pro (Adobe, Reserved Font Name “Source”) | `AIO_QC/fonts/SourceCodePro-Regular.ttf` | SIL OFL 1.1 | [SourceCodePro-OFL-1.1.txt](SourceCodePro-OFL-1.1.txt) |
| AV Hershey Simplex (Stewart C. Russell) | `AIO_QC/fonts/AVHersheySimplexMedium.ttf` | CC0 / WTFPL | [AVHersheySimplex-CC0-WTFPL.txt](AVHersheySimplex-CC0-WTFPL.txt) |

The labelme sources also contain the licenses of the AI models they reference (SAM, SAM 2, SAM 3, EfficientSAM,
YOLO-World, CLIP) under `AIO_QC/editors/sources/labelme-*/labelme/osam/_models/*/LICENSE`; they stay with those files.

Libraries installed by pip (PyQt6, NumPy, OpenCV, loguru, and the optional ones) are **not** bundled; they come
under their own licenses — see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
