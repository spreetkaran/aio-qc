# Modifications to labelimg-1.8.6 (as bundled with AIO QC)

This folder contains a **modified copy** of labelimg 1.8.6 (https://github.com/HumanSignal/labelImg),
licensed under the MIT License; the original LICENSE file is kept unchanged in this folder.

Modified by the AIO QC project, October 2026. Changes relative to the upstream release:

- Trimmed to the files needed to run the program (examples, tests, docs, build files removed).
- Added `aioqc_source.json` (metadata: version, command-line flags, dependencies, patches) and this file.
- `labelImg.py`: distutils was removed in Python 3.12 (the import is unused)
- `labelImg.py`: Qt needs an int (Python 3.10+)
- `labelImg.py`: Qt needs an int
- `libs/canvas.py`: Qt needs ints
- `libs/shape.py`: Qt needs ints
- `labelImg.py`: save YOLO .txt, never Pascal VOC .xml
- `labelImg.py`: auto-save on, like labelme --autosave

AIO QC starts this program as a separate process and exchanges annotation files with it; it does not
link to it. The complete corresponding source of this modified version is this folder itself.
