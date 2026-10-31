# Modifications to labelme-5.11.4 (as bundled with AIO QC)

This folder contains a **modified copy** of labelme 5.11.4 (https://github.com/wkentaro/labelme),
licensed under the GNU General Public License v3.0 (GPL-3.0-only); the original LICENSE file is kept unchanged in this folder.

Modified by the AIO QC project, October 2026. Changes relative to the upstream release:

- Trimmed to the files needed to run the program (examples, tests, docs, build files removed).
- Added `aioqc_source.json` (metadata: version, command-line flags, dependencies, patches) and this file.
- `labelme/config/__init__.py`: settings file ~/.labelme5rc (not the shared ~/.labelmerc)
- `labelme/app.py`: settings file ~/.labelme5rc (not the shared ~/.labelmerc)

AIO QC starts this program as a separate process and exchanges annotation files with it; it does not
link to it. The complete corresponding source of this modified version is this folder itself.
