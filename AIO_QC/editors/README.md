# External editors (Fix in Editor)

`sources/` holds trimmed editor sources that ship with AIO QC (only what is needed to run them):

| Folder | Editor | Notes |
|---|---|---|
| `labelme-6.3.1` | labelme 6.3.1 | does **not** store the image inside the JSON by default |
| `labelme-5.11.4` | labelme 5.11.4 | **stores** the image inside the JSON by default |
| `labelimg-1.8.6` | labelImg 1.8.6 | YOLO boxes; patched (see below) |

Each folder has `aioqc_source.json` (kind, version, detected command-line flags, dependencies, patches).

## Adding a version
In AIO QC: *Fix in Editor ▸ Add source zip…* and pick the zip exactly as downloaded from GitHub, named like
`labelme-6_4_0.zip` / `labelImg-1_8_6.zip`. AIO QC finds the project inside, keeps only what is needed, reads the
supported command-line flags and dependencies, and stores it in your user folder
(`%LOCALAPPDATA%\AIO_QC\editors\sources` on Windows, `~/.local/share/aio_qc/editors/sources` on Linux). Several
versions can coexist; an added copy of a bundled version replaces it.

## How the editors run
* **labelme** runs from source through a tiny launcher (`aioqc_labelme_launcher.py`): the source's version comes
  from git tags, so `pip install` of a source zip fails. Only its dependencies are installed.
* **labelImg** runs as `python labelImg.py`. Patches applied on import (each must match exactly, or the source is
  rejected): removes the unused `distutils` import (Python 3.12+), casts floats to int in 7 Qt calls (Python 3.10+
  crashes), always saves YOLO (default was Pascal VOC XML), auto-save on.
* Environments: one per editor and Python version in `…/AIO_QC/editors/envs/`, shared by all datasets, reused.

## Licences
labelme is GPL-3.0, labelImg is MIT; their LICENSE files are kept in each folder. AIO QC starts them as separate
programs and exchanges files with them.
