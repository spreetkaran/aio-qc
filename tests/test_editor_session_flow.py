# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Fix in Editor end to end with a stand-in editor: a Python that, when asked to start labelme, edits the session's
working copies the way a person would (relabels a shape, saves a broken file, saves one unchanged) and quits. Then
the window syncs every save back into the dataset, shows each file's state, undoes the session, and finishes."""
import json
import sys
from pathlib import Path

import pytest

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in editor is a shell script")

EDIT = r'''
import json, os, sys, time
work = sys.argv[sys.argv.index(next(a for a in sys.argv if a.endswith("launcher.py") or "launch" in a)) + 3]
files = sorted(f for f in os.listdir(work) if f.endswith(".json"))
time.sleep(0.6)
d = json.load(open(os.path.join(work, files[0])))
d["shapes"][0]["label"] = "truck"                                  # 1: a real fix
json.dump(d, open(os.path.join(work, files[0]), "w"))
time.sleep(0.6)
open(os.path.join(work, files[1]), "w").write("{ broken")         # 2: a save that is not valid JSON
time.sleep(0.6)
os.utime(os.path.join(work, files[2]))                            # 3: saved without changes
time.sleep(0.8)
'''


def _fake_editor(tmp: Path) -> str:
    (tmp / "fake_editor.py").write_text(EDIT)
    py = tmp / "editorpython"
    py.write_text(f'#!/bin/sh\ncase "$*" in *launch*) exec "{sys.executable}" "{tmp / "fake_editor.py"}" "$@" ;; '
                  f'esac\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    return str(py)


def test_fix_in_editor_session(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    keys = sorted(w.session.by_key)[:3]
    before = {k: Path(k).read_text() for k in keys}
    dlg = EditorSessionDialog(w, keys=keys)
    qtbot.addWidget(dlg)
    assert dlg.source is not None, "the bundled labelme source"
    dlg.settings.setValue(f"editors/{dlg.kind}/python", _fake_editor(tmp_path))
    dlg.settings.setValue(f"editors/{dlg.kind}/verified", [dlg.source.deps_hash()])
    dlg._refresh_env()
    i = next(n for n in range(dlg.scope.count()) if set(dlg.scope.itemData(n) or []) == set(keys))
    dlg.scope.setCurrentIndex(i)
    dlg._update_start()
    assert dlg.start_btn.isEnabled()
    dlg._start()
    qtbot.waitUntil(lambda: getattr(dlg, "es", None) is not None, timeout=20000)
    es = dlg.es
    assert all(k in w.session.locked for k in keys)                     # nobody else edits them meanwhile
    qtbot.waitUntil(lambda: dlg.proc is None or dlg.proc.state() == 0, timeout=30000)
    dlg._sync_tick()
    st = {Path(e.key).name: e.status for e in es.entries}
    names = [Path(k).name for k in keys]
    assert st[names[0]] == "fixed" and st[names[1]] in ("rejected", "error"), st
    assert json.loads(Path(keys[0]).read_text())["shapes"][0]["label"] == "truck"
    assert Path(keys[1]).read_text() == before[keys[1]]                 # the broken save never reached the dataset
    assert dlg.table.rowCount() == 3
    dlg._update_header()
    # undo the whole session: the dataset files are as they were
    dlg._undo()
    assert Path(keys[0]).read_text() == before[keys[0]]
    dlg._finish()
    assert not any(k in w.session.locked for k in keys)
    dlg.close()


EDIT_CLS = r'''
import json, os, sys, time
work = sys.argv[sys.argv.index(next(a for a in sys.argv if a.endswith("launcher.py") or "launch" in a)) + 3]
files = sorted(f for f in os.listdir(work) if f.endswith(".json") and f[:4].isdigit())
time.sleep(0.6)
d = json.load(open(os.path.join(work, files[0])))
cur = [k for k, v in d["flags"].items() if v][0]
other = next(k for k in d["flags"] if k != cur)
d["flags"] = {k: k == other for k in d["flags"]}                    # 1: give the image another class
json.dump(d, open(os.path.join(work, files[0]), "w"))
time.sleep(0.6)
d = json.load(open(os.path.join(work, files[1])))
d["flags"] = {k: True for k in d["flags"]}                          # 2: every class ticked: rejected
json.dump(d, open(os.path.join(work, files[1]), "w"))
time.sleep(0.8)
'''


def test_fix_in_editor_class_folders(shell, qtbot, tmp_path):  # noqa: F811
    """Class folders in the editor: each image is a LabelMe copy with one flag per class; ticking another class
    moves the image to that class folder (through AIO QC); two classes ticked is rejected."""
    from AIO_QC.core.dataset import DatasetType
    from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog
    root = FX.classification(tmp_path / "cls")
    w = shell.window("image")
    w.open_dataset(Path(root), DatasetType.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    s = w.session
    keys = sorted(s.by_key)[:2]
    cls_before = {k: s.tags_of(k) for k in keys}
    dlg = EditorSessionDialog(w, keys=keys)
    qtbot.addWidget(dlg)
    (tmp_path / "fake_editor.py").write_text(EDIT_CLS)
    py = tmp_path / "editorpython"
    py.write_text(f'#!/bin/sh\ncase "$*" in *launch*) exec "{sys.executable}" "{tmp_path / "fake_editor.py"}" "$@" '
                  f';; esac\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    dlg.settings.setValue(f"editors/{dlg.kind}/python", str(py))
    dlg.settings.setValue(f"editors/{dlg.kind}/verified", [dlg.source.deps_hash()])
    dlg._refresh_env()
    i = next(n for n in range(dlg.scope.count()) if set(dlg.scope.itemData(n) or []) == set(keys))
    dlg.scope.setCurrentIndex(i)
    dlg._update_start()
    dlg._start()
    qtbot.waitUntil(lambda: getattr(dlg, "es", None) is not None, timeout=20000)
    es = dlg.es
    assert es.class_folders and all(k in s.locked for k in keys)
    flags = json.loads((es.work / es.entries[0].mirror).read_text())["flags"]
    assert flags[cls_before[keys[0]][0]] is True and len(flags) == len(s.cls_classes())
    qtbot.waitUntil(lambda: dlg.proc is None or dlg.proc.state() == 0, timeout=30000)
    dlg._sync_tick()
    e0, e1 = es.entries
    assert e0.status == "fixed", (e0.status, e0.message)
    assert not Path(keys[0]).exists() and Path(e0.key).is_file()                 # the image moved
    assert Path(e0.key).parent.name != cls_before[keys[0]][0] and s.tags_of(e0.key) != cls_before[keys[0]]
    assert e1.status == "rejected" and Path(keys[1]).is_file()
    dlg._finish()
    assert not any(k in s.locked for k in (e0.key, keys[1]))
    dlg.close()
