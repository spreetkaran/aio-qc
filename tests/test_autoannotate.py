# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Auto-Annotate: the engine's five steps, re-runs, stop / continue, LabelMe 5.11.4 / 6.3.1 files (read back with each
version's own reader), the torch-free model reader, the command line and the window."""
import importlib
import importlib.util
import json
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest

from tests import fx_fixtures
from AIO_QC.tools.frame_extractor.core.autoannotate import engine as E
from AIO_QC.tools.frame_extractor.core.autoannotate.model_info import info as model_info
from AIO_QC.tools.frame_extractor.core.autoannotate.recipe import Recipe, STAGES

D = fx_fixtures.D
MODEL = os.path.join(D, "models", "dummy_a.onnx")
SOURCES = Path(__file__).resolve().parents[1] / "AIO_QC" / "editors" / "sources"


@pytest.fixture
def vids(tmp_path):
    d = tmp_path / "videos"
    (d / "cam2").mkdir(parents=True)
    shutil.copy(os.path.join(D, "a.mp4"), d / "a.mp4")
    shutil.copy(os.path.join(D, "b.mp4"), d / "cam2" / "b.mp4")
    return str(d)


def _recipe(**kw):
    base = dict(model=MODEL, step=5, stages=list(STAGES), keep_when="any_of", keep_classes=["plate"],
                group_by="video", split_size=20)
    base.update(kw)
    return Recipe(**base)


def _all_images(root):
    ws = E.Workspace(root)
    return ws.frame_images(), E.find_images(ws.similar), E.find_images(ws.extra)


# ------------------------------------------------------------------------------------------------ engine
def test_full_run_and_layout(vids, tmp_path):
    root = str(tmp_path / "run")
    res = E.run(root, _recipe(), source=vids)
    assert list(res) == list(STAGES)
    frames, similar, extra = _all_images(root)
    assert res["frames"]["saved"] == len(frames) + len(similar) + len(extra) > 0
    assert res["similar"]["similar"] == len(similar)
    assert res["rules"]["set_aside"] == len(extra)
    for p in frames:
        jp = os.path.splitext(p)[0] + ".json"
        d = json.loads(Path(jp).read_text(encoding="utf-8"))
        assert d["version"] == "6.3.1" and d["imageData"] is None
        assert d["imagePath"] == os.path.basename(p)
        assert list(d) == ["version", "flags", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth",
                           "provenance"]                    # what made the shapes (the model, each score)
        assert all(s["provenance"]["created"]["by"] == "model" for s in d["shapes"])
        assert any(s["label"] == "plate" for s in d["shapes"])
        for s in d["shapes"]:
            assert list(s) == ["label", "points", "group_id", "description", "shape_type", "flags", "mask",
                               "provenance"]
            (x1, y1), (x2, y2) = s["points"]
            assert 0 <= x1 < x2 <= d["imageWidth"] and 0 <= y1 < y2 <= d["imageHeight"]
    # package: a folder per video, parts of 20, zips with the part folder inside, parts.csv
    pkg = Path(root) / "package"
    assert sorted(p.name for p in pkg.iterdir() if p.is_dir()) == ["a", "b"]
    zips = sorted(pkg.rglob("*.zip"))
    assert zips and all(z.with_suffix("").is_dir() for z in zips)
    with zipfile.ZipFile(zips[0]) as z:
        names = z.namelist()
    assert all(n.startswith(zips[0].stem + "/") for n in names)
    assert sum(n.endswith(".json") for n in names) == sum(n.endswith(".jpg") for n in names) <= 20
    rows = (pkg / "parts.csv").read_text().splitlines()
    assert len(rows) - 1 == len(zips)
    assert (Path(root) / "reports" / "summary.html").is_file()
    st = E.Workspace(root).state()
    assert all(st["stages"][s]["status"] == "done" for s in STAGES)


def test_rerun_is_cheap_and_settings_changes(vids, tmp_path):
    root = str(tmp_path / "run")
    r = _recipe(stages=["frames", "annotate"])
    first = E.run(root, r, source=vids)
    again = E.run(root, r, source=vids)
    assert again["annotate"]["annotated"] == 0
    assert again["annotate"]["up_to_date"] == first["annotate"]["frames"]
    assert again["annotate"]["boxes"] == first["annotate"]["boxes"]          # totals cover the whole set
    frames = E.Workspace(root).frame_images()
    edited = os.path.splitext(frames[0])[0] + ".json"
    d = json.loads(Path(edited).read_text())
    d["shapes"] = d["shapes"][:1]
    Path(edited).write_text(json.dumps(d))
    foreign_img = frames[1]
    foreign = os.path.splitext(foreign_img)[0] + ".json"
    ws = E.Workspace(root)
    ws.db.execute("DELETE FROM anns WHERE img=?", (ws.rel(foreign_img),))
    ws.db.commit()
    ws.close()
    Path(foreign).write_text(json.dumps({"version": "5.2.1", "shapes": [], "flags": {}, "imagePath": "x",
                                         "imageData": None, "imageHeight": 1, "imageWidth": 1}))
    r.conf = 0.7
    res = E.run(root, r, stages=["annotate"])
    assert res["annotate"]["edited_kept"] == 1 and res["annotate"]["not_ours_kept"] == 1
    assert res["annotate"]["annotated"] == len(frames) - 2
    assert len(json.loads(Path(edited).read_text())["shapes"]) == 1
    assert json.loads(Path(foreign).read_text())["version"] == "5.2.1"
    r.existing = "replace"
    res = E.run(root, r, stages=["annotate"])
    assert res["annotate"]["annotated"] == len(frames)
    # frame settings changed: asks first, then replaces everything made from the old frames
    r.step = 7
    with pytest.raises(E.NeedsConfirmation):
        E.run(root, r, stages=["frames"], source=vids)
    assert len(E.Workspace(root).frame_images()) == len(frames)
    res = E.run(root, r, stages=["frames"], source=vids, replace_previous=True)
    now = E.Workspace(root).frame_images()
    assert len(now) == res["frames"]["frames"] != len(frames)
    assert not any(os.path.exists(os.path.splitext(p)[0] + ".json") for p in now)


def test_rules_and_similar_put_back_before_running_again(vids, tmp_path):
    root = str(tmp_path / "run")
    r = _recipe(stages=["frames", "similar", "annotate", "rules"], similar="loose")
    E.run(root, r, source=vids)
    f1, s1, x1 = _all_images(root)
    assert s1, "loose similarity sets something aside on the test videos"
    total = len(f1) + len(s1) + len(x1)
    r.keep_when, r.keep_classes = "all_of", ["car", "plate"]
    E.run(root, r, stages=["rules"])
    f2, s2, x2 = _all_images(root)
    assert len(s2) == len(s1) and len(f2) + len(s2) + len(x2) == total
    r.others = "keep"
    E.run(root, r, stages=["rules"])
    f3, _, x3 = _all_images(root)
    assert not x3 and len(f3) == len(f1) + len(x1)
    r.similar = "strict"
    E.run(root, r, stages=["similar"])
    f4, s4, x4 = _all_images(root)
    assert len(f4) + len(s4) + len(x4) == total and len(s4) <= len(s1)
    # the moved frames take their annotation files along
    for p in s4:
        assert os.path.exists(os.path.splitext(p)[0] + ".json")


def test_stop_and_continue(vids, tmp_path):
    root = str(tmp_path / "run")
    r = _recipe(stages=["frames"])
    E.run(root, r, source=vids)
    total = len(E.Workspace(root).frame_images())
    ctx = E.Context()
    seen = []

    def progress(stage, done, tot, text):
        seen.append(done)
        if stage == "annotate" and done > 0:
            ctx.cancel.set()
    ctx._progress = progress
    r.stages, r.batch = ["annotate"], 2
    with pytest.raises(E.Stopped):
        E.run(root, r, ctx=ctx)
    assert E.Workspace(root).stage_record("annotate")["status"] == "stopped"
    done = sum(os.path.exists(os.path.splitext(p)[0] + ".json") for p in E.Workspace(root).frame_images())
    assert 0 < done < total
    res = E.run(root, r)
    assert res["annotate"]["annotated"] == total - done and res["annotate"]["up_to_date"] == done


def test_images_source_links_and_never_touches_the_folder(tmp_path):
    src = tmp_path / "imgs"
    (src / "sub").mkdir(parents=True)
    import cv2
    import numpy as np
    for i in range(6):
        img = np.full((90, 160, 3), 30 * i, np.uint8)
        cv2.putText(img, str(i), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(str(src / ("sub" if i % 2 else "") / f"img_{i}.jpg"), img)
    before = sorted(str(p.relative_to(src)) for p in src.rglob("*"))
    root = str(tmp_path / "run")
    r = _recipe(source="images", stages=["frames", "annotate", "rules", "package"], keep_when="always",
                group_by="none", split_size=0, zip_parts=False)
    res = E.run(root, r, source=str(src))
    assert res["frames"]["images_found"] == 6 and res["annotate"]["frames"] == 6
    assert sorted(str(p.relative_to(src)) for p in src.rglob("*")) == before
    assert len(list((Path(root) / "package").glob("*.json"))) == 6


def test_shapes_classes_rename_drop_and_per_class_conf():
    names = {0: "car", 1: "plate", 2: "person"}
    r = Recipe(conf=0.5, min_box=5, classes={"0": {"keep": True, "name": "vehicle"}, "1": {"keep": True, "conf": 0.2},
                                             "2": {"keep": False}}, also_yolo=True, description="conf")
    rows = [(10, 10, 50, 50, 0.6, 0), (10, 10, 50, 50, 0.4, 0), (5, 5, 30, 30, 0.3, 1), (0, 0, 40, 40, 0.9, 2),
            (-20, -5, 400, 300, 0.9, 1), (1, 1, 3, 40, 0.9, 1)]
    shapes, yolo = E.shapes_for(rows, r, names, 200, 100)
    assert [s["label"] for s in shapes] == ["vehicle", "plate", "plate"]
    assert shapes[0]["description"] == "conf=0.600"
    assert shapes[2]["points"] == [[0.0, 0.0], [200.0, 100.0]]                 # clipped to the image
    assert [y[0] for y in yolo] == [0, 1, 1] and E.output_classes(r, names) == ["vehicle", "plate"]


def test_package_pattern_and_part_names(vids, tmp_path):
    root = str(tmp_path / "run")
    E.run(root, _recipe(stages=["frames"]), source=vids)
    r = _recipe(group_by="pattern", group_pattern=r"^(\w)_frame", split_size=10, part_names="start",
                zip_parts=False)
    plan = E.plan_package(E.Workspace(root), r)
    assert set(plan) == {"a", "b"}
    res = E.run(root, r, stages=["package"])
    parts = sorted(p.name for p in (Path(root) / "package" / "a").iterdir())
    assert parts[:2] == ["part_0", "part_10"] and res["package"]["zips"] == 0
    r.group_pattern = r"nomatch(\d+)"
    assert set(E.plan_package(E.Workspace(root), r)) == {E.UNMATCHED}


def test_problems_explain_what_is_missing():
    r = Recipe(stages=["annotate", "rules", "package"], keep_when="any_of", keep_classes=[], group_by="pattern",
               group_pattern="(", package_images=False)
    p = " ".join(r.problems({0: "car"}))
    assert "choose a model" in p and "classes a frame must have" in p and "regular expression" in p and \
        "imageData" in p
    r2 = Recipe(stages=["rules"], keep_when="any_of", keep_classes=["truck"])
    assert any("truck is not among" in x for x in r2.problems({0: "car"}))
    assert Recipe.from_dict(Recipe(model="m.pt", conf=0.4).to_dict() | {"unknown": 1}).conf == 0.4


# ------------------------------------------------------------------------------------------------ LabelMe readers
def _labelme_reader(version):
    src = str(SOURCES / f"labelme-{version}")
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == "labelme" or k.startswith(("labelme.", "PyQt5"))}
    for k in saved:
        sys.modules.pop(k)
    pkg = types.ModuleType("labelme")
    pkg.__path__ = [src + "/labelme"]
    pkg.__version__ = version
    q = types.ModuleType("PyQt5")
    q.QtGui, q.QtCore = types.ModuleType("QtGui"), types.ModuleType("QtCore")
    sys.modules.update({"labelme": pkg, "PyQt5": q, "PyQt5.QtGui": q.QtGui, "PyQt5.QtCore": q.QtCore})
    spec = importlib.util.spec_from_file_location("labelme.utils", src + "/labelme/utils/image.py")
    um = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(um)
    sys.modules["labelme.utils"] = pkg.utils = um
    try:
        return importlib.import_module("labelme._label_file")
    finally:
        for k in [k for k in sys.modules if k == "labelme" or k.startswith(("labelme.", "PyQt5"))]:
            sys.modules.pop(k)
        sys.modules.update(saved)


@pytest.mark.parametrize("version", ["5.11.4", "6.3.1"])
@pytest.mark.parametrize("image_data", [False, True])
def test_files_load_in_labelme(vids, tmp_path, version, image_data):
    if version == "6.3.1" and importlib.util.find_spec("tifffile") is None:
        pytest.skip("LabelMe 6.3.1's reader needs tifffile")
    root = str(tmp_path / "run")
    E.run(root, _recipe(stages=["frames", "annotate"], labelme_version=version, image_data=image_data,
                        step=25), source=vids)
    mod = _labelme_reader(version)
    files = [os.path.splitext(p)[0] + ".json" for p in E.Workspace(root).frame_images()][:4]
    for f in files:
        lf = mod.LabelFile(f)
        data = getattr(lf, "image_data", None) if hasattr(lf, "image_data") else lf.imageData
        assert data is not None                                     # embedded, or read from imagePath
        assert len(lf.shapes) == len(json.loads(Path(f).read_text())["shapes"]) > 0
        assert json.loads(Path(f).read_text())["version"] == version
        assert (json.loads(Path(f).read_text())["imageData"] is not None) == image_data


# ------------------------------------------------------------------------------------------------ model reader
def test_model_info_onnx_and_pt_without_torch(tmp_path):
    inf = model_info(MODEL)
    assert inf["names"] == {0: "car", 1: "plate"} and inf["imgsz"] == 320 and not inf["error"]
    from tests.aa_fake_pt import make
    make(str(tmp_path / "best.pt"), {0: "tractor", 1: "truck", 2: "trailer"}, 1024)
    make(str(tmp_path / "old.pt"), ["a", "b"], [1280, 1280], legacy=True)
    a, b = model_info(str(tmp_path / "best.pt")), model_info(str(tmp_path / "old.pt"))
    assert a["names"] == {0: "tractor", 1: "truck", 2: "trailer"} and a["imgsz"] == 1024
    assert b["names"] == {0: "a", 1: "b"} and b["imgsz"] == 1280
    (tmp_path / "bad.pt").write_bytes(b"PK\x03\x04 not really a zip")
    bad = model_info(str(tmp_path / "bad.pt"))
    assert bad["error"] and bad["names"] == {}
    # nothing in a checkpoint runs: a reduce to os.system becomes an inert stand-in
    import pickle

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))
    with zipfile.ZipFile(tmp_path / "evil.pt", "w") as z:
        z.writestr("evil/data.pkl", pickle.dumps({"train_args": {"imgsz": 640}, "x": Evil()}, protocol=2))
    assert model_info(str(tmp_path / "evil.pt"))["imgsz"] == 640
    assert not (tmp_path / "pwned").exists()


# ------------------------------------------------------------------------------------------------ command line
def test_command_line(vids, tmp_path, capsys):
    from AIO_QC.tools.frame_extractor.core.cli import main
    root = str(tmp_path / "run")
    setup = str(tmp_path / "p.aasetup.json")
    rc = main(["autoannotate", root, "--videos", vids, "--model", MODEL, "--every", "6", "--keep-classes", "plate",
               "--rename", "0=vehicle", "--split", "15", "--only", "frames,annotate,rules,package",
               "--save-setup", setup])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Annotate:" in out and "summary:" in out
    assert Recipe.load(setup).classes == {"0": {"keep": True, "name": "vehicle"}}
    labels = {s["label"] for p in E.Workspace(root).frame_images()
              for s in json.loads(Path(os.path.splitext(p)[0] + ".json").read_text())["shapes"]}
    assert labels <= {"vehicle", "plate"} and "vehicle" in labels
    assert main(["autoannotate", root, "--only", "frames", "--every", "9"]) == 2      # continues the run's setup
    assert main(["autoannotate", root, "--status"]) == 0
    assert "Package" in capsys.readouterr().out
    assert main(["autoannotate", root, "--only", "annotate", "--model", str(tmp_path / "missing.onnx")]) == 1


# ------------------------------------------------------------------------------------------------ window
def test_window(qtbot, vids, tmp_path, monkeypatch):
    from AIO_QC.utils.qt_utils import app_settings
    app_settings().remove("autoannotate")
    from AIO_QC.tools.frame_extractor.widgets import autoannotate_window as W
    (tmp_path / "setups").mkdir()
    monkeypatch.setattr(W, "setups_dir", lambda: str(tmp_path / "setups"))
    root = str(tmp_path / "run")
    w = W.AutoAnnotateWindow(None, vids, root)
    qtbot.addWidget(w)
    w.show()
    w.model.setEditText(MODEL)
    qtbot.waitUntil(lambda: w.class_table.rowCount() == 2, timeout=8000)
    assert "320 px" in w.model_card.text() and not w.imgsz.isEnabled()          # fixed ONNX input size
    assert [c.text() for c in w.keep_chips] == ["car", "plate"]
    # rename a class: the keep rule offers the new name
    w.class_table.item(0, 1).setText("vehicle")
    qtbot.waitUntil(lambda: [c.text() for c in w.keep_chips] == ["vehicle", "plate"], timeout=3000)
    # a recipe survives the round trip through the widgets
    r = Recipe.from_dict(w.recipe().to_dict())
    r.step, r.conf, r.labelme_version, r.image_data, r.split_size = 4, 0.4, "5.11.4", True, 33
    r.keep_when, r.keep_classes = "all_of", ["plate"]
    w.set_recipe(r)
    back = w.recipe()
    for k in ("step", "conf", "labelme_version", "image_data", "split_size", "keep_when", "keep_classes", "classes",
              "model"):
        assert getattr(back, k) == getattr(r, k), k
    # frames only
    for s in STAGES:
        w.rows[s].check.setChecked(s == "frames")
    assert w.run_btn.text() == "Run Frames only" and w.run_btn.isEnabled()
    w.start(None)
    qtbot.waitUntil(lambda: w.worker is None, timeout=120000)
    qtbot.waitUntil(lambda: w.rows["frames"].status.text().startswith("Done"), timeout=8000)
    assert E.Workspace(root).frame_images()
    assert not any(p.endswith(".json") and not p.startswith(".") for p in os.listdir(os.path.join(root, "frames", "a")))
    # then only annotate, from its page
    w.start(["annotate"])
    qtbot.waitUntil(lambda: w.worker is None, timeout=120000)
    qtbot.waitUntil(lambda: w.rows["annotate"].status.text().startswith("Done"), timeout=8000)
    assert w.qc_btn.isEnabled() and "annotated" in w.run_info.text()
    # preview draws the boxes and says what the rule would do
    w._preview()
    qtbot.waitUntil(lambda: not w.preview_area.isHidden(), timeout=30000)
    assert "the keep rule would keep" in w.preview_note.text()
    w.close()


def test_entry_points():
    fx = Path(__file__).resolve().parents[1] / "AIO_QC" / "tools" / "frame_extractor"
    src = "".join(p.read_text() for p in [fx / "app.py", *sorted((fx / "main_window").glob("*.py"))])
    assert "a.autoannotate" in src and "Ctrl+Shift+U" in src
    pkg = Path(__file__).resolve().parents[1] / "AIO_QC"
    app = "".join(p.read_text() for p in [pkg / "app.py", *sorted((pkg / "main_window").glob("*.py"))])
    assert "self.a.autoannotate" in app                    # (the Image QC window is app.py + main_window/)
    from AIO_QC.__main__ import CLI_COMMANDS
    assert "autoannotate" in CLI_COMMANDS


def test_package_keeps_same_named_frames_apart(tmp_path):
    import cv2
    import numpy as np
    src = tmp_path / "imgs"
    for sub in ("day", "night"):
        (src / sub).mkdir(parents=True)
        img = np.full((60, 80, 3), 40 if sub == "day" else 200, np.uint8)
        cv2.imwrite(str(src / sub / "frame_1.jpg"), img)
    root = str(tmp_path / "run")
    r = _recipe(source="images", stages=["frames", "annotate", "package"], group_by="none", split_size=0,
                zip_parts=False)
    E.run(root, r, source=str(src))
    pkg = Path(root) / "package"
    names = sorted(p.name for p in pkg.iterdir() if p.suffix in (".jpg", ".json"))
    assert names == ["frame_1.jpg", "frame_1.json", "night__frame_1.jpg", "night__frame_1.json"]
    assert json.loads((pkg / "night__frame_1.json").read_text())["imagePath"] == "night__frame_1.jpg"
    assert json.loads((pkg / "frame_1.json").read_text())["imagePath"] == "frame_1.jpg"


@pytest.mark.parametrize("files", ["link", "copy"])
def test_package_links_or_copies(vids, tmp_path, files):
    root = str(tmp_path / "run")
    E.run(root, _recipe(stages=["frames", "annotate"], step=25), source=vids)
    res = E.run(root, _recipe(group_by="none", split_size=0, zip_parts=False, package_files=files),
                stages=["package"])
    img = E.Workspace(root).frame_images()[0]
    out = Path(root) / "package" / os.path.basename(img)
    same = os.stat(img).st_ino == os.stat(out).st_ino
    assert same == (files == "link")
    assert res["package"]["files_as"] == ("hard links" if files == "link" else "copies")
