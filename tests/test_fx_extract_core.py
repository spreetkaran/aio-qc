# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Frame Extractor's batch extraction without the window, on small videos written here: every-Nth and sharpest-of-N
selection, blur filter with negatives, near-duplicate filter, scene cuts, ROI crop, chunked jobs giving the same
frames as one pass, resume after a stop, timeline markers, cross-video duplicates, and model mining / pre-annotation
/ temporal mining with a toy detector (finds the red square; its confidence is the square's brightness)."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from AIO_QC.tools.frame_extractor.core import extract as X
from AIO_QC.tools.frame_extractor.core.detector import Detection


def _video(path: Path, n=60, size=(160, 120), cut=None, blur=(), red=None, static=False) -> Path:
    """Frames of noise-free texture; a white square moves 2 px per frame (or stays, static=True). cut: frame from
    which the colours change completely. blur: frames that are smeared. red: {frame: brightness} of a red square."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    assert vw.isOpened()
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx // 8 + yy // 8) % 2 * 120 + 60).astype(np.uint8)
    for i in range(n):
        f = np.dstack([base, base, base]).copy()
        if cut is not None and i >= cut:
            f = 255 - f
            f[..., 0] = 30
        x = 10 if static else 10 + 2 * i % (w - 40)
        cv2.rectangle(f, (x, 40), (x + 25, 65), (255, 255, 255), -1)
        if red and i in red:
            cv2.rectangle(f, (100, 10), (140, 40), (0, 0, int(red[i])), -1)
        if i in blur:
            f = cv2.GaussianBlur(f, (31, 31), 12)
        vw.write(f)
    vw.release()
    return path


class Toy:
    """A detector stand-in: one 'plate' where the frame is strongly red; conf = red brightness / 255."""
    names = {0: "plate"}

    def __init__(self, name="toy", offset=0):
        self.name, self.offset = name, offset

    def detect(self, img):
        b, g, r = (img[..., k].astype(int) for k in range(3))
        m = (r > 100) & (g < 60) & (b < 60)
        if m.sum() < 50:
            return []
        ys, xs = np.nonzero(m)
        return [Detection(float(xs.min()) + self.offset, float(ys.min()), float(xs.max()) + self.offset,
                          float(ys.max()), float(r[m].mean()) / 255, 0, "plate", self.name)]


def _run(video, out, **kw):
    job_kw = {k: kw.pop(k) for k in ("roi", "chunk", "chunks", "frames", "start", "end") if k in kw}
    detectors = kw.pop("detectors", None)
    s = X.ExtractSettings(**kw)
    seen = []
    st = X.extract(X.Job(str(video), str(out), **job_kw), s, progress=lambda *a: seen.append(a), detectors=detectors)
    assert seen and seen[-1][0] == seen[-1][1] or st.get("cancelled")
    return st


def _saved(out: Path, ext=".jpg"):
    return sorted(p.name for p in out.glob("*" + ext))


def test_every_nth_frame_presets_and_resume(tmp_path):
    v = _video(tmp_path / "v.avi")
    out = tmp_path / "out"
    st = _run(v, out, step=5)
    assert st["saved"] == 12 and st["frames"] == 60, st
    names = _saved(out)
    assert "v_frame_0.jpg" in names and "v_frame_55.jpg" in names, names
    # finished jobs are not run again (resume), unless the settings change
    st2 = _run(v, out, step=5)
    assert st2.get("resumed") == "already complete" and st2["saved"] == 12
    st3 = _run(v, out, step=10, on_exists="skip", resume=False)
    assert st3["saved"] == 0 and st3["skipped_existing"] == 6
    # presets: machine-specific fields are not saved
    s = X.ExtractSettings(step=7, dedup=True, env_python="/x/python", hash_db="/x.db")
    s.save_preset(tmp_path / "p.json")
    d = json.loads((tmp_path / "p.json").read_text())
    assert "env_python" not in d and "hash_db" not in d
    back = X.ExtractSettings.load_preset(tmp_path / "p.json")
    assert back.step == 7 and back.dedup and back.env_python == ""
    assert X.ExtractSettings.from_dict({"step": 3, "unknown": 1}).step == 3


def test_range_roi_and_png(tmp_path):
    v = _video(tmp_path / "v.avi")
    out = tmp_path / "out"
    st = _run(v, out, step=1, start="1", end="2", roi=(10, 20, 90, 100), ext=".png")
    assert 9 <= st["saved"] <= 12, st
    img = cv2.imread(str(next(out.glob("*.png"))))
    assert img.shape[:2] == (80, 80)
    # an ROI past the frame is clamped to it
    st = _run(v, tmp_path / "o2", step=30, roi=(100, 80, 999, 999))
    assert cv2.imread(str(next((tmp_path / "o2").glob("*.jpg")))).shape[:2] == (40, 60)


def test_chunks_give_the_same_frames_as_one_pass(tmp_path):
    v = _video(tmp_path / "v.avi", n=50)
    one = tmp_path / "one"
    _run(v, one, step=4, mode="sharpest")
    parts = tmp_path / "parts"
    for c in range(3):
        _run(v, parts, step=4, mode="sharpest", chunk=c, chunks=3)
    assert _saved(one) == _saved(parts) and len(_saved(one)) == 13


def test_blur_filter_negatives_and_sharpest(tmp_path):
    blur = set(range(0, 60, 2))
    v = _video(tmp_path / "v.avi", blur=blur)
    out = tmp_path / "out"
    from AIO_QC.tools.frame_extractor.core.metrics import sharpness
    cap = cv2.VideoCapture(str(v))
    vals = []
    for _ in range(4):
        vals.append(sharpness(cap.read()[1]))
    cap.release()
    thr = (vals[0] + vals[1]) / 2                     # between a blurred and a sharp frame
    st = _run(v, out, step=1, min_sharpness=thr, keep_blurry_every=5)
    assert st["blurry"] == 30 and st["saved"] == 30 and st["negatives"] == 6, st
    assert len(list((out / X.NEGATIVE_DIR).glob("*.jpg"))) == 6
    # sharpest of every 2: always the sharp one
    out2 = tmp_path / "sharp"
    st = _run(v, out2, step=2, mode="sharpest")
    assert st["saved"] == 30 and all(int(n.split("_")[-1].split(".")[0]) % 2 == 1 for n in _saved(out2)), _saved(out2)


def test_duplicates_and_scene_cuts(tmp_path):
    v = _video(tmp_path / "still.avi", static=True, cut=30)
    st = _run(v, tmp_path / "dd", step=1, dedup=True)
    assert st["saved"] <= 4 and st["duplicates"] >= 56, st                    # one per shot (+ codec noise)
    st = _run(v, tmp_path / "ex", step=1, dedup=True, dedup_mode="exact")
    assert st["saved"] >= st.get("exact_duplicates", 0) or st["saved"] > 0
    st = _run(v, tmp_path / "sc", step=7, scene_detect=True)
    assert st["scenes"] == 2, st
    assert any(n.endswith("_30.jpg") for n in _saved(tmp_path / "sc")), _saved(tmp_path / "sc")   # the grid restarts at the cut


def test_cross_video_duplicates(tmp_path):
    ds = tmp_path / "dataset"
    a = _video(tmp_path / "a.avi")
    b = _video(tmp_path / "b.avi")                                           # the same footage again
    db = str(tmp_path / "hashes.db")
    st = _run(a, ds / "frames" / "a", step=20, cross_dedup=True, hash_db=db)
    assert st["saved"] == 3
    st = _run(b, ds / "frames" / "b", step=20, cross_dedup=True, hash_db=db)
    assert st["saved"] == 0 and st["cross_duplicates"] == 3, st


def test_stop_and_resume(tmp_path):
    v = _video(tmp_path / "v.avi")
    out = tmp_path / "out"
    calls = [0]

    def stop_soon():
        calls[0] += 1
        return calls[0] > 25
    s = X.ExtractSettings(step=1)
    st = X.extract(X.Job(str(v), str(out)), s, cancelled=stop_soon)
    assert st.get("cancelled") and 20 <= st["saved"] <= 25
    first = st["saved"]
    st = X.extract(X.Job(str(v), str(out)), s)
    assert "resumed_from" in st and st["saved"] == 60 and len(_saved(out)) == 60, (first, st)


def test_markers_save_exactly_the_picked_frames(tmp_path):
    v = _video(tmp_path / "v.avi")
    out = tmp_path / "out"
    st = _run(v, out, frames=[[3, None], [10, [0, 0, 50, 50]], 3], min_sharpness=1e9)
    assert st["saved"] == 2 and st["markers"] == 2, st                       # filters never drop a marker
    sizes = sorted(cv2.imread(str(out / n)).shape[:2] for n in _saved(out))
    assert sizes == [(50, 50), (120, 160)]


def test_model_mining_and_pre_annotations(tmp_path):
    red = {i: (80 if i < 20 else 120 if i < 40 else 240) for i in range(10, 60)}     # none · uncertain · confident
    v = _video(tmp_path / "v.avi", red=red)
    out = tmp_path / "mine"
    st = _run(v, out, step=5, models=["toy.onnx"], detectors=[Toy()], mine=["uncertain"], unc_low=0.3,
              unc_high=0.6, preannot="labelme")
    assert st["saved"] == 4 and st["mined_uncertain"] == 4 and st["mined_out"] == 8, st
    log = next(out.glob("mining_*.csv")).read_text().splitlines()
    assert log[0].startswith("frame,") and len(log) == 5 and "uncertain" in log[1]
    j = json.loads((out / _saved(out)[0].replace(".jpg", ".json")).read_text())
    assert j["flags"] == {"uncertain": True} and [x["label"] for x in j["shapes"]] == ["plate"]
    out = tmp_path / "empty"
    st = _run(v, out, step=5, models=["toy.onnx"], detectors=[Toy()], mine=["empty"])
    assert st["saved"] == 4, st                                              # frames 0, 5 (no red), 10, 15 (too dark)
    out = tmp_path / "cls"
    st = _run(v, out, step=5, models=["toy.onnx"], detectors=[Toy()], mine=["classes"], mine_classes=["Plate"],
              preannot="yolo")
    assert st["saved"] == 8 and (out / "classes.txt").read_text() == "plate\n"            # frames 20 … 55
    line = (out / _saved(out)[0].replace(".jpg", ".txt")).read_text().split()
    assert line[0] == "0" and len(line) == 5
    out = tmp_path / "dis"
    st = _run(v, out, step=5, models=["a.onnx", "b.onnx"], detectors=[Toy("a"), Toy("b", offset=30)],
              mine=["disagree"])
    assert st["saved"] == 8 and st["mined_disagree"] == 8, st                # every frame both models see
    # pre-annotations only, nothing mined out
    out = tmp_path / "pre"
    st = _run(v, out, step=10, models=["toy.onnx"], detectors=[Toy()], preannot="labelme")
    assert st["saved"] == 6
    j = json.loads((out / _saved(out)[-1].replace(".jpg", ".json")).read_text())
    assert [s["label"] for s in j["shapes"]] == ["plate"]


def test_temporal_mining_finds_flicker_and_blip(tmp_path):
    red = {i: 240 for i in range(5, 40) if i not in (20, 21)}               # vanishes for 2 frames: flicker
    red.update({50: 240})                                                   # appears once: blip
    v = _video(tmp_path / "v.avi", red=red)
    out = tmp_path / "t"
    st = _run(v, out, step=1, models=["toy.onnx"], detectors=[Toy()], mine=["temporal"], temporal_gap=2)
    assert st.get("mined_flicker", 0) == 2 and st.get("mined_blip", 0) == 1, st
    assert {n.split("_")[-1].split(".")[0][-2:] for n in _saved(out)} == {"20", "21", "50"}, _saved(out)


def test_temporal_monitor_directly():
    mon = X.TemporalMonitor(gap=1)
    d = Detection(0, 0, 10, 10, 0.9, 0, "a")
    out = []
    for i, dets in enumerate([[d], [d], [], [d], [], [], [d], [], []]):
        out += [(idx, why) for idx, _img, _d, why in mon.step(i, None, dets)]
    assert (2, "flicker") in out and (6, "blip") in out


def test_job_process_reports_through_the_queue(tmp_path):
    import queue
    import threading
    v = _video(tmp_path / "v.avi", n=20)
    q = queue.Queue()
    X.job_process(7, X.Job(str(v), str(tmp_path / "o")), X.ExtractSettings(step=5), q, threading.Event())
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    assert msgs and msgs[-1][0] == "done" and msgs[-1][1] == 7, msgs[-3:]
    X.job_process(8, X.Job(str(tmp_path / "missing.avi"), str(tmp_path / "o")), X.ExtractSettings(), q,
                  threading.Event())
    last = None
    while not q.empty():
        last = q.get()
    assert last[0] == "error" and last[1] == 8, last
    assert os.path.isdir(tmp_path / "o")
    with pytest.raises(ValueError, match="ROI tracking"):
        X.extract(X.Job(str(v), str(tmp_path / "o")), X.ExtractSettings(roi_track=True))


class ToyCls:
    """A classifier stand-in: 'red' when the frame's red square is bright, else 'dark' (whole-frame answers)."""
    names = {0: "dark", 1: "red"}
    task = "classify"

    def detect(self, img):
        b, g, r = (img[..., k].astype(int) for k in range(3))
        h, w = img.shape[:2]
        p = 0.95 if ((r > 100) & (g < 60) & (b < 60)).sum() >= 50 else 0.05
        return [Detection(0, 0, w, h, p, 1, "red", "cls"), Detection(0, 0, w, h, 1 - p, 0, "dark", "cls")]


def test_classifier_pre_annotations(tmp_path):
    """A classifier's pre-annotations are classes: LabelMe files with the class as a flag (no shapes), or — the
    YOLO / ultralytics classification layout — each frame in its class folder."""
    red = {i: (240 if i >= 30 else 0) for i in range(60)}
    v = _video(tmp_path / "v.avi", red=red)
    out = tmp_path / "lm"
    st = _run(v, out, step=10, models=["c.onnx"], detectors=[ToyCls()], preannot="labelme")
    assert st["saved"] == 6
    flags = [json.loads((out / n.replace(".jpg", ".json")).read_text()) for n in _saved(out)]
    assert all(not j["shapes"] for j in flags)
    assert [k for j in flags for k, val in j["flags"].items() if val] == ["dark"] * 3 + ["red"] * 3
    out = tmp_path / "folders"
    st = _run(v, out, step=10, models=["c.onnx"], detectors=[ToyCls()], preannot="yolo")
    assert st["saved"] == 6 and not (out / "classes.txt").exists()
    assert len(_saved(out / "dark")) == 3 and len(_saved(out / "red")) == 3 and not _saved(out)
    from AIO_QC.core.dataset import DatasetType, detect_dataset_types
    assert DatasetType.CLS_FOLDERS in detect_dataset_types(out)
    from AIO_QC.core import fx_to_qc as FQ
    assert FQ.inspect(out)["kind"] == "cls_folders"
    assert FQ.inspect(tmp_path / "lm")["kind"] == "cls_labelme"
