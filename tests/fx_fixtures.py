# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Media for the Frame Extractor / Testing tests, generated once and cached (videos via the bundled ffmpeg, two
tiny YOLO-layout ONNX detectors built with `onnx`, and a test set made by the CLI). D = the folder."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

VERSION = 1
D = str(Path(tempfile.gettempdir()) / f"aioqc_test_media_v{VERSION}")


def _videos(d: Path):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    jobs = {
        "a.mp4": ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=8", "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-g", "25"],
        "b.mp4": ["-f", "lavfi", "-i", "mandelbrot=size=480x270:rate=30", "-t", "5", "-c:v", "libx264",
                  "-pix_fmt", "yuv420p"],
        "c.avi": ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=4", "-c:v", "mpeg4"],
    }
    for name, args in jobs.items():
        if not (d / name).is_file():
            subprocess.run([ff, "-v", "error", "-y", *args, str(d / name)], check=True)


def _model(path: Path, seed_w: int):
    """YOLOv8 layout: images (N,3,320,320) -> output0 (N, 4+2, 12); scores depend on the image colours."""
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    P, NC, S = 12, 2, 320
    r0 = np.random.default_rng(5)
    B = np.stack([r0.uniform(40, 280, P), r0.uniform(40, 280, P), r0.uniform(20, 90, P),
                  r0.uniform(20, 90, P)]).astype(np.float32)[None]
    W = np.random.default_rng(seed_w).uniform(-4, 4, (3, NC * P)).astype(np.float32)
    inits = [nh.from_array(B, "B"), nh.from_array(W, "W"), nh.from_array(np.array([0.0], np.float32), "zero"),
             nh.from_array(np.array([0, NC, P], np.int64), "shp"),
             nh.from_array(np.array([-1, 1, 1], np.int64), "shp1"),
             nh.from_array(np.array([2, 3], np.int64), "ax23"), nh.from_array(np.array([1, 2, 3], np.int64), "ax123")]
    nodes = [h.make_node("ReduceMean", ["images", "ax23"], ["m"], keepdims=0), h.make_node("MatMul", ["m", "W"], ["mw"]),
             h.make_node("Sigmoid", ["mw"], ["sg"]), h.make_node("Reshape", ["sg", "shp"], ["scores"]),
             h.make_node("ReduceMean", ["images", "ax123"], ["g"], keepdims=0),
             h.make_node("Mul", ["g", "zero"], ["z"]), h.make_node("Reshape", ["z", "shp1"], ["z3"]),
             h.make_node("Add", ["z3", "B"], ["boxes"]), h.make_node("Concat", ["boxes", "scores"], ["output0"], axis=1)]
    g = h.make_graph(nodes, path.stem, [h.make_tensor_value_info("images", T.FLOAT, ["batch", 3, S, S])],
                     [h.make_tensor_value_info("output0", T.FLOAT, ["batch", 4 + NC, P])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 18)])
    m.ir_version = 9
    for k, v in {"names": "{0: 'car', 1: 'plate'}", "imgsz": "[320, 320]", "task": "detect"}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(path))


def ensure() -> str:
    d = Path(D)
    done = d / ".complete"
    if done.is_file():
        return D
    (d / "models").mkdir(parents=True, exist_ok=True)
    _videos(d)
    for name, seed in (("dummy_a.onnx", 0), ("dummy_b.onnx", 7)):
        if not (d / "models" / name).is_file():
            _model(d / "models" / name, seed)
    if not (d / "ts" / "testset.json").is_file():
        from AIO_QC.tools.frame_extractor.core.cli import main as cli
        rc = cli(["testset", str(d / "ts"), "--add", str(d / "a.mp4"), str(d / "b.mp4"), "--every", "10f",
                  "--unique", "balanced", "--prelabel", str(d / "models" / "dummy_a.onnx")])
        assert rc in (0, None), rc
    done.write_text("ok")
    return D


ensure()
os.environ.setdefault("FX_TEST_MEDIA", D)
