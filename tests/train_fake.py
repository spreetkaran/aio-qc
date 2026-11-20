# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A fake Python environment for training tests: `ultralytics` + `torch` stand-ins that run the callbacks the real
trainer runs, quickly, without a GPU or downloads."""
import stat
import sys
import textwrap
from pathlib import Path

ULTRA = '''
__version__ = "8.9.0-fake"
import json, os, time
from pathlib import Path

class _M:
    def __init__(self, v):
        self.map50, self.map, self.mp, self.mr = v, v / 2, v, v
        self.p = self.r = self.ap50 = self.ap = [v, v]
        self.ap_class_index = [0, 1]

class _Metrics:
    def __init__(self, v):
        self.box = _M(v)
        self.nt_per_class = [5, 3]
        self.save_dir = "."

class _Validator:
    def __init__(self):
        self.metrics = _Metrics(0.5)

class _Trainer:
    pass

class YOLO:
    def __init__(self, w):
        self.w = w
        self.cb = {}
        self.names = {0: "a", 1: "b"}
    def add_callback(self, ev, fn):
        self.cb.setdefault(ev, []).append(fn)
    def _run(self, ev, tr):
        for fn in self.cb.get(ev, []):
            fn(tr)
    def train(self, **kw):
        if kw.get("resume"):
            save = Path(self.w).parent.parent
            kw = json.loads((save / "args.json").read_text())
            start = int((save / "done.txt").read_text())
        else:
            save = Path(kw["project"]) / kw["name"]
            save.mkdir(parents=True, exist_ok=True)
            (save / "args.json").write_text(json.dumps(kw))
            start = 0
        if os.environ.get("FAKE_CHILD"):            # a data-loader worker stand-in that would outlive a crash
            import subprocess, sys
            c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            Path(os.environ["FAKE_CHILD"]).write_text(str(c.pid))
        if kw.get("epochs") == 99:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        tr = _Trainer()
        tr.epochs, tr.start_epoch, tr.save_dir, tr.stop = kw["epochs"], start, save, False
        tr.data = {"names": {0: "a", 1: "b"}}
        tr.train_loader = [0] * 4
        tr.test_loader = None
        tr.device = "cpu"
        tr.batch_size = 2
        tr.tloss = None
        tr.label_loss_items = lambda l=None, prefix="train": {f"{prefix}/box_loss": 1.0, f"{prefix}/cls_loss": 2.0}
        w = save / "weights"
        w.mkdir(exist_ok=True)
        tr.last, tr.best = w / "last.pt", w / "best.pt"
        tr.validator = _Validator()
        self.trainer = tr
        self._run("on_pretrain_routine_end", tr)
        for e in range(start, tr.epochs):
            tr.epoch = e
            self._run("on_train_epoch_start", tr)
            for _b in range(4):
                time.sleep(float(os.environ.get("FAKE_BATCH_S", "0.05")))
                self._run("on_train_batch_end", tr)
                if tr.stop:
                    break
            self._run("on_train_epoch_end", tr)
            tr.metrics = {"metrics/mAP50-95(B)": 0.1 * (e + 1), "metrics/mAP50(B)": 0.2 * (e + 1),
                          "metrics/precision(B)": 0.5, "metrics/recall(B)": 0.4, "val/box_loss": 1.5}
            tr.fitness = tr.best_fitness = 0.1 * (e + 1)
            tr.lr = {"lr/pg0": 0.01}
            tr.epoch_time = 0.2
            tr.last.write_text("x")
            tr.best.write_text("x")
            (save / "done.txt").write_text(str(e + 1))
            tr.stop = tr.stop or e + 1 >= tr.epochs
            self._run("on_fit_epoch_end", tr)
            if tr.stop:
                break
        tr.epoch += 1
        self._run("on_fit_epoch_end", tr)          # the final check with best.pt
        tr.epoch -= 1
        self._run("on_train_end", tr)
    def val(self, **kw):
        return _Metrics(0.7)
'''


def fake_python(tmp: Path, requires=()) -> str:
    """A 'python' that is this Python with the fake packages first on the path. requires: Requires-Dist lines of the
    fake ultralytics (to fake a dependency that is not installed)."""
    site = tmp / "fake_site"
    if requires:
        di = site / "ultralytics-8.9.0.dist-info"
        di.mkdir(parents=True, exist_ok=True)
        (di / "METADATA").write_text("Metadata-Version: 2.1\nName: ultralytics\nVersion: 8.9.0\n"
                                     + "".join(f"Requires-Dist: {r}\n" for r in requires))
    (site / "ultralytics").mkdir(parents=True, exist_ok=True)
    (site / "ultralytics" / "__init__.py").write_text(textwrap.dedent(ULTRA))
    (site / "ultralytics" / "data").mkdir(exist_ok=True)       # (what the augmentation recipe hooks into)
    (site / "ultralytics" / "data" / "__init__.py").write_text("")
    (site / "ultralytics" / "data" / "dataset.py").write_text(
        "class _Compose:\n    def __init__(self):\n        self.transforms = ['Mosaic', 'Format']\n"
        "class YOLODataset:\n    augment = True\n    def build_transforms(self, hyp=None):\n"
        "        return _Compose()\n")
    (site / "torch").mkdir(exist_ok=True)
    (site / "torch" / "__init__.py").write_text("__version__ = '9.9-fake'\nclass cuda:\n    @staticmethod\n"
                                                "    def is_available():\n        return False\n")
    py = tmp / "fakepython"
    py.write_text(f"#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} \"$@\"\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return str(py)


def yolo_dataset(root: Path, n=6):
    from PIL import Image
    for sp in ("train", "val"):
        (root / sp / "images").mkdir(parents=True, exist_ok=True)
        (root / sp / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (64, 48), (i * 9, 50, 90)).save(root / sp / "images" / f"{sp}{i}.jpg")
            (root / sp / "labels" / f"{sp}{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.3 0.1 0.1\n")
    (root / "dataset.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: a\n  1: b\n")
    return root / "dataset.yaml"


def wait_ready(td, yaml, qtbot, timeout=60000):
    """Train Model has checked THIS dataset and its environment (the window is reused between tests); says what it
    was waiting for when it does not get there."""
    try:
        qtbot.waitUntil(lambda: td.check is not None and str(td.check.yaml) == str(yaml) and td.env is not None,
                        timeout=timeout)
    except Exception:
        import threading
        raise AssertionError(
            f"Train Model not ready: check={getattr(td.check, 'yaml', None)} want={yaml} env={bool(td.env)} "
            f"ds_info={td.ds_info.text()[:300]!r} env_info={td.env_info.text()[:300]!r} "
            f"workers={[(w.objectName(), w.isRunning(), w.isFinished()) for w in td.workers]} "
            f"python={td._python()!r} threads={[t.name for t in threading.enumerate()]}") from None
