# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A checkpoint laid out like an ultralytics best.pt (torch zip format: archive/data.pkl + storages) — made without
torch, so the torch-free reader can be tested. Fake modules exist only while pickling."""
import collections
import io
import pickle
import sys
import types
import zipfile


def make(path, names, imgsz=1024, task="detect", legacy=False):
    made = {}
    for name in ("torch", "torch._utils", "torch.nn", "torch.nn.modules", "torch.nn.modules.container",
                 "ultralytics", "ultralytics.nn", "ultralytics.nn.tasks"):
        if name not in sys.modules:
            made[name] = sys.modules[name] = types.ModuleType(name)
    try:
        U, T = sys.modules["ultralytics.nn.tasks"], sys.modules["torch._utils"]

        class DetectionModel:
            pass
        DetectionModel.__module__ = "ultralytics.nn.tasks"
        DetectionModel.__qualname__ = "DetectionModel"
        U.DetectionModel = DetectionModel

        class FloatStorage:
            pass
        FloatStorage.__module__ = "torch"
        FloatStorage.__qualname__ = "FloatStorage"
        sys.modules["torch"].FloatStorage = FloatStorage

        def _rebuild_tensor_v2(*a):
            return None
        _rebuild_tensor_v2.__module__ = "torch._utils"
        _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
        T._rebuild_tensor_v2 = _rebuild_tensor_v2

        class Tensor:
            def __reduce__(self):
                return (_rebuild_tensor_v2, (FloatStorage(), 0, (3,), (1,), False, collections.OrderedDict()))

        m = DetectionModel()
        m.__dict__.update(training=False, _parameters=collections.OrderedDict(w=Tensor()), names=names,
                          yaml={"nc": len(names)}, args={"imgsz": imgsz, "task": task})
        ck = {"date": "2026-01-01", "version": "8.3.0", "epoch": -1, "model": m, "ema": None,
              "train_args": {"imgsz": imgsz, "task": task, "epochs": 100}}

        class P(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, FloatStorage):
                    return ("storage", FloatStorage, "0", "cpu", 3)
                return None

        buf = io.BytesIO()
        P(buf, protocol=2).dump(ck)
        if legacy:
            with open(path, "wb") as f:
                for x in (0x1950A86A20F9469CFC6C, 1001, {"protocol_version": 1001}):
                    pickle.dump(x, f, protocol=2)
                f.write(buf.getvalue())
        else:
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("best/data.pkl", buf.getvalue())
                z.writestr("best/data/0", b"\0" * 12)
                z.writestr("best/version", "3\n")
    finally:
        for name in made:
            sys.modules.pop(name, None)
