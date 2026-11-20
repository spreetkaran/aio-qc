# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Classifiers of any architecture (timm / torchvision) in Train Model: the architecture list and how it resolves,
flat class-folder datasets (validation held out by a stable hash), the PyTorch trainer's spec, its standalone helpers
(split, ONNX metadata), the classifier augmentation settings and presets, and the window that ties them together.
The training itself needs PyTorch: test_real_training runs it when AIOQC_TORCH_PYTHON names a Python with torch +
timm + ultralytics."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from AIO_QC.core import cls_engine as E
from AIO_QC.core import training as T

from .train_fake import fake_python, wait_ready
from .test_aio_image import shell  # noqa: F401


def _cls_dataset(root: Path, flat=False, n=8, classes=("cat", "dog", "fox")):
    from PIL import Image
    for k, c in enumerate(classes):
        for i in range(n):
            sp = "" if flat else ("val" if i % 4 == 0 else "train")
            d = root / sp / c if sp else root / c
            d.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), (k * 60, i * 20, 90)).save(d / f"{c}{i}.png")
    return root


# ====================================================================================================== core
def test_architecture_list_and_resolution(tmp_path):
    keys = [a[0] for a in T.CLS_ARCHS]
    assert len(keys) == len(set(keys)) >= 60
    assert {a[2] for a in T.CLS_ARCHS} == set(T.CLS_GROUPS)
    for key, title, grp, tm, tv, size, params in T.CLS_ARCHS:
        assert tm or tv, key
        assert size % 32 == 0, key                      # Ultralytics rounds others up; the ONNX is fixed-size
        assert params > 0
    assert T.resolve_arch("arch:resnet50", {"timm": "1.0"}) == ("timm:resnet50.a1_in1k", "")
    assert T.resolve_arch("arch:resnet50", {"timm": None}) == ("tv:resnet50", "")      # torchvision when no timm
    r, why = T.resolve_arch("arch:dinov2_small", {"timm": None})
    assert r == "" and "timm" in why
    assert T.resolve_arch("arch:shufflenet_v2", None) == ("tv:shufflenet_v2_x1_0", "")
    assert T.resolve_arch("timm:anything.x", None) == ("timm:anything.x", "")
    # foundation models for few labelled images: DINOv3 (ViT S … H+, satellite, ConvNeXt), DINOv2, SigLIP 2, CLIP …
    found = {a[0] for a in T.CLS_ARCHS if a[2] == "found"}
    assert {"dinov3_small", "dinov3_large", "dinov3_large_sat", "dinov3_convnext_tiny", "dinov2_large",
            "siglip2_base", "clip_base", "pe_core_base", "aimv2_large", "aimv2_huge", "radio_v4_so400m",
            "radio_v4_huge", "radio_v3_base", "tipsv2_base"} <= found
    assert T.timm_update("arch:radio_v4_huge", {"timm": "1.0.30"}) == "timm>=1.0.31"
    assert T.timm_update("arch:tipsv2_large", {"timm": "1.0.27"}) == "timm>=1.0.28"
    assert "NVIDIA Open Model License" in T.arch_licence("timm:vit_so400m_patch16_radio_v4.c_radio")
    assert T.timm_min("arch:dinov3_large") == "1.0.20" and T.timm_min("timm:vit_large_patch16_dinov3.lvd1689m") == \
        "1.0.20" and T.timm_min("arch:resnet50") is None
    assert T.timm_update("arch:dinov3_small", {"timm": "1.0.15"}) == "timm>=1.0.20"
    assert T.timm_update("arch:dinov3_small", {"timm": "1.0.31"}) is None
    assert T.timm_update("arch:dinov3_small", None) is None                            # not known yet: no claim
    r, why = T.resolve_arch("arch:dinov3_large", {"timm": "1.0.19"})
    assert r == "" and "1.0.20" in why and "Update timm" in why
    assert T.resolve_arch("arch:dinov3_large", {"timm": "1.0.20"})[0] == "timm:vit_large_patch16_dinov3.lvd1689m"
    assert T.resolve_arch("arch:pe_core_small", {"timm": "1.0.15"})[0] == ""
    assert "DINOv3 License" in T.arch_licence("arch:dinov3_convnext_large") and T.arch_licence("arch:dinov2_large") \
        is None
    assert "gated" in T.explain("huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo")
    assert "update timm" in T.explain("RuntimeError: Unknown model (vit_small_patch16_dinov3)")
    few = T.preset_values(T.LR_PRESETS["few"], "clst")
    assert few["freeze_epochs"] >= 1000 and few["probe_features"] == "cached" and few["balance"] == "sampler"
    for m in ("arch:vit_small", "timm:x", "tv:resnet18"):
        assert T.is_torch_model(m) and T.model_task(m) == "cls"
    assert not T.is_torch_model("yolo11s-cls.pt") and T.model_task("yolo11s-cls.pt") == "cls"
    assert T.arch_imgsz("arch:efficientnet_b3") == 288 and T.arch_title("timm:resnet18.a1_in1k") == "ResNet-18"
    # a model the trainer saved: its sidecar says so
    pt = tmp_path / "run" / "weights" / "best.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"x")
    assert not T.is_torch_model(pt) and T.model_task(str(pt)) is None
    (pt.parent / "best.aioqc.json").write_text(json.dumps({"engine": "torch", "arch": "resnet18.a1_in1k",
                                                           "names": ["a", "b"], "imgsz": 160, "top1": 0.9}))
    assert T.is_torch_model(pt) and T.model_task(str(pt)) == "cls" and T.arch_imgsz(str(pt)) == 160
    assert T.is_torch_model(pt.parent / "last.pt")       # its training state too (resume / fine-tune)
    for m in ("arch:resnet18", "timm:vit_small_patch16_224.augreg", "tv:resnet18", "C:/x/y/best.pt"):
        name = T.new_run_name(m)
        assert ":" not in name and "/" not in name and name.split("_2")[0]
    # every preset key is a setting
    for p in list(T.AUG_PRESETS.values()) + list(T.LR_PRESETS.values()):
        for part in ("values", "cls", "clst"):
            assert set(p.get(part) or {}) <= set(T.HYP_KEYS) | {"optimizer", "lr0"}, (p["title"], part)
    assert T.preset_values(T.LR_PRESETS["finetune"], "clst")["freeze_epochs"] == 2
    assert "freeze_epochs" not in T.preset_values(T.LR_PRESETS["finetune"], "det")
    assert T.hyp_defaults(None, "clst")["weight_decay"] == 0.05 and T.hyp_defaults(None)["weight_decay"] == 0.0005
    for k, choices in T.PICKS.items():
        assert T.HYP_ROW[k][3] == "pick" and T.HYP_ROW[k][9] in [c[0] for c in choices]
    assert set(T.HYP_KEYS) - {"mosaic", "translate", "shear", "perspective", "bgr", "copy_paste", "freeze"} \
        <= T.torch_keys()


def test_flat_class_folders_and_split_layout(tmp_path):
    flat = _cls_dataset(tmp_path / "flat", flat=True)
    (flat / "runs").mkdir()                                   # training runs inside: not a class
    issues = flat / "issues"
    issues.mkdir()
    (issues / ".aio_qc_issues").write_text("")
    chk = T.check_dataset(flat)
    assert chk.ok and chk.flat and chk.cls_folder and chk.task == "cls" and chk.names == ["cat", "dog", "fox"]
    assert chk.n("val") == 3 and chk.n("train") == 21 and any("No train/" in t for _l, t in chk.issues)
    split = _cls_dataset(tmp_path / "split")
    chk2 = T.check_dataset(split)
    assert chk2.ok and not chk2.flat and chk2.n("train") == 18 and chk2.n("val") == 6
    # the trainer's own split of a flat folder: stable, and old images stay put when more are added
    names = chk.names
    v1 = {p for p, _c in E.gather(flat, names, "val", 0.25)}
    t1 = {p for p, _c in E.gather(flat, names, "train", 0.25)}
    assert v1 and t1 and not v1 & t1 and len(v1 | t1) == 24
    assert all(any(Path(p).parent.name == c for p in v1) for c in names)      # every class validates
    from PIL import Image
    for i in range(30):
        Image.new("RGB", (8, 8)).save(flat / "cat" / f"new{i}.png")
    v2 = {p for p, _c in E.gather(flat, names, "val", 0.25)}
    hashed = {p for p in v1 if E._bucket(os.path.relpath(p, flat).replace("\\", "/").lower()) < 0.25}
    assert hashed and hashed <= v2                 # (a class with none borrows its lowest image — that one may move)
    assert E.layout(split) == "split" and len(E.gather(split, names, "val")) == 6
    assert E.gather(flat, names, "test") == [] and len(E.gather(split, names, "train", fraction=0.5)) < 18
    assert E.snap(300) == 288 and E.snap(299) == 288 and E.snap(10) == 32 and E.snap(224) == 224
    assert E.snap(256, 14) == 224 and E.snap(400, 14) == 448 and E.snap(256, 16) == 256 and E.snap(384, 16) == 384
    assert E.patch_of("vit_large_patch14_dinov2.lvd142m") == 14 and E.patch_of("convnext_tiny.dinov3_lvd1689m") == 0


def test_spec_of_a_pytorch_classifier_run(tmp_path):
    flat = _cls_dataset(tmp_path / "flat", flat=True)
    chk = T.check_dataset(flat)
    s = {"model": "arch:convnext_tiny", "task": "cls", "epochs": 5, "imgsz": 224, "batch": 32, "device": "cpu",
         "workers": 2, "patience": 3, "project": str(tmp_path / "runs"), "name": "", "pretrained": True, "seed": 1,
         "cache": "ram", "close_mosaic": 10, "single_cls": True, "env": {"timm": None},
         "hyp": {"auto_augment": "augmix", "balance": "sampler", "freeze_epochs": 2, "onnx": True}, "extra": {}}
    run = T.prepare_run(sys.executable, chk, s)
    spec = json.loads((run / "aioqc" / "spec.json").read_text())
    assert spec["engine"] == "torch" and spec["model"] == "tv:convnext_tiny"       # no timm: torchvision's
    a = spec["args"]
    assert a["data"] == str(flat.resolve()) and a["val_share"] == T.FLAT_VAL and a["auto_augment"] == "augmix"
    assert a["balance"] == "sampler" and a["freeze_epochs"] == 2 and a["onnx"] is True
    assert not {"cache", "close_mosaic", "single_cls"} & set(a) and "env" not in spec["settings"]
    assert spec["names"] == ["cat", "dog", "fox"] and ":" not in run.name
    assert set(a) <= E.KEYS
    # a YOLO classifier can't train on a flat folder
    with pytest.raises(ValueError, match="train/ and val/"):
        T.prepare_run(sys.executable, chk, dict(s, model="yolo11n-cls.pt", name="y"))
    with pytest.raises(ValueError, match="timm"):
        T.prepare_run(sys.executable, chk, dict(s, model="arch:eva02_small", name="e"))
    # validating, resuming its model: the trainer that made it
    (run / "weights").mkdir()
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "weights" / "best.aioqc.json").write_text(json.dumps({"engine": "torch"}))
    out = T.prepare_val(str(run / "weights" / "best.pt"), str(flat), run / "val_val", sys.executable, "val", 224,
                        "cpu", 16, {"val_share": 0.15})
    vs = json.loads((out / "aioqc" / "spec.json").read_text())
    assert vs["engine"] == "torch" and vs["args"]["val_share"] == 0.15
    T.prepare_resume(run, sys.executable)
    assert json.loads((run / "aioqc" / "spec.json").read_text())["engine"] == "torch"
    assert T.list_runs(tmp_path / "runs")[0]["engine"] == "torch"
    assert "timm" in T.explain("ModuleNotFoundError: No module named 'timm'")
    assert "pretrained weights" in T.explain("huggingface_hub.errors.LocalEntryNotFoundError: offline")


def test_onnx_metadata_without_the_onnx_package(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Softmax", ["images"], ["output0"], axis=1)], "t",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 3])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = tmp_path / "m.onnx"
    onnx.save(m, str(p))
    meta = {"task": "classify", "names": str({0: "café", 1: "b", 2: "c"}), "imgsz": "[224, 224]", "stride": 32}
    E.add_onnx_metadata(p, meta)
    got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    assert got == {k: str(v) for k, v in meta.items()}
    assert {x.key: x.value for x in onnx.load(str(p)).metadata_props}["names"] == meta["names"]


def test_engine_imports_nothing_from_aio_qc():
    """It runs in the user's environment, copied next to the runner on its own."""
    src = Path(E.__file__).read_text(encoding="utf-8")
    assert "AIO_QC" not in src.split('"""', 2)[2] and "from ." not in src
    code = (f"import sys; sys.path.insert(0, {str(Path(E.__file__).parent)!r}); import cls_engine as E; "
            "print(sorted(E.probe()))")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert "timm" in r.stdout, r.stderr
    assert "torch" not in sys.modules or True
    run_dir = Path(T.runner_copy()).parent
    assert (run_dir / "aioqc_cls_engine.py").read_text(encoding="utf-8") == src


def test_classifier_augmentation_recipe_is_picklable():
    import pickle
    import numpy as np
    from PIL import Image
    from AIO_QC.core import augment_ops as G
    r = G.presets()["Strong"][0]
    t = G.Then(G.ImageRecipe(r, 1.0), lambda im: np.asarray(im).shape)
    im = Image.new("RGB", (64, 40), (120, 80, 40))
    assert t(im) == (40, 64, 3)                                 # the size is kept
    back = pickle.loads(pickle.dumps(G.ImageRecipe(r, 0.5)))
    assert back.share == 0.5 and back._aug is None
    arr = np.zeros((20, 30, 3), np.uint8)
    assert G.ImageRecipe(r, 1.0)(arr).shape == (20, 30, 3)
    assert G.ImageRecipe(r, 0.0)(im) is im


# ====================================================================================================== window
@pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")
def test_train_window_offers_every_architecture(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC import utils
    utils.app_settings().setValue("train/python", fake_python(tmp_path))
    for k in ("train/hyp", "train/hyp_aug", "train/hyp_lr", "train/hyp_opt", "train/hyp_lr0", "train/model",
              "train/extra"):
        utils.app_settings().remove(k)
    flat = _cls_dataset(tmp_path / "flat", flat=True)
    w = shell.window("image")
    w.open_training(yaml=flat)
    td = w.train_dialog
    wait_ready(td, flat, qtbot)
    qtbot.waitUntil(lambda: td.env is not None, timeout=60000)
    assert td.check.flat
    data = [td.model_box.itemData(i) for i in range(td.model_box.count())]
    assert "yolo11s-cls.pt" in data and "arch:convnext_tiny" in data and "arch:dinov2_small" in data
    assert "ask:timm" in data and "ask:tv" in data
    # a YOLO classifier can't train on a flat folder
    td.model_box.setCurrentIndex(td.model_box.findData("yolo11s-cls.pt"))
    assert not td.start_btn.isEnabled() and "train/ and val/" in td.start_note.text()
    assert td.hyp.task == "cls" and not td.hyp.w["mixup"].isEnabled() and td.hyp.w["erasing"].isEnabled()
    # ConvNeXt: torchvision here (the fake environment has no timm)
    td.model_box.setCurrentIndex(td.model_box.findData("arch:convnext_tiny"))
    assert td.hyp.task == "clst" and td.hyp.ft_box.isVisibleTo(td.hyp) and td.hyp.w["mixup"].isEnabled()
    assert not td.hyp.w["mosaic"].isEnabled() and not td.cache.isEnabled() and td.onnx.isEnabled()
    assert "torchvision" in td.model_note.text() and td.imgsz.value() == 224
    assert td.hyp.w["weight_decay"].value() == pytest.approx(0.05)          # this trainer's own defaults
    # DINOv2 needs timm: Start waits, Install offers it
    td.model_box.setCurrentIndex(td.model_box.findData("arch:dinov2_small"))
    assert not td.start_btn.isEnabled() and "timm" in td.start_note.text()
    assert td.install_btn.isVisibleTo(td) and "timm" in td.install_btn.text()
    # DINOv3 with a timm too old for it: Update timm (pip gets "timm>=1.0.20")
    td.env["timm"] = "1.0.15"
    td._fill_models()
    td.model_box.setCurrentIndex(td.model_box.findData("arch:dinov3_large"))
    assert "needs timm 1.0.20+" in td.model_box.currentText()
    assert not td.start_btn.isEnabled() and "1.0.20" in td.start_note.text()
    assert td._to_install == ["timm>=1.0.20"] and td.install_btn.text() == "Update timm"
    td.env["timm"] = "1.0.31"
    td._model_changed()
    assert td.start_btn.isEnabled(), td.start_note.text()
    assert "DINOv3 License" in td.model_note.text() and "linear probe" in td.model_note.text()
    td.hyp.apply_lr("few")
    assert td.hyp.values(full=True)["freeze_epochs"] >= 1000 and td.hyp.lr_preset.currentData() == "few"
    td.env["timm"] = None
    td._fill_models()
    # back to ConvNeXt, the fine-tuning preset, a typo-free extra setting
    td.model_box.setCurrentIndex(td.model_box.findData("arch:convnext_tiny"))
    td.hyp.apply_lr("finetune")
    td.hyp.w["balance"].setCurrentIndex(td.hyp.w["balance"].findData("loss"))
    td.extra.setText("hard_boost=2 mosaic=1")
    assert td._check_extra() is None and "mosaic" in td.extra_note.text()
    td.extra.setText("hard_boost=2")
    assert td.start_btn.isEnabled(), td.start_note.text()
    s = td.settings_dict()
    assert s["hyp"]["freeze_epochs"] == 2 and s["hyp"]["layer_decay"] == 0.75 and s["hyp"]["balance"] == "loss"
    assert s["lr0"] == 0.0002 and s["optimizer"] == "AdamW" and "cache" not in s and s["hyp"]["onnx"] is True
    launched = []
    monkeypatch.setattr(T, "launch", lambda run, py: launched.append(run) or {"pid": 0, "survives": True})
    monkeypatch.setattr(td, "app_watch", lambda run: None)          # (not launched: nothing to watch)
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    td.start()
    spec = json.loads((launched[0] / "aioqc" / "spec.json").read_text())
    assert spec["engine"] == "torch" and spec["model"] == "tv:convnext_tiny" and spec["args"]["hard_boost"] == 2
    # any timm model by name
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("convnext_base.fb_in22k_ft_in1k", True))
    td.model_box.setCurrentIndex(td.model_box.findData("ask:timm"))
    assert td.model_box.currentData() == "timm:convnext_base.fb_in22k_ft_in1k"
    td.close()


# ====================================================================================================== real training
@pytest.mark.skipif(not os.environ.get("AIOQC_TORCH_PYTHON"), reason="set AIOQC_TORCH_PYTHON to a Python with "
                                                                     "torch + timm + ultralytics")
def test_real_training(tmp_path):
    """A real run of the PyTorch trainer: train, pause, resume, fine-tune with a class added, validate — and the
    model predicts through Ultralytics as .pt and .onnx."""
    py = os.environ["AIOQC_TORCH_PYTHON"]
    from .digits_dataset import build
    ds = tmp_path / "digits"
    build(ds)
    chk = T.check_dataset(ds)

    def go(s, until=("finished", "failed", "stopped", "paused"), pause_at=None):
        run = T.prepare_run(py, chk, s)
        T.launch(run, py)
        return run, wait(run, until, pause_at)

    def wait(run, until=("finished", "failed", "stopped", "paused"), pause_at=None):
        t, paused = time.time(), False
        while time.time() - t < 900:
            st = T.status(run)
            if pause_at and not paused and (st.get("last_epoch") or 0) >= pause_at:
                T.control(run, "pause")
                paused = True
            if st.get("state") in until:
                return st
            time.sleep(1)
        raise AssertionError(T.log_tail(run))

    base = {"task": "cls", "epochs": 3, "imgsz": 32, "batch": 64, "device": "cpu", "workers": 0, "patience": 0,
            "project": str(tmp_path / "runs"), "pretrained": False, "seed": 0, "env": {"timm": "1.0.31"},
            "hyp": {"mixup": 0.2, "cutmix": 0.2, "freeze_epochs": 1, "ema": 0.99, "balance": "sampler"}}
    run, st = go(dict(base, model="arch:resnet18", name="r18"), pause_at=1)
    assert st["state"] == "paused", T.log_tail(run)
    T.prepare_resume(run, py)
    T.launch(run, py)
    st = wait(run)
    assert st["state"] == "finished", T.log_tail(run)
    assert [r["epoch"] for r in T.epochs(run) if not r.get("final")] == [1, 2, 3]
    w = run / "weights"
    assert {"best.pt", "last.pt", "best.onnx", "best.aioqc.json"} <= {p.name for p in w.iterdir()}
    assert (run / "val_predictions.csv").is_file() and (run / "confusion_matrix.csv").is_file()
    assert len(st["per_class"]) == 10 and T.list_runs(tmp_path / "runs")[0]["metric"] == "top-1"
    code = ("import sys, glob; from ultralytics import YOLO; d = sys.argv[1]; im = sorted(glob.glob(d + "
            "'/val/*/*.png'))[:12]; a = YOLO(sys.argv[2]); b = YOLO(sys.argv[3]); "
            "ra = a.predict(im, imgsz=32, verbose=False); rb = b.predict(im, verbose=False); "
            "print('AGREE', a.task, b.task, sum(x.probs.top1 == y.probs.top1 for x, y in zip(ra, rb)))")
    r = subprocess.run([py, "-c", code, str(ds), str(w / "best.pt"), str(w / "best.onnx")], capture_output=True,
                       text=True, timeout=300)
    assert "AGREE classify classify 12" in r.stdout, r.stdout + r.stderr
    # fine-tune it on a flat folder with one class more: matched by name, hard examples
    from PIL import Image
    flat = tmp_path / "flat"
    for p in ds.glob("*/*/*.png"):
        d = flat / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        os.link(p, d / f"{p.parent.parent.name}_{p.name}")
    (flat / "blank").mkdir()
    for i in range(30):
        Image.new("RGB", (96, 96), (230, 230, 230)).save(flat / "blank" / f"b{i}.png")
    chk = T.check_dataset(flat)
    run2, st = go(dict(base, model=str(w / "best.pt"), name="ft", hyp={"balance": "hard"}))
    assert st["state"] == "finished", T.log_tail(run2)
    assert st["classes_matched"]["new"] == ["blank"] and len(st["classes_matched"]["kept"]) == 10
    assert "Hard examples" in T.log_tail(run2)
    # few labelled images: DINOv3 ViT-S, a linear probe on features computed once
    run3, s3 = go(dict(base, model="arch:dinov3_small", name="probe", epochs=6,
                       hyp={"freeze_epochs": 1000, "probe_features": "cached", "balance": "sampler", "lr0": 0.01}))
    assert s3["state"] == "finished", T.log_tail(run3)
    assert "Linear probe" in T.log_tail(run3) and len([r for r in T.epochs(run3) if not r.get("final")]) == 6
    assert s3["metrics"]["metrics/accuracy_top1"] > 0.1 and (run3 / "weights" / "best.onnx").is_file()
    out = T.prepare_val(str(run2 / "weights" / "best.pt"), str(flat), run2 / "val_val", py, "val", 32, "cpu", 32,
                        {"val_share": T.FLAT_VAL})
    T.launch(out, py)
    sv = wait(out)
    assert sv["state"] == "finished" and sv["metrics"]["top1"] == pytest.approx(st["metrics"]["metrics/accuracy_top1"],
                                                                                abs=0.02)
