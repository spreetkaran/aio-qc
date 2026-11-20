# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""AI annotate: the project file, keys, the services (OpenAI-compatible / Gemini / Anthropic, against a local fake),
answers to boxes, checks, the cache and budget, rejected images, targets, the env worker, YOLO / LabelMe output,
readiness, the command line and the panel."""
import json
import os
import shutil
import sys

import pytest

from tests import fx_fixtures
from tests.ai_fake_server import Fake
from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
from AIO_QC.tools.frame_extractor.core.ai_annotate import keys as K
from AIO_QC.tools.frame_extractor.core.ai_annotate import parse as PA
from AIO_QC.tools.frame_extractor.core.ai_annotate import profiles as PF
from AIO_QC.tools.frame_extractor.core.ai_annotate import providers as PV
from AIO_QC.tools.frame_extractor.core.ai_annotate import readiness as RD
from AIO_QC.tools.frame_extractor.core.ai_annotate import writers as W
from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import Annotator
from AIO_QC.tools.frame_extractor.core.autoannotate import engine as E
from AIO_QC.tools.frame_extractor.core.autoannotate.recipe import Recipe

D = fx_fixtures.D


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setenv("AIQC_TEST_KEY", "sk-test")
    yield f
    f.close()


def _cfg(fake, provider="litellm", **kw):
    cfg = C.normalize({
        "project": "t", "connection": {"provider": provider, "base_url": fake.url, "api_key_env": "AIQC_TEST_KEY",
                                       "workers": 4, "retries": 3, "timeout_s": 10},
        "classes": [{"name": "car", "description": "a car"},
                    {"name": "plate", "inside": "car", "text": True, "text_check": "iso6346"}],
        "vlm": {"model": "gemini-2.5-flash"}, "pricing": {"input": 1.0, "output": 2.0}})
    for k, v in kw.items():
        cfg[k] = C._merge(cfg[k], v) if isinstance(v, dict) else v
    return cfg


def _answer(body, plate_text="CSQU3054383"):
    return {"detections": [{"label": "car", "box_2d": [100, 100, 500, 400], "confidence": 0.9},
                           {"label": "plate", "box_2d": [300, 150, 360, 300], "confidence": 0.8, "text": plate_text},
                           {"label": "plate", "box_2d": [900, 900, 950, 990], "confidence": 0.9, "text": "XX"},
                           {"label": "bogus", "box_2d": [0, 0, 10, 10]}],
            "photorealistic": 5, "flags": {"busy": "yes"}}


@pytest.fixture
def images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    import cv2
    import numpy as np
    for i in range(6):
        img = np.full((200, 400, 3), 20 * i, np.uint8)
        cv2.putText(img, f"#{i}", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(str(d / f"cam{i % 2}_img_{i}.jpg"), img)
    return d


# ------------------------------------------------------------------------------------------------ project file
def test_config_roundtrip_defaults_and_problems(tmp_path):
    cfg = C.normalize({"classes": ["a", {"name": "b", "parts": "x, y"}], "flags": ["f1"],
                       "references": {"views": {"default": ["r.jpg"]}}, "custom_key": {"kept": 1}})
    assert cfg["classes"][1]["parts"] == ["x", "y"] and cfg["flags"] == [{"name": "f1", "description": ""}]
    assert cfg["output"]["format"] == "yolo" and cfg["custom_key"] == {"kept": 1}
    p = str(tmp_path / "p.yaml")
    C.save(cfg, p)
    text = open(p, encoding="utf-8").read()
    assert "api_key" not in text.replace("api_key_env", "") and text.startswith("# AIO QC")
    back = C.load(p)
    assert back["classes"][1]["parts"] == ["x", "y"] and back["custom_key"] == {"kept": 1}
    msgs = " ".join(m for _, m in C.check(back, str(tmp_path)))
    assert "Part “x”" in msgs and "Part “y”" in msgs and "Example image not found" in msgs
    assert "Choose the vision model" in msgs
    with pytest.raises(C.ConfigError) as e:
        C.loads("a: [1, 2\nb: 3")
    assert e.value.line
    bad = C.normalize({"classes": [{"name": "a b"}, {"name": "c", "inside": "zzz"}], "checks": {"accept": ["???"]}})
    msgs = " ".join(m for _, m in C.check(bad))
    assert "no spaces" in msgs and "zzz" in msgs and "not understood" in msgs
    for k in C.TEMPLATES:
        assert C.from_template(k, "x")["project"] == "x"


def test_line_of_points_at_the_setting():
    cfg = C.normalize({"classes": [{"name": "a"}, {"name": "b"}], "vlm": {"model": "m"}})
    text = C.dumps(cfg)
    ln = C.line_of(text, "vlm.model")
    assert "model: m" in text.splitlines()[ln - 1]
    ln = C.line_of(text, "classes.1.name")
    assert "name: b" in text.splitlines()[ln - 1]


# ------------------------------------------------------------------------------------------------ keys
def test_keys_saved_as_environment_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("AIQC_K", raising=False)
    home = str(tmp_path)
    (tmp_path / ".bashrc").write_text("alias ll=ls\n")
    where = K.save("AIQC_K", "sk-a'b", home=home)
    assert ".bashrc" in where and os.environ["AIQC_K"] == "sk-a'b"
    del os.environ["AIQC_K"]
    assert K.get("AIQC_K", home=home) == "sk-a'b"                 # a later start finds it
    K.save("AIQC_K", "v2", home=home)
    assert (tmp_path / ".bashrc").read_text().count("AIQC_K") == 3   # one block, replaced
    K.forget("AIQC_K", home=home)
    assert K.get("AIQC_K", home=home) == "" and (tmp_path / ".bashrc").read_text() == "alias ll=ls\n"
    with pytest.raises(ValueError):
        K.save("bad name", "x", home=home)
    assert K.mask("sk-1234567890") == "sk-…890"


# ------------------------------------------------------------------------------------------------ parsing
def test_answers_to_boxes():
    assert PA.extract_json("ok ```json\n{\"a\": [1,],}\n``` bye") == {"a": [1]}
    assert PA.extract_json('x {"s": "}{"} y') == {"s": "}{"}
    with pytest.raises(ValueError):
        PA.extract_json('{"a": [1, 2')
    # the same box in the five conventions -> the same pixels (image sent at 500x250, original 1000x500)
    want = (200.0, 60.0, 600.0, 300.0)
    for fmt, b in (("yxyx_1000", [120, 200, 600, 600]), ("xyxy_1000", [200, 120, 600, 600]),
                   ("xyxy_px", [100, 30, 300, 150]), ("xyxy_norm", [0.2, 0.12, 0.6, 0.6]),
                   ("yxyx_norm", [0.12, 0.2, 0.6, 0.6])):
        assert tuple(round(v, 6) for v in PA.to_pixels(b, fmt, 500, 250, 2, 2)) == want, fmt
    assert PA.iso6346_ok("MSDU 843366 6") and not PA.iso6346_ok("MSDU8433667") and PA.luhn_ok("79927398713")
    assert PA.rule_ok("len(artifacts) <= 1", {"artifacts": ["a"]}, {})[0]
    assert not PA.rule_ok("same_viewpoint == true", {"same_viewpoint": False}, {})[0]
    assert PA.rule_ok("count(car) >= 2", {}, {"car": 2})[0] and not PA.rule_ok("boxes > 2", {}, {"car": 2})[0]
    assert "missing" in PA.rule_ok("score >= 1", {}, {})[1]
    assert PF.profile("gemini-2.5-flash")["box_format"] == "yxyx_1000"
    assert PF.profile("Qwen/Qwen2.5-VL-72B-Instruct")["box_format"] == "xyxy_px"
    assert PF.profile("gpt-4o")["family"] == "GPT" and PF.profile("my-model")["family"] == "Other"


# ------------------------------------------------------------------------------------------------ services
@pytest.mark.parametrize("provider", ["litellm", "gemini", "anthropic"])
def test_services_annotate_one_image(fake, images, provider, tmp_path):
    fake.answer = _answer
    cfg = _cfg(fake, provider)
    a = Annotator(cfg, cache_path=str(tmp_path / "c.db"), audit_path=str(tmp_path / "audit.jsonl"))
    img = str(sorted(images.iterdir())[0])
    r = a.image(img)
    a.close()
    assert not r.error, r.error
    labels = sorted(s["label"] for s in r.shapes)
    assert labels == ["car", "plate"]                     # bogus label dropped, plate outside the car dropped
    car = next(s for s in r.shapes if s["label"] == "car")
    assert [round(v) for v in car["box"]] == [40, 20, 160, 100]   # yxyx_1000 on a 400x200 image
    plate = next(s for s in r.shapes if s["label"] == "plate")
    assert plate["text"] == "CSQU3054383" and plate["text_ok"]
    assert r.usage == {"input": 1000, "output": 100} and abs(r.cost - 0.0012) < 1e-9
    path, body = fake.calls[-1]
    sent = json.dumps(body)
    assert "sk-test" not in sent and "CLASSES" in sent and "base64" in sent or "inline_data" in sent
    assert {"litellm": "/v1/chat/completions", "gemini": ":generateContent", "anthropic": "/v1/messages"}[provider] \
        in path
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "sk-test" not in audit and '"car"' in audit


def test_retries_cache_and_plain_errors(fake, images, tmp_path):
    fake.answer = _answer
    fake.fail_next = [429, 503]
    cfg = _cfg(fake)
    a = Annotator(cfg, cache_path=str(tmp_path / "c.db"))
    img = str(sorted(images.iterdir())[0])
    r = a.image(img)
    assert not r.error and r.calls == 1 and len(fake.calls) == 3       # two retried failures, then the answer
    r2 = a.image(img)
    assert r2.cached == 1 and r2.calls == 0 and r2.cost == 0 and len(fake.calls) == 3   # cached: free
    a.close()
    fake.key = "other"
    cfg["vlm"]["instructions"] = "new rules"                              # a new prompt: not cached
    a = Annotator(cfg, cache_path=str(tmp_path / "c.db"))
    r3 = a.image(img)
    a.close()
    assert "key was not accepted" in r3.error and not r3.accepted
    t = PV.test_connection(cfg)
    assert not t["ok"] and "401" in t["message"]
    fake.key = "sk-test"
    t = PV.test_connection(cfg)
    assert t["ok"] and t["model_listed"] is True and "gpt-4o" in t["models"]


def test_detect_vlm_checks_detector_boxes_with_env_worker(fake, images, tmp_path):
    def ans(body):
        text = json.dumps(body)
        assert "Candidate boxes" in text
        return {"boxes": [{"id": 1, "keep": True, "label": "car", "confidence": 0.8},
                          {"id": 2, "keep": False, "label": "plate"}]}
    fake.answer = ans
    cfg = _cfg(fake, mode="detect_vlm", detector={"type": "_test", "model": "x"})
    a = Annotator(cfg, sys.executable)
    r = a.image(str(sorted(images.iterdir())[0]))
    a.close()
    assert not r.error, r.error
    assert [s["label"] for s in r.shapes] == ["car"] and len(r.candidates) == 2
    cfg2 = _cfg(fake, mode="detect", detector={"type": "_test", "model": "x"})
    a = Annotator(cfg2, sys.executable)
    r = a.image(str(sorted(images.iterdir())[0]))
    a.close()
    assert sorted(s["label"] for s in r.shapes) == ["car", "plate"]
    with pytest.raises(Exception) as e:
        Annotator(cfg2, "")
    assert "Python environment" in str(e.value)


def test_roi_crops_and_box_mapping(fake, images, tmp_path):
    img = str(sorted(images.iterdir())[0])
    W.write(img, 400, 200, [{"label": "car", "box": (100, 50, 300, 150), "conf": 1}],
            {"format": "labelme"}, ["car", "plate"])
    seen = []

    def ans(body):
        seen.append(json.dumps(body))
        return {"detections": [{"label": "plate", "box_2d": [0, 0, 1000, 1000], "confidence": 0.9,
                                "text": "CSQU3054383"}]}
    fake.answer = ans
    cfg = _cfg(fake, roi={"class": "car", "padding": 0.0})
    a = Annotator(cfg)
    r = a.image(img)
    a.close()
    assert "close-up crop of a car" in seen[0]
    plate = [s for s in r.shapes if s["label"] == "plate"][0]
    assert [round(v) for v in plate["box"]] == [100, 50, 300, 150]       # the whole crop, mapped back


# ------------------------------------------------------------------------------------------------ the step
def _run(tmp_path, images, cfg, **kw):
    proj = str(tmp_path / "proj.yaml")
    C.save(cfg, proj)
    root = str(tmp_path / "run")
    r = Recipe(source="images", annotator="ai", ai_config=proj, stages=["frames", "annotate"], **kw)
    return root, r, E.run(root, r, source=str(images))


def test_step_yolo_output_rejects_and_review(fake, images, tmp_path):
    n = {"i": 0}

    def ans(body):
        n["i"] += 1
        a = _answer(body, plate_text="CSQU3054383" if n["i"] % 2 else "WRONG")
        a["photorealistic"] = 5 if n["i"] % 3 else 2
        return a
    fake.answer = ans
    cfg = _cfg(fake, checks={"fields": {"photorealistic": "1-5"}, "accept": ["photorealistic >= 4"]},
               flags=[{"name": "busy", "description": "many cars"}])
    root, r, res = _run(tmp_path, images, cfg)
    a = res["annotate"]
    assert a["annotated"] + a["rejected"] == 6 and a["rejected"] == 2 and a["needs_review"] >= 1
    ws = E.Workspace(root)
    assert len(E.find_images(ws.rejected)) == 2 and (tmp_path / "run" / "reports" / "rejected.csv").is_file()
    kept = ws.frame_images()
    assert len(kept) == 4
    txt = open(os.path.splitext(kept[0])[0] + ".txt").read().split("\n")
    assert txt[0].startswith("0 ") and txt[1].startswith("1 ")                 # class order car, plate
    assert open(os.path.join(ws.frames, "classes.txt")).read().split() == ["car", "plate"]
    assert "nc: 2" in open(os.path.join(ws.frames, "data.yaml")).read()
    assert not os.path.exists(os.path.splitext(kept[0])[0] + ".json")          # YOLO only (the default)
    meta = dict(ws.db.execute("SELECT name, data FROM ai_meta").fetchall())
    assert json.loads(meta[os.path.basename(kept[0])])["flags"] == {"busy": True}
    # again: everything is up to date, the rejected images are asked again (from the cache: free) and rejected again
    calls = len(fake.calls)
    res = E.run(root, r, stages=["annotate"])
    assert res["annotate"]["up_to_date"] == 4 and res["annotate"]["rejected"] == 2 and len(fake.calls) == calls
    # the keep rule reads YOLO files
    r.stages = ["rules"]
    r.keep_when, r.keep_classes = "any_of", ["plate"]
    assert E.run(root, r)["rules"]["kept"] == 4
    # LabelMe + YOLO, merged into a file that is already there
    cfg["output"]["format"] = "both"
    C.save(cfg, r.ai_config)
    res = E.run(root, r, stages=["annotate"])
    d = json.load(open(os.path.splitext(kept[0])[0] + ".json"))
    assert d["version"] == "6.3.1" and d["imageData"] is None and {s["label"] for s in d["shapes"]} == {"car", "plate"}
    assert any("CSQU3054383" in s["description"] or "WRONG" in s["description"] for s in d["shapes"])


def test_budget_stops_safely_and_targets_finish_early(fake, images, tmp_path):
    fake.answer = _answer
    fake.usage = (100000, 0)                          # $0.10 per image at $1 / M input tokens
    cfg = _cfg(fake, limits={"budget_usd": 0.25}, connection={"workers": 1})
    with pytest.raises(E.Stopped) as e:
        _run(tmp_path, images, cfg)
    assert "budget" in str(e.value)
    ws = E.Workspace(str(tmp_path / "run"))
    done = sum(os.path.exists(os.path.splitext(p)[0] + ".txt") for p in ws.frame_images())
    assert 3 <= done < 6
    shutil.rmtree(tmp_path / "run")
    fake.usage = (10, 0)
    cfg = _cfg(fake, limits={"targets": {"car": 2}}, connection={"workers": 1})
    root, r, res = _run(tmp_path, images, cfg)
    assert res["annotate"]["finished_early"] == "every target is reached" and res["annotate"]["annotated"] == 2


def test_command_line_ai(fake, images, tmp_path, capsys):
    fake.answer = _answer
    cfg = _cfg(fake)
    proj = str(tmp_path / "proj.yaml")
    C.save(cfg, proj)
    from AIO_QC.tools.frame_extractor.core.cli import main
    rc = main(["autoannotate", str(tmp_path / "run"), "--images", str(images), "--ai-config", proj, "--only",
               "frames,annotate,package", "--split", "3", "--yolo-layout", "split", "--no-zip"])
    assert rc == 0, capsys.readouterr().err
    part = tmp_path / "run" / "package" / "part_001"
    assert (part / "images").is_dir() and (part / "labels" / "classes.txt").is_file() and (part / "data.yaml").is_file()
    assert "path:" not in (part / "data.yaml").read_text() and (part / "meta.jsonl").is_file()
    assert len(list((part / "labels").glob("*.txt"))) == 4                  # 3 labels + classes.txt


def test_readiness(fake, monkeypatch):
    cfg = _cfg(fake)
    items = {i.key: i for i in RD.check(cfg, "")}
    assert items["app"].status == "ok" and items["env"].status == "skip" and items["key"].status == "ok"
    assert items["service"].status == "ok" and "offered" in items["service"].detail
    monkeypatch.delenv("AIQC_TEST_KEY")
    items = {i.key: i for i in RD.check(cfg, "")}
    assert items["key"].status == "fail" and items["key"].fix == "enter_key"
    cfg2 = _cfg(fake, mode="detect", detector={"type": "grounding_dino", "model": "x"})
    items = {i.key: i for i in RD.check(cfg2, "")}
    assert items["env"].status == "fail" and items["env"].fix == "choose_env"
    items = {i.key: i for i in RD.check(cfg2, sys.executable)}
    assert items["env"].status == "fail" and items["env"].fix == "install_env" and "torch" in items["env"].fix_args


# ------------------------------------------------------------------------------------------------ panel
def test_panel(qtbot, fake, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.widgets import ai_annotate_panel as AP
    monkeypatch.setattr(AP, "projects_dir", lambda: str(tmp_path))
    cfg = _cfg(fake)
    path = str(tmp_path / "p.yaml")
    C.save(cfg, path)
    p = AP.AIAnnotatePanel()
    qtbot.addWidget(p)
    p.show()
    p.load_project(path)
    qtbot.waitUntil(lambda: p.body.isEnabled() and p.ready(), timeout=10000)
    assert p.cls_table.rowCount() == 2 and p.cls_table.item(1, 1).text() == "plate"
    # edits in the app are saved to the project file
    p.cls_table.item(0, 2).setText("any road vehicle")
    p.cls_table.selectRow(1)
    p.cd_parts.setText("owner_code, serial")
    p._cls_parts()
    qtbot.waitUntil(lambda: "owner_code" in open(path).read(), timeout=3000)
    back = C.load(path)
    assert back["classes"][0]["description"] == "any road vehicle"
    assert C.class_names(back) == ["car", "plate", "owner_code", "serial"] and back["classes"][2]["inside"] == "plate"
    # rename a class: what refers to it follows
    p.cls_table.item(0, 1).setText("vehicle")
    p.flush()
    back = C.load(path)
    assert back["classes"][1]["inside"] == "vehicle"
    # the YAML tab applies edits too
    p.yaml.setPlainText(C.dumps(dict(back, project="renamed")))
    p._yaml_apply()
    p.flush()
    assert C.load(path)["project"] == "renamed"
    # a missing key is shown as the first thing to fix
    monkeypatch.delenv("AIQC_TEST_KEY")
    p.check_ready(True)
    qtbot.waitUntil(lambda: bool(p._ready_items) and not p.ready(), timeout=10000)
    assert any(i.fix == "enter_key" for i in p._ready_items)
    assert any("API key" in m for m in p.run_problems())
    p.wait_calls()


def test_window_with_ai_annotate(qtbot, fake, images, tmp_path, monkeypatch):
    from AIO_QC.utils.qt_utils import app_settings
    app_settings().remove("autoannotate")
    from AIO_QC.tools.frame_extractor.widgets import autoannotate_window as WW
    (tmp_path / "setups").mkdir()
    monkeypatch.setattr(WW, "setups_dir", lambda: str(tmp_path / "setups"))
    fake.answer = _answer
    proj = str(tmp_path / "setups" / "p.yaml")
    C.save(_cfg(fake), proj)
    root = str(tmp_path / "run")
    w = WW.AutoAnnotateWindow(None, str(images), root)
    qtbot.addWidget(w)
    w.show()
    w.src_set("images")
    w.ann_set("ai")
    assert not w.ai_panel.isHidden() and w.model_box.isHidden()
    w.ai_panel.load_project(proj)
    qtbot.waitUntil(lambda: w.ai_panel.ready(), timeout=10000)
    r = w.recipe()
    assert r.annotator == "ai" and r.ai_config == proj
    # the keep rule offers the project's classes
    w._fill_keep_list(r)
    assert [c.text() for c in w.keep_chips] == ["car", "plate"]
    r.output_format, r.yolo_layout = "both", "split"
    w.set_recipe(r)
    back = w.recipe()
    assert (back.annotator, back.ai_config, back.output_format, back.yolo_layout) == ("ai", proj, "both", "split")
    for s in ("frames", "similar", "annotate", "rules", "package"):
        w.rows[s].check.setChecked(s in ("frames", "annotate"))
    w._update_steps()
    assert w.run_btn.isEnabled(), w.run_btn.toolTip()
    w.start(None)
    qtbot.waitUntil(lambda: w.worker is None, timeout=60000)
    qtbot.waitUntil(lambda: w.rows["annotate"].status.text().startswith("Done"), timeout=8000)
    txts = [p for p in E.Workspace(root).frame_images() if os.path.exists(os.path.splitext(p)[0] + ".txt")]
    assert len(txts) == 6
    # Try in the panel uses the run folder's frames
    w.ai_panel._try()
    qtbot.waitUntil(lambda: w.ai_panel.try_btn.isEnabled() and bool(getattr(w.ai_panel, "_last_try", None)), timeout=20000)
    assert "kept" in w.ai_panel.try_note.text()
    w.close()
