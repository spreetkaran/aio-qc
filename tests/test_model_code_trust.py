# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Models that run their own code (Florence-2, Moondream …) are never allowed by default: refused with a plain reason
until the user allows that model on this computer — never through a project's config file."""
import pytest

from AIO_QC.tools.frame_extractor.core.ai_annotate import runner as RN, trust


class FakeWorker:
    calls = []

    def __init__(self, python):
        pass

    def call(self, op, **kw):
        FakeWorker.calls.append((op, kw))
        return {}

    def close(self):
        pass


def _cfg(mode="detect", det="florence2", model="microsoft/Florence-2-large", vlm="", provider="openai"):
    from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
    cfg = C.normalize({})
    cfg["mode"] = mode
    cfg["detector"].update(type=det, model=model)
    cfg["vlm"]["model"] = vlm
    cfg["connection"]["provider"] = provider
    cfg["classes"] = [{"name": "car", "description": "a car"}]
    return cfg


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(trust, "_file", lambda: tmp_path / "trusted.json")
    monkeypatch.setattr(RN, "LocalWorker", FakeWorker)
    FakeWorker.calls = []


def test_store_and_rules(tmp_path):
    assert trust.needs_own_code("florence2", "x") and trust.needs_own_code("", "vikhyatk/moondream2")
    assert not trust.needs_own_code("grounding_dino", "IDEA-Research/grounding-dino-base")
    assert not trust.trusted("microsoft/Florence-2-large")
    trust.set_trusted("Microsoft/Florence-2-large", True)
    assert trust.trusted("microsoft/florence-2-large")                    # names are not case-sensitive
    d = tmp_path / "local_model"
    d.mkdir()
    trust.set_trusted(str(d), True)
    assert trust.trusted(str(d / ".." / "local_model"))                    # folders by their real path
    trust.set_trusted("microsoft/Florence-2-large", False)
    assert not trust.trusted("microsoft/Florence-2-large") and trust.trusted(str(d))
    assert "has not been allowed" in trust.refusal("m")


def test_runner_refuses_until_allowed(tmp_path):
    from AIO_QC.core.explain import explain
    with pytest.raises(ValueError, match="runs code from its download") as e:
        RN.Annotator(_cfg(), env_python="/x/python", cache_path=str(tmp_path / "c.db"))
    assert "allowed once, by name" in explain(e.value).why
    assert not any(op == "load_detector" for op, _ in FakeWorker.calls)
    trust.set_trusted("microsoft/Florence-2-large", True)
    a = RN.Annotator(_cfg(), env_python="/x/python", cache_path=str(tmp_path / "c.db"))
    a.close()
    op, kw = FakeWorker.calls[-1]
    assert op == "load_detector" and kw["trust_remote_code"] is True
    # a model that does not need it: loaded, its own code not allowed
    a = RN.Annotator(_cfg(det="grounding_dino", model="IDEA-Research/grounding-dino-base"), env_python="/x/python",
                     cache_path=str(tmp_path / "c2.db"))
    a.close()
    assert FakeWorker.calls[-1][1]["trust_remote_code"] is False
    # a local chat VLM that needs it
    with pytest.raises(ValueError, match="moondream"):
        RN.Annotator(_cfg(mode="vlm_detect", vlm="vikhyatk/moondream2", provider="local"), env_python="/x/python",
                     cache_path=str(tmp_path / "c3.db"))


def test_a_shared_config_cannot_allow_it(tmp_path):
    cfg = _cfg()
    cfg["detector"]["trust_remote_code"] = True                           # written into a config by someone
    with pytest.raises(ValueError, match="has not been allowed"):
        RN.Annotator(cfg, env_python="/x/python", cache_path=str(tmp_path / "c.db"))


def test_worker_never_allows_by_default():
    import inspect
    from AIO_QC.tools.frame_extractor.core.ai_annotate import local_worker as LW
    for fn in (LW.load_detector, LW.load_vlm):
        assert inspect.signature(fn).parameters["trust_remote_code"].default is False
    assert "trust_remote_code=True" not in inspect.getsource(LW)


def test_panel_checkbox(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from AIO_QC.tools.frame_extractor.widgets import ai_annotate_panel as AP
    p = AP.AIAnnotatePanel()
    qtbot.addWidget(p)
    p.body.setEnabled(True)                                              # (as with a project open)
    p.det_type.setCurrentIndex(p.det_type.findData("florence2"))
    p.det_model.setText("microsoft/Florence-2-base")
    cb = p.det_trust
    assert not cb.isHidden() and not cb.isChecked()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    cb.click()
    assert not cb.isChecked() and not trust.trusted("microsoft/Florence-2-base")       # said No: still not allowed
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    cb.click()
    assert cb.isChecked() and trust.trusted("microsoft/Florence-2-base")
    cb.click()
    assert not trust.trusted("microsoft/Florence-2-base")
    p.det_type.setCurrentIndex(p.det_type.findData("grounding_dino"))
    p.det_model.setText("IDEA-Research/grounding-dino-base")
    assert cb.isHidden()                                                 # not needed for this one
