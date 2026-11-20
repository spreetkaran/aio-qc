# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Claude Review against a local stand-in for the Messages API (same endpoints, headers, tool-call answers and
batches): the request carries the numbered image, label list, class descriptions and guidelines; findings land as
rows on the right shape; an unchanged image is never sent twice; Apply (relabel / add the suggested box) with Undo;
Dismiss is remembered; a batch is sent, polled and collected; a refused key and a dataset without permission stop
with clear messages; the key never reaches a file."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from .test_aio_image import _open, shell  # noqa: F401
from .test_similarity_fake import _dataset, _labels


class Fake:
    def __init__(self):
        self.requests, self.batches, self.polls = [], {}, 0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, raw=False):
                b = obj if raw else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.headers.get("x-api-key") != "sk-test":
                    return self._send(401, {"error": {"message": "invalid x-api-key"}})
                parts = self.path.strip("/").split("/")
                bid = parts[3]
                if self.path.endswith("/results"):
                    lines = [json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": fake.answer(p)}})
                             for cid, p in fake.batches[bid]]
                    return self._send(200, "\n".join(lines).encode(), raw=True)
                fake.polls += 1
                ended = fake.polls >= 2
                return self._send(200, {"id": bid, "processing_status": "ended" if ended else "in_progress",
                                        "request_counts": {"processing": 0 if ended else len(fake.batches[bid])},
                                        "results_url": f"http://127.0.0.1:{fake.port}/v1/messages/batches/{bid}/results"})

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.headers.get("x-api-key") != "sk-test":
                    return self._send(401, {"error": {"message": "invalid x-api-key"}})
                assert self.headers.get("anthropic-version") == "2023-06-01"
                if self.path == "/v1/messages/batches":
                    bid = f"msgbatch_{len(fake.batches) + 1}"
                    fake.batches[bid] = [(r["custom_id"], r["params"]) for r in body["requests"]]
                    return self._send(200, {"id": bid, "processing_status": "in_progress"})
                fake.requests.append(body)
                return self._send(200, fake.answer(body))

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def answer(self, p):
        text = "\n".join(c["text"] for c in p["messages"][0]["content"] if c["type"] == "text")
        found = []
        if "#2 red" in text:                                    # im4: the blue square labelled red
            found.append({"type": "wrong_class", "shape": 2, "suggested_class": "blue", "confidence": 0.9,
                          "reason": "the square is blue"})
        if "#1 red" in text:                                    # low confidence: below the list's limit
            found.append({"type": "box_too_loose", "shape": 1, "confidence": 0.3, "reason": "a bit loose"})
        if fake_missing(p):
            found.append({"type": "missing_object", "shape": None, "suggested_class": "red",
                          "box": [100, 90, 140, 110], "confidence": 0.8, "reason": "an unlabelled red mark"})
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": p["model"],
                "content": [{"type": "tool_use", "id": "t1", "name": "report_findings",
                             "input": {"findings": found, "image_ok": not found}}],
                "usage": {"input_tokens": 1000, "output_tokens": 50}}

    def close(self):
        self.srv.shutdown()


def fake_missing(p) -> bool:
    """A red mark was painted on im0 only: look at the sent picture for it."""
    img = next(c for c in p["messages"][0]["content"] if c["type"] == "image")
    a = cv2.imdecode(np.frombuffer(base64.b64decode(img["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    patch = a[95:105, 110:130]
    return bool((patch[:, :, 2] > 180).mean() > 0.5 and (patch[:, :, 0] < 80).mean() > 0.5)


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    for v in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield f
    f.close()


def _ds(tmp):
    root = _dataset(tmp / "ds")
    img = cv2.imread(str(root / "im0.jpg"))
    cv2.rectangle(img, (105, 92), (135, 108), (30, 30, 230), -1)        # an unlabelled red mark on im0
    cv2.imwrite(str(root / "im0.jpg"), img)
    from AIO_QC.core import text_search as TS
    TS.save_descriptions(root, {"template": "", "classes": {"red": "a red square", "blue": "a blue square"}})
    return root


def _run(d, qtbot):
    d.status.setText("")
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None and d.status.text().startswith(("Done", "Stopped")), timeout=60000)


def _tick(d, rows):
    from PyQt6.QtCore import Qt
    for t in range(d.table.rowCount()):
        d.table.item(t, 0).setCheckState(Qt.CheckState.Checked if d._row_at(t) in rows else Qt.CheckState.Unchecked)


def test_claude_review_live_apply_dismiss_batch(shell, qtbot, tmp_path, fake):  # noqa: F811
    root = _ds(tmp_path)
    w = _open(shell, qtbot, root)
    w.session.guides.set_text("red", "Label every red mark, however small.")
    w.session.guides.save()
    w.settings.setValue("claude/base_url", f"http://127.0.0.1:{fake.port}")
    w.settings.setValue("claude/backend", "api")
    w.settings.setValue("claude/scope", "all")
    w.settings.setValue("claude/how", "live")
    w.open_tool("claude_review")
    d = w.open_dialogs["claude_review"]
    _run(d, qtbot)
    assert len(fake.requests) == 10, (d.status.text(), d.errors)
    p = fake.requests[0]
    txt = "\n".join(c["text"] for c in p["messages"][0]["content"] if c["type"] == "text")
    assert p["tool_choice"] == {"type": "tool", "name": "report_findings"} and p["model"] == "claude-sonnet-5-5"
    assert "- red: a red square" in txt and "Label every red mark" in txt and "#1 red — rectangle, box [10, 20, 60, 70]" \
        in txt
    got = sorted((Path(r.key).name, r.cells[0], r.shape_index, r.cells[2]) for r in d.rows)
    assert got == [("im0.json", "Missing object", None, "red"), ("im4.json", "Wrong class", 1, "blue")], got
    assert "10 reviewed now (10,000 / 500 tokens in / out)" in d.summarize()
    # nothing is sent again for unchanged images
    _run(d, qtbot)
    assert len(fake.requests) == 10 and len(d.rows) == 2 and "10 from earlier runs" in d.summarize()
    # Apply: relabel im4's square and add the missing box on im0; then Undo
    _tick(d, list(d.rows))
    d._apply()
    assert _labels(root, "im4.json") == ["red", "blue"]
    added = json.loads((root / "im0.json").read_text())["shapes"][-1]
    assert added["label"] == "red" and abs(added["points"][0][0] - 100) < 1 and abs(added["points"][1][1] - 110) < 1
    assert d.undo_btn.isEnabled() and d.store.verdicts()["wrong_class"] == (1, 0)
    d._undo()
    assert _labels(root, "im4.json") == ["red", "red"] and len(json.loads((root / "im0.json").read_text())["shapes"]) == 2
    # Dismiss is remembered (still from the cache: the labels are back as they were)
    _run(d, qtbot)
    assert len(fake.requests) == 10 and len(d.rows) == 2
    _tick(d, [r for r in d.rows if r.key.endswith("im0.json")])
    d._dismiss()
    _run(d, qtbot)
    assert [Path(r.key).name for r in d.rows] == ["im4.json"] and d.store.verdicts()["missing_object"] == (1, 1)
    # a batch: a changed image is sent as a batch, polled, collected
    from AIO_QC.core import claude_review as CR
    lm = json.loads((root / "im4.json").read_text())
    lm["shapes"][0]["description"] = "rough"
    (root / "im4.json").write_text(json.dumps(lm))
    d.how.setCurrentIndex(d.how.findData("batch"))
    _run(d, qtbot)
    assert d.ctx["batch_ids"] == ["msgbatch_1"] and len(fake.batches["msgbatch_1"]) == 1 and len(fake.requests) == 10
    assert "sent as 1 batch(es)" in d.summarize()
    d._poll()
    qtbot.waitUntil(lambda: not getattr(d, "_polling", False), timeout=10000)
    assert d.store.open_batches()                                      # still running at the first look
    d._poll()
    qtbot.waitUntil(lambda: not getattr(d, "_polling", False), timeout=10000)
    assert not d.store.open_batches() and "Batch results are in: 1" in d.status.text()
    d.how.setCurrentIndex(d.how.findData("live"))
    _run(d, qtbot)
    assert len(fake.requests) == 10 and any(r.key.endswith("im4.json") and r.cells[0] == "Wrong class" for r in d.rows)
    # the key is never written anywhere in the dataset
    for p in (root / ".aio_qc").rglob("*"):
        if p.is_file():
            assert b"sk-test" not in p.read_bytes(), p
    assert CR.ResultStore(root).usage()[0] == 10
    d.close()


def test_claude_review_refusals(shell, qtbot, tmp_path, fake, monkeypatch):  # noqa: F811
    root = _ds(tmp_path)
    w = _open(shell, qtbot, root)
    w.settings.setValue("claude/base_url", f"http://127.0.0.1:{fake.port}")
    w.settings.setValue("claude/backend", "api")
    w.settings.setValue("claude/scope", "all")
    w.settings.setValue("claude/how", "live")
    w.open_tool("claude_review")
    d = w.open_dialogs["claude_review"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
    _run(d, qtbot)
    assert any("refused the key" in e for e in d.errors) and not fake.requests
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        d.make_context()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    d.store.set_allowed(False)
    with pytest.raises(ValueError, match="off for this dataset"):
        d.make_context()
    d.close()


def test_request_and_parse(tmp_path):
    from AIO_QC.core import claude_review as CR
    img = np.full((1000, 2000, 3), 200, np.uint8)
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), img)
    shapes = [{"label": "car", "shape_type": "rectangle", "points": [[100, 100], [500, 400]]},
              {"label": "plate", "shape_type": "polygon", "points": [[600, 600], [800, 600], [800, 700]],
               "description": "MSKU1234565"}]
    pr = CR.prepare("k", "a.jpg", p, shapes, [], ["car", "plate"], {"car": "a whole vehicle"}, "", "m", max_side=1000)
    assert pr.scale == 0.5 and pr.labels == ["car", "plate"]
    txt = pr.params["messages"][0]["content"][-1]["text"]
    assert "#1 car — rectangle, box [50, 50, 250, 200]" in txt and 'text "MSKU1234565"' in txt and "1000 × 500" in txt
    msg = {"content": [{"type": "tool_use", "name": "report_findings", "input": {"findings": [
        {"type": "missing_object", "shape": None, "suggested_class": "car", "box": [10, 20, 30, 40], "confidence": 0.7,
         "reason": "x"},
        {"type": "wrong_text", "shape": 2, "text": "MSKU1234S65", "confidence": 2, "reason": "y"},
        {"type": "bogus", "shape": 9, "confidence": 0.1, "reason": "z"}], "image_ok": False}}]}
    fs = CR.parse(msg, pr)
    assert fs[0].box == [20, 40, 60, 80] and fs[1].shape == 1 and fs[1].label == "plate" and fs[1].confidence == 1
    assert fs[2].type == "other" and fs[2].shape is None
    assert fs[0].fid != fs[1].fid and len(fs[0].fid) == 16
    with pytest.raises(CR.ApiError, match="NOPE_KEY"):
        CR.api_key("NOPE_KEY")
