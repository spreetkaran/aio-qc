# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Claude Review — Claude reads each image with its labels drawn and numbered (plus your class descriptions and written
guidelines) and reports what is wrong, in words: missing objects, wrong classes, loose / cut-off boxes, double
labels, text that does not match, guideline breaches (core.claude_review).

    This image                 live, a few seconds — Ctrl+Alt+V opens it on the shown image and starts
    Selected / flagged / list   live, several requests at a time (cost estimate first)
    Batch                      the Message Batches API: half the price, results in the background (this window
                               checks every minute; batches survive closing the app)
Results are kept per image (.aio_qc/claude_review.sqlite): an unchanged image is never sent twice; after a fix only
that image is checked again. Apply turns a finding into an edit (relabel / add the suggested box / remove a label —
backed up, Undo); Dismiss hides it for good. Your accept / dismiss choices give a running precision per finding type.
The API key comes from an environment variable — it is never stored, logged or shown.

Or use your Claude plan: "Claude Code on this computer" runs the unmodified Claude Code you installed and signed in to
yourself (`claude -p`, structured output). AIO QC never sees the sign-in; API-key variables are removed from its
environment so the plan is what gets used. Plans are for ordinary, individual use, so this route is live only and
limited to a few images per run (CR.CC_PLAN_LIMIT) — bulk reviews and batches use an API key.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QVBoxLayout, QWidget,
)

from AIO_QC.core import claude_review as CR
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import newButton, newIcon, style as apply_style
from .base_scan import BaseScanDialog, ResultRow, dspin, ispin
from .inference_service import FuncWorker

API, PLAN = "api", "claude_code"
SCOPES = [("This image", "current"), ("Selected files", "selected"), ("Flagged files", "flagged"),
          ("Files in current list", "list"), ("All files", "all")]


def _store(session):
    st = getattr(session, "_claude_store", None)
    if st is None or str(st.path.parent.parent) != str(session.root):
        st = session._claude_store = CR.ResultStore(session.root)
    return st


class ConnectionDialog(QDialog):
    """Where the key comes from (the variable's NAME), which server, and whether this dataset may be sent."""

    def __init__(self, parent, app, store):
        super().__init__(parent)
        self.app, self.store = app, store
        st = app.settings
        self.setWindowTitle("Claude — connection")
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.env = QLineEdit(st.value("claude/key_env", CR.DEFAULT_KEY_ENV) or CR.DEFAULT_KEY_ENV)
        self.env.setToolTip("The NAME of the environment variable that holds your Anthropic API key — the key itself "
                            "is never stored by AIO QC")
        self.env.textChanged.connect(self._key_state)
        self.key_state = QLabel("")
        self.base = QLineEdit(st.value("claude/base_url", CR.DEFAULT_BASE) or CR.DEFAULT_BASE)
        self.base.setToolTip("https://api.anthropic.com, or your company's gateway that speaks the same API")
        self.allow = QCheckBox("Allow sending this dataset's images to Anthropic (API key or Claude Code)")
        self.allow.setChecked(bool(store.allowed()))
        api_t = QLabel("<b>Anthropic API key</b> — pay per use; batches at half price; any number of images")
        f.addRow(api_t)
        f.addRow("API key from variable:", self.env)
        f.addRow("", self.key_state)
        f.addRow("Server:", self.base)
        how = QLabel("Set the key once (then restart AIO QC):<br>"
                     "<b>Windows</b>: <code>setx ANTHROPIC_API_KEY \"sk-ant-…\"</code><br>"
                     "<b>macOS / Linux</b>: add <code>export ANTHROPIC_API_KEY=\"sk-ant-…\"</code> to ~/.zshrc or "
                     "~/.bashrc<br>Create a key at console.anthropic.com ▸ API keys.")
        how.setWordWrap(True)
        apply_style(how, "color:@muted;")
        f.addRow(how)
        row = QHBoxLayout()
        self.test_lbl = QLabel("")
        self.test_lbl.setWordWrap(True)
        row.addWidget(newButton("Test the key", slot=self._test), 0)
        row.addWidget(self.test_lbl, 1)
        f.addRow(row)
        # Claude Code on this computer (your plan)
        cc_t = QLabel("<b>Claude Code on this computer</b> — uses your Claude plan (Pro / Max / Team / Enterprise); "
                      f"for a few images at a time (at most {CR.CC_PLAN_LIMIT} per run)")
        cc_t.setWordWrap(True)
        f.addRow(cc_t)
        self.cc_path = QLineEdit(st.value("claude/cc_path", "") or "")
        self.cc_path.setPlaceholderText(CR.find_claude() or "not found — install it (below), or choose the program")
        self.cc_path.textChanged.connect(lambda _t: self.cc_state.setText(""))
        prow = QWidget()
        pl = QHBoxLayout(prow)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.cc_path, 1)
        pl.addWidget(newButton("Choose…", slot=self._pick_cc))
        f.addRow("Program:", prow)
        self.cc_state = QLabel("")
        self.cc_state.setWordWrap(True)
        f.addRow("", self.cc_state)
        cch = QLabel("Install it once and sign in with your Claude account (AIO QC never sees the sign-in):<br>"
                     "<b>Windows</b> (PowerShell): <code>irm https://claude.ai/install.ps1 | iex</code> — or "
                     "<code>winget install Anthropic.ClaudeCode</code><br>"
                     "<b>macOS / Linux</b>: <code>curl -fsSL https://claude.ai/install.sh | bash</code><br>"
                     "Then run <code>claude</code> once in a terminal and log in. Anthropic's terms allow your own "
                     "signed-in Claude Code; plans are meant for ordinary, individual use — use an API key for whole "
                     "datasets.")
        cch.setWordWrap(True)
        apply_style(cch, "color:@muted;")
        f.addRow(cch)
        crow = QHBoxLayout()
        self.cc_lbl = QLabel("")
        self.cc_lbl.setWordWrap(True)
        crow.addWidget(newButton("Test Claude Code", icon="terminal-spark", slot=self._test_cc), 0)
        crow.addWidget(self.cc_lbl, 1)
        f.addRow(crow)
        f.addRow("", self.allow)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._key_state()

    def _key_state(self):
        name = self.env.text().strip() or CR.DEFAULT_KEY_ENV
        ok = bool(os.environ.get(name, "").strip())
        self.key_state.setText(f"✓ {name} is set" if ok else f"✗ {name} is not set in this session")
        apply_style(self.key_state, "color:@ok;" if ok else "color:@error;")

    def _test(self):
        self.test_lbl.setText("Testing …")
        client = CR.Client(self.env.text().strip(), self.base.text().strip(), timeout=30, retries=0)
        model = self.app.settings.value("claude/model", CR.DEFAULT_MODEL) or CR.DEFAULT_MODEL
        w = FuncWorker(lambda: client.test(model), self)
        w.done.connect(lambda r: self.test_lbl.setText(f"✓ {model} answered: {str(r)[:40]}"))
        w.failed.connect(lambda e: self.test_lbl.setText(f"✗ {e}"))
        w.finished.connect(w.deleteLater)
        w.start()
        self._w = w

    def _pick_cc(self):
        from PyQt6.QtWidgets import QFileDialog
        p, _f = QFileDialog.getOpenFileName(self, "Claude Code program", str(CR.Path.home()))
        if p:
            self.cc_path.setText(p)

    def _test_cc(self):
        self.cc_lbl.setText("Testing … (a few seconds)")
        cc = CR.ClaudeCode(self.cc_path.text().strip())
        if not cc.path:
            self.cc_lbl.setText("✗ Claude Code not found — install it (above) or choose the program")
            return
        model = self.app.settings.value("claude/model", CR.DEFAULT_MODEL) or CR.DEFAULT_MODEL

        def work():
            return cc.version(), cc.test(model)
        w = FuncWorker(work, self)
        w.done.connect(lambda r: self.cc_lbl.setText(f"✓ {r[0]} — {model} answered: {r[1]}"))
        w.failed.connect(lambda e: self.cc_lbl.setText(f"✗ {e}"))
        w.finished.connect(w.deleteLater)
        w.start()
        self._w2 = w

    def _save(self):
        st = self.app.settings
        st.setValue("claude/cc_path", self.cc_path.text().strip())
        st.setValue("claude/key_env", self.env.text().strip() or CR.DEFAULT_KEY_ENV)
        st.setValue("claude/base_url", self.base.text().strip() or CR.DEFAULT_BASE)
        self.store.set_allowed(self.allow.isChecked())
        self.accept()


class ClaudeReviewDialog(BaseScanDialog):
    TITLE, ICON = "Claude Review", ""
    HELP = ("Claude looks at each image with its labels drawn and numbered — together with your class descriptions "
            "and written guidelines — and says what is wrong, in words. Images are sent to Anthropic's API (only for "
            "datasets you allow; the key comes from an environment variable). Unchanged images are never sent "
            "twice. Findings are suggestions: Apply makes the edit (backed up, Undo), Dismiss hides it.")
    COLUMNS = ["Finding", "Label", "Suggests", "Conf.", "Why"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = False

    # ------------------------------------------------------------------ parameters
    def build_params(self, form):
        st = self.app.settings
        self.store = _store(self.session)
        self.model = QComboBox()
        self.model.setEditable(True)
        for mid, title in CR.MODELS:
            self.model.addItem(title, mid)
        cur = st.value("claude/model", CR.DEFAULT_MODEL) or CR.DEFAULT_MODEL
        i = self.model.findData(cur)
        if i >= 0:
            self.model.setCurrentIndex(i)
        else:
            self.model.setEditText(cur)
        mrow = QWidget()
        ml = QHBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.model, 1)
        ml.addWidget(newButton("Connection…", slot=self._connection, tip="API key variable, server, permission"))
        self.conn_note = QLabel("")
        self.conn_note.setWordWrap(True)
        apply_style(self.conn_note, "color:@muted;")
        self.backend = QComboBox()
        self.backend.addItem(newIcon("ai-box"), "Anthropic API key", API)
        self.backend.addItem(newIcon("terminal-spark"), "Claude Code on this computer — your Claude plan", PLAN)
        self.backend.setToolTip("API key: pay per use, any number of images, batches at half price.\nClaude Code: "
                                f"your own signed-in Claude Code and plan; live, at most {CR.CC_PLAN_LIMIT} images "
                                "per run.")
        self.backend.setCurrentIndex(max(0, self.backend.findData(st.value("claude/backend", API))))
        self.scope = QComboBox()
        for t, k in SCOPES:
            self.scope.addItem(t, k)
        self.scope.setCurrentIndex(max(0, self.scope.findData(st.value("claude/scope", "current"))))
        self.how = QComboBox()
        self.how.addItem("Now (live)", "live")
        self.how.addItem("Batch — half price, results in the background", "batch")
        self.how.setCurrentIndex(max(0, self.how.findData(st.value("claude/how", "live"))))
        self.guides = QCheckBox("guidelines")
        self.guides.setChecked(True)
        self.guides.setToolTip("Send the written labelling guidelines (Guidelines…) — text only")
        self.descs = QCheckBox("class descriptions")
        self.descs.setChecked(True)
        self.descs.setToolTip("Send the one-line class descriptions (Search in Words ▸ Class descriptions…)")
        self.clean = QCheckBox("clean copy too")
        self.clean.setToolTip("Also send the image without drawings (helps reading small text; about twice the cost)")
        self.parallel = ispin(1, 16, int(st.value("claude/parallel", 4)))
        self.parallel.setToolTip("Live requests at the same time")
        self.max_side = ispin(256, 4096, int(st.value("claude/max_side", CR.MAX_SIDE)))
        self.max_side.setSuffix(" px")
        self.max_side.setToolTip("Longer side of the image sent")
        self.min_conf = dspin(0.0, 1.0, float(st.value("claude/min_conf", 0.5)), 2, step=0.05)
        self.min_conf.setToolTip("List findings with at least this confidence")
        from AIO_QC.utils.qt_utils import flow_row
        form.addRow("Use:", self.backend)
        form.addRow("Model:", mrow)
        form.addRow("", self.conn_note)
        form.addRow("Images:", flow_row([self.scope, ("how", self.how)]))
        form.addRow("Send:", flow_row([self.guides, self.descs, self.clean, ("longer side", self.max_side),
                                       ("at a time", self.parallel), ("show conf. ≥", self.min_conf)]))
        self.add_action("✓ Apply", self._apply, "Make the suggested edit for the ticked (or selected) findings: "
                                                "relabel, add the suggested box, or remove a label on nothing — "
                                                "backed up, undoable")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Put back the files changed by the last Apply")
        self.undo_btn.setEnabled(False)
        self.add_action("✕ Dismiss", self._dismiss, "Hide the ticked (or selected) findings for good (they count as "
                                                   "'not a problem' in the precision per type)")
        self.add_action("⟳ Check batches", self._poll, "Ask the API whether batches sent earlier are done")
        self._last_backup = None
        self._limit_keys = None
        self.backend.currentIndexChanged.connect(lambda _i: self._backend_ui())
        self._backend_ui()
        self._timer = QTimer(self, interval=60_000, timeout=self._poll)
        self._timer.start()
        QTimer.singleShot(0, self._poll)

    def _backend_ui(self):
        plan = self.backend.currentData() == PLAN
        bi = self.how.findData("batch")
        item = self.how.model().item(bi)
        if item is not None:
            item.setEnabled(not plan)
        if plan and self.how.currentData() == "batch":
            self.how.setCurrentIndex(self.how.findData("live"))
        self.parallel.setMaximum(3 if plan else 16)
        self._conn_text()

    def _conn_text(self):
        st = self.app.settings
        allowed = self.store.allowed()
        n, tin, tout = self.store.usage()
        if self.backend.currentData() == PLAN:
            path = CR.find_claude(st.value("claude/cc_path", "") or "")
            head = (f"Claude Code: {path} ✓ (your plan; at most {CR.CC_PLAN_LIMIT} images per run, no batches)"
                    if path else "⚠ Claude Code not found (Connection… shows how to install it)")
        else:
            env = st.value("claude/key_env", CR.DEFAULT_KEY_ENV) or CR.DEFAULT_KEY_ENV
            ok = bool(os.environ.get(env, "").strip())
            head = f"Key: {env} ✓" if ok else f"⚠ {env} is not set (Connection…)"
        self.conn_note.setText(head +
                               {None: " · sending this dataset: asked on the first run", True: " · sending allowed",
                                False: " · ⚠ sending is OFF for this dataset"}[allowed] +
                               (f" · reviewed so far {n:,} image(s), {tin:,} / {tout:,} tokens in / out" if n else ""))

    def _connection(self):
        if ConnectionDialog(self, self.app, self.store).exec():
            self._conn_text()

    # ------------------------------------------------------------------ scope
    def samples_to_scan(self):
        out = self._all_samples()
        if self._limit_keys is not None:
            out = [x for x in out if x.key in self._limit_keys]
        return out

    def _all_samples(self):
        s, scope = self.session, self.scope.currentData()
        keys = []
        if scope == "current":
            keys = [self.app.current_key] if getattr(self.app, "current_key", None) else []
        elif scope == "selected":
            keys = self.app.file_dock.selected_keys()
        elif scope == "flagged":
            keys = s.keys_with_review([REVIEW_FLAG])
        elif scope == "list":
            keys = self.app.file_dock.visible_keys()
        else:
            return list(s.samples)
        return [s.by_key[k] for k in keys if k in s.by_key]

    # ------------------------------------------------------------------ scan
    def make_context(self):
        st = self.app.settings
        model = (self.model.currentData() if self.model.currentText() == self.model.itemText(self.model.currentIndex())
                 else None) or self.model.currentText().strip()
        if not model:
            raise ValueError("Choose a model.")
        env = st.value("claude/key_env", CR.DEFAULT_KEY_ENV) or CR.DEFAULT_KEY_ENV
        backend = self.backend.currentData()
        st.setValue("claude/backend", backend)
        cc_path = None
        if backend == PLAN:
            cc_path = CR.find_claude(st.value("claude/cc_path", "") or "")
            if not cc_path:
                raise ValueError("Claude Code was not found on this computer. Connection… shows how to install it "
                                 "and sign in.")
        else:
            try:
                CR.api_key(env)
            except CR.ApiError as exc:
                raise ValueError(f"{exc}. Connection… shows how.") from None
        self._limit_keys = None
        samples = self.samples_to_scan()
        if not samples:
            raise ValueError({"current": "No image is shown.", "selected": "No files are selected in the file list.",
                              "flagged": "No file is flagged."}.get(self.scope.currentData(), "No files."))
        allowed = self.store.allowed()
        if allowed is None:
            allowed = QMessageBox.question(
                self, self.TITLE, "Send this dataset's images to Anthropic's API for review?\n\nEach image is sent "
                "with its labels drawn on it, the label list and (if ticked) your class descriptions and guidelines. "
                "Your choice is kept for this dataset (Connection… changes it).") == QMessageBox.StandardButton.Yes
            self.store.set_allowed(allowed)
            self._conn_text()
        if not allowed:
            raise ValueError("Sending images is off for this dataset (Connection… ▸ Allow).")
        for k, w in (("model", model), ("scope", self.scope.currentData()), ("how", self.how.currentData()),
                     ("parallel", self.parallel.value()), ("max_side", self.max_side.value()),
                     ("min_conf", self.min_conf.value())):
            st.setValue(f"claude/{k}", w)
        s = self.session
        classes = list(s.cls_classes()) if s.is_classification else \
            list(dict.fromkeys(list(s.true_classes or []) + [lab for lab, _n in s.label_totals().most_common()]))
        from AIO_QC.core import text_search as TS
        data = TS.load_descriptions(s.root) if self.descs.isChecked() else {"classes": {}}
        descs = {c: d for c, d in zip(classes, TS.descriptions_for(classes, data))} if self.descs.isChecked() else {}
        guides = CR.guidelines_text(getattr(s, "guides", None), classes) if self.guides.isChecked() else ""
        extra = hashlib.sha1(json.dumps([descs, guides, self.clean.isChecked(), self.max_side.value()],
                                        sort_keys=True).encode()).hexdigest()[:12]
        if backend == PLAN and len(samples) > CR.CC_PLAN_LIMIT:
            cap = CR.CC_PLAN_LIMIT
            if QMessageBox.question(
                    self, self.TITLE, f"Your Claude plan is meant for a few images at a time: review the first {cap} "
                    f"of these {len(samples):,} images?\n\nFor whole datasets use an Anthropic API key (Use: API "
                    "key) — it also offers batches at half price.") != QMessageBox.StandardButton.Yes:
                raise ValueError("Not started.")
            samples = samples[:cap]
            self._limit_keys = {x.key for x in samples}
        if backend == PLAN and self.how.currentData() == "batch":
            self.how.setCurrentIndex(self.how.findData("live"))
        if len(samples) > 1 and backend == API:
            from AIO_QC.core.io import image_size
            wh = next((image_size(x.image_path) for x in samples[:5] if x.image_path is not None), None) or (1280, 720)
            k = min(1.0, self.max_side.value() / max(wh))
            per = int(wh[0] * k * wh[1] * k / 750) * (2 if self.clean.isChecked() else 1) + 900
            batch = self.how.currentData() == "batch"
            if QMessageBox.question(
                    self, self.TITLE, f"Review up to {len(samples):,} image(s) with {model}"
                    f"{' as a batch (half price, results in the background)' if batch else ''}?\n\n"
                    f"About {per:,} input tokens per image, ≈ {per * len(samples):,} in all. Images reviewed before and "
                    "unchanged are not sent again.") != QMessageBox.StandardButton.Yes:
                raise ValueError("Not started.")
        return {"model": model, "env": env, "base": st.value("claude/base_url", CR.DEFAULT_BASE) or CR.DEFAULT_BASE,
                "how": self.how.currentData(), "parallel": self.parallel.value(), "max_side": self.max_side.value(),
                "clean": self.clean.isChecked(), "min_conf": self.min_conf.value(), "classes": classes,
                "descs": descs, "guides": guides, "extra": extra, "todo": [], "cached": 0, "sent": 0, "tin": 0,
                "tout": 0, "api_errors": [], "cancel": threading.Event(), "files": {}, "batch_ids": [],
                "cls": s.is_classification, "root": str(s.root), "backend": backend, "cc_path": cc_path,
                "plan_stop": ""}

    def stop_scan(self):
        if self.ctx and self.ctx.get("cancel") is not None:
            self.ctx["cancel"].set()
        super().stop_scan()

    def _shapes(self, s):
        """(image path, shapes in pixels, image tags, (W, H))"""
        ad = self.session.adapter
        ann = ad.load(s, light=True)
        img = ann.image_path or s.image_path
        tags = list(self.session.tags_of(s.key)) if self.session.is_classification else list(ann.tags or [])
        if img is None:
            raise ValueError("image not found")
        shapes = [dict(x) for x in ann.shapes if not x.get("_error")] if not self.session.is_classification else []
        W, H = float(ann.width or 0), float(ann.height or 0)
        if ann.normalized or not (W and H):
            size = ad.image_size(s, ann)
            if not size:
                raise ValueError("image size unknown")
            W, H = float(size[0]), float(size[1])
        if ann.normalized:
            shapes = [dict(x, points=[[float(p[0]) * W, float(p[1]) * H] for p in x.get("points") or []])
                      for x in shapes]
        return str(img), shapes, tags, (W, H)

    def scan_sample(self, ctx, s):
        img, shapes, tags, wh = self._shapes(s)
        rel = os.path.relpath(s.key, ctx["root"]).replace("\\", "/")
        sig = CR.signature(img, shapes, tags, ctx["model"], ctx["extra"])
        ctx["files"][s.key] = {"rel": rel, "wh": wh, "shapes": shapes}
        got = self.store.get(rel, sig)
        if got is not None:
            ctx["cached"] += 1
            return self._rows(ctx, s.key, got)
        ctx["todo"].append((s.key, rel, img, shapes, tags, sig))
        return []

    def _prep(self, ctx, item) -> CR.Prepared:
        key, rel, img, shapes, tags, sig = item
        p = CR.prepare(key, rel, img, shapes, tags, ctx["classes"], ctx["descs"], ctx["guides"], ctx["model"],
                       ctx["max_side"], ctx["clean"], ctx["cls"])
        p.sig = sig
        return p

    def finalize(self, ctx):
        todo = ctx["todo"]
        if not todo:
            return []
        plan = ctx["backend"] == PLAN
        client = CR.ClaudeCode(ctx["cc_path"]) if plan else CR.Client(ctx["env"], ctx["base"])
        out = []
        if ctx["how"] == "batch" and not plan:
            self._send_batches(ctx, client, todo)
            return out
        done = 0
        lock = threading.Lock()

        def one(item):
            if ctx["cancel"].is_set():
                return item, None, None
            p = self._prep(ctx, item)
            msg = client.review(p, ctx["cancel"]) if plan else client.message(p.params)
            return item, p, msg

        with ThreadPoolExecutor(max_workers=max(1, min(ctx["parallel"], 3) if plan else ctx["parallel"])) as ex:
            futs = [ex.submit(one, it) for it in todo]
            for fu in as_completed(futs):
                try:
                    item, p, msg = fu.result()
                except CR.PlanError as exc:                          # not signed in / plan limit: stop
                    ctx["cancel"].set()
                    ctx["plan_stop"] = str(exc)
                    ctx["api_errors"].append(f"Claude Code: {exc}")
                    for f in futs:
                        f.cancel()
                    break
                except CR.ApiError as exc:
                    if exc.status in (401, 403):                     # the key: nothing else will work
                        ctx["cancel"].set()
                        ctx["api_errors"].append(f"the API refused the key ({exc}) — check {ctx['env']}")
                        for f in futs:
                            f.cancel()
                        break
                    ctx["api_errors"].append(str(exc))
                    continue
                except Exception as exc:                          # one image failed: say which, go on
                    ctx["api_errors"].append(f"{type(exc).__name__}: {exc}")
                    continue
                if p is None:
                    continue
                try:
                    fs = CR.parse(msg, p)
                except ValueError as exc:
                    ctx["api_errors"].append(f"{item[1]}: {exc}")
                    continue
                tin, tout = CR.usage_of(msg)
                self.store.put(p.rel, p.sig, ctx["model"], fs, tin, tout)
                with lock:
                    ctx["sent"] += 1
                    ctx["tin"] += tin
                    ctx["tout"] += tout
                    out.extend(self._rows(ctx, p.key, fs))
                    done += 1
                w = self.worker
                if w is not None:
                    w.progress.emit(done, len(todo))
        return out

    def _send_batches(self, ctx, client, todo):
        cur, items, size, n = [], {}, 0, 0
        for i, it in enumerate(todo):
            if ctx["cancel"].is_set():
                break
            try:
                p = self._prep(ctx, it)
            except Exception as exc:
                ctx["api_errors"].append(f"{it[1]}: {exc}")
                continue
            cid = f"r{i}"
            req = {"custom_id": cid, "params": p.params}
            b = len(json.dumps(p.params))
            if cur and (size + b > (180 << 20) or len(cur) >= 50_000):
                self._create(ctx, client, cur, items)
                cur, items, size = [], {}, 0
            cur.append(req)
            items[cid] = {"key": p.key, "rel": p.rel, "sig": p.sig, "scale": p.scale, "labels": p.labels}
            size += b
            n += 1
        if cur:
            self._create(ctx, client, cur, items)
        ctx["sent"] = n

    def _create(self, ctx, client, reqs, items):
        b = client.batch_create(reqs)
        self.store.add_batch(b["id"], ctx["model"], items)
        ctx["batch_ids"].append(b["id"])

    def _rows(self, ctx, key, findings) -> list[ResultRow]:
        info = ctx["files"].get(key) or {}
        rel = info.get("rel", "")
        gone = self.store.dismissed(rel)
        rows = []
        for f in findings:
            if f.fid in gone or f.confidence < ctx["min_conf"]:
                continue
            lab = f.label or (", ".join(self.session.tags_of(key)) if ctx["cls"] else "")
            rows.append(ResultRow(key, (CR.TYPES.get(f.type, f.type), lab, f.suggested_class or f.text,
                                        round(f.confidence, 2), f.reason), f.shape, payload=f,
                                  tone=CR.TONES.get(f.type, "muted")))
        return rows

    def on_scan_complete(self, ctx):
        if ctx.get("api_errors"):
            self.errors = list(self.errors) + ctx["api_errors"]
            self.app.log_errors([f"[{self.TITLE}] {e}" for e in ctx["api_errors"]])
        self._conn_text()

    def summarize(self):
        c = self.ctx or {}
        if "cached" not in c:
            return super().summarize()
        parts = [f"{c['cached'] + c['sent']:,} image(s): {c['cached']:,} from earlier runs (not sent again)"]
        if c.get("batch_ids"):
            parts.append(f"{c['sent']:,} sent as {len(c['batch_ids'])} batch(es) — results come in the background "
                         "(usually within the hour); this window checks every minute, or reopen it later")
        else:
            parts.append(f"{c['sent']:,} reviewed now" + (" with Claude Code (your plan)" if c.get("backend") == PLAN
                                                          else "") + f" ({c['tin']:,} / {c['tout']:,} tokens in / out)")
        if c.get("plan_stop"):
            parts.append(f"stopped: {c['plan_stop']}")
        parts.append(f"{len(self.rows):,} finding(s)")
        if c.get("api_errors"):
            parts.append(f"{len(c['api_errors'])} not reviewed (see the errors)")
        v = self.store.verdicts()
        if v:
            parts.append("your verdicts: " + ", ".join(f"{CR.TYPES.get(t, t).lower()} {a}✓ {d}✕"
                                                       for t, (a, d) in sorted(v.items())))
        return " · ".join(parts)

    # ------------------------------------------------------------------ batches
    def _poll(self):
        if getattr(self, "_polling", False):
            return
        open_b = self.store.open_batches()
        if not open_b:
            return
        st = self.app.settings
        client = CR.Client(st.value("claude/key_env", CR.DEFAULT_KEY_ENV), st.value("claude/base_url",
                                                                                     CR.DEFAULT_BASE))
        self._polling = True

        def work():
            res = []
            for bid, model, items in open_b:
                res.append(CR.collect_batch(client, self.store, bid, items, model))
            return res

        def done(res):
            self._polling = False
            ended = [r for r in res if r["status"] == "ended"]
            n = sum(r["done"] for r in ended)
            errs = [e for r in ended for e in r["errors"]]
            if errs:
                self.app.log_errors([f"[{self.TITLE}] {e}" for e in errs])
            if ended:
                self.status.setText(f"Batch results are in: {n:,} image(s) reviewed" +
                                    (f", {len(errs)} failed" if errs else "") + " — Rescan shows them.")
                try:
                    self.app.toast.show_message(f"<b>Claude Review</b><br>Batch results are in: {n:,} image(s).",
                                                kind="success", actions=[("Show", self.start_scan)])
                except Exception:
                    pass
            else:
                left = sum((r["counts"] or {}).get("processing", 0) for r in res)
                if left:
                    self.status.setText(f"Batch still running: {left:,} image(s) to go.")
            self._conn_text()

        w = FuncWorker(work, self)
        w.done.connect(done)
        w.failed.connect(lambda e: (setattr(self, "_polling", False), self.status.setText(f"⚠ batch check: {e}")))
        w.finished.connect(w.deleteLater)
        w.start()
        self._pw = w

    # ------------------------------------------------------------------ actions
    def _apply(self):
        rows = [r for r in self.target_rows() if isinstance(r.payload, CR.Finding)]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) findings first.")
            return
        c = self.ctx or {}
        fixes, cls_moves, skipped, applied = {}, [], 0, []
        for r in rows:
            f = r.payload
            info = c["files"].get(r.key) or {}
            W, H = info.get("wh") or (0, 0)
            shapes = info.get("shapes") or []

            def rb(i):
                pts = shapes[i].get("points") or []
                xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
                return (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H)
            if c.get("cls") and f.type == "wrong_class" and f.suggested_class:
                cls_moves.append((r.key, f.suggested_class))
            elif f.type == "wrong_class" and f.shape is not None and f.suggested_class and W:
                fixes.setdefault(r.key, []).append(("relabel", f.shape, rb(f.shape), f.suggested_class))
            elif f.type == "missing_object" and f.box and f.suggested_class and W:
                x1, y1, x2, y2 = f.box
                fixes.setdefault(r.key, []).append(("add", f.suggested_class, (max(0, x1 / W), max(0, y1 / H),
                                                                               min(1, x2 / W), min(1, y2 / H))))
            elif f.type in ("not_an_object", "duplicate_label") and f.shape is not None and W:
                fixes.setdefault(r.key, []).append(("delete", f.shape, rb(f.shape)))
            else:
                skipped += 1
                continue
            applied.append(r)
        n = sum(len(v) for v in fixes.values()) + len(cls_moves)
        if not n:
            QMessageBox.information(self, self.TITLE, "None of these can be applied automatically (box fit and "
                                                      "guideline findings are fixed on the image: Claude's boxes are "
                                                      "approximate).")
            return
        if not self.confirm(f"Apply {n:,} suggestion(s)?" + (f" ({skipped} cannot be applied automatically)" if
                                                              skipped else "") + "\n\nThe originals are backed up "
                                                                                 "(Undo)."):
            return
        lines, errs = [], []
        if cls_moves:
            done = 0
            from AIO_QC.core import provenance
            for k, cl in cls_moves:
                with provenance.origin("model", "AI Review"):
                    ok, err = self.session.set_class([k], cl)
                done += len(ok)
                errs += err
            lines.append(f"Class changed for {done:,} image(s).")
        if fixes:
            from AIO_QC.core import provenance
            with provenance.origin("model", "AI Review"):
                st = self.session.apply_fixes(self, fixes, what="claude_review")
            if st["files"]:
                self._last_backup = st["backup"]
                self.undo_btn.setEnabled(True)
                self.session.set_review(st["changed"], REVIEW_FLAG, "Claude Review: changed — check it")
            lines.append(f"{st['done']:,} edit(s) in {st['files']:,} file(s); the files are flagged for a look.")
            if st["missing"]:
                lines.append(f"{st['missing']:,} skipped: the label changed since the review.")
            if st["skipped"]:
                lines.append(f"{st['skipped']:,} skipped: already there.")
            if st["refused"]:
                lines.append(f"{st['refused']:,} skipped: not a class of this dataset.")
            errs += st["errors"]
        for r in applied:
            self.store.verdict(r.payload.type, True)
        ids = {id(r) for r in applied}
        self.remove_rows(lambda x: id(x) in ids)
        self.summary.setText(self.summarize())
        self.report("Done", lines, errs)

    def _undo(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last Apply?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.report("Undone", [f"Restored {len(keys):,} file(s)."] +
                    ([f"{len(skipped):,} changed since and left alone."] if skipped else []), [])

    def _dismiss(self):
        rows = [r for r in self.target_rows() if isinstance(r.payload, CR.Finding)]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) findings first.")
            return
        c = self.ctx or {}
        for r in rows:
            rel = (c.get("files", {}).get(r.key) or {}).get("rel") or \
                os.path.relpath(r.key, str(self.session.root)).replace("\\", "/")
            self.store.dismiss(rel, r.payload)
        ids = {id(r) for r in rows}
        self.remove_rows(lambda x: id(x) in ids)
        self.summary.setText(self.summarize())

    def post_init(self):
        st = self.app.settings
        ready = (CR.find_claude(st.value("claude/cc_path", "") or "") is not None
                 if self.backend.currentData() == PLAN else
                 bool(os.environ.get(st.value("claude/key_env", CR.DEFAULT_KEY_ENV) or CR.DEFAULT_KEY_ENV)))
        if self.scope.currentData() == "current" and getattr(self.app, "current_key", None) and \
                self.store.allowed() and ready:
            QTimer.singleShot(0, self.start_scan)

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)
