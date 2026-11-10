# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Output settings shared by Save frame, markers, batch extraction and the CLI presets:
image format + quality (0-100, default 95) and the file-naming template.

  OutputQuickBar       toolbar: format · quality slider · "Naming…" button
  OutputSettingsDialog full editor: formats with notes, quality with a live size estimate on the current frame,
                       naming presets, token menu, validation, live example paths, collision policy
"""
import os
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox, QSlider, QSpinBox, QLabel, QToolButton, QDialog,
    QLineEdit, QMenu, QDialogButtonBox, QGroupBox, QPushButton, QPlainTextEdit,
)

from AIO_QC.tools.frame_extractor.core import imageio as iio
from AIO_QC.tools.frame_extractor.core.naming import (
    PRESETS,
    TOKENS,
    ON_EXISTS,
    DEFAULT_TEMPLATE,
    NameContext,
    render,
    validate,
)
from AIO_QC.tools.frame_extractor.utils import newIcon
from AIO_QC.utils.fx_settings import fx_settings

DEFAULTS = dict(ext=iio.DEFAULT_EXT, quality=iio.DEFAULT_QUALITY, template=DEFAULT_TEMPLATE, on_exists="overwrite")


def load_output_cfg():
    st = fx_settings()
    ext = iio.normalize_ext(st.value("save/ext", st.value("save_format", "jpg")))
    if ext not in iio.available_formats():
        ext = iio.DEFAULT_EXT
    tpl = st.value("save/template", DEFAULT_TEMPLATE) or DEFAULT_TEMPLATE
    if validate(tpl)[0]:
        tpl = DEFAULT_TEMPLATE
    pol = st.value("save/on_exists", "overwrite")
    q = int(st.value("save/quality", iio.DEFAULT_QUALITY))
    if st.value("save/quality_default") is None:     # v0.8 shipped 100 as the default: move to 95 once
        if q == 100:
            q = 95
            st.setValue("save/quality", q)
        st.setValue("save/quality_default", iio.DEFAULT_QUALITY)
    return dict(ext=ext, quality=q, template=tpl,
                on_exists=pol if pol in ON_EXISTS else "overwrite")


def save_output_cfg(cfg):
    st = fx_settings()
    st.setValue("save/ext", cfg["ext"])
    st.setValue("save/quality", int(cfg["quality"]))
    st.setValue("save/template", cfg["template"])
    st.setValue("save/on_exists", cfg["on_exists"])


def summary(cfg):
    label = iio.FORMATS[cfg["ext"]][0]
    q = f" · Q{cfg['quality']}" if iio.is_lossy(cfg["ext"]) else " · lossless"
    return f"{label}{q} · {cfg['template']}{iio.normalize_ext(cfg['ext'])}"


def _fill_formats(combo, current):
    combo.blockSignals(True)
    combo.clear()
    for ext in iio.available_formats():
        label, lossy, note = iio.FORMATS[ext]
        combo.addItem(f"{label} ({ext[1:]})", ext)
        combo.setItemData(combo.count() - 1, note, Qt.ItemDataRole.ToolTipRole)
    combo.setCurrentIndex(max(0, combo.findData(current)))
    combo.blockSignals(False)


class OutputQuickBar(QWidget):
    """Toolbar widget: format · quality slider · naming button. Emits changed(cfg)."""
    changed = pyqtSignal(dict)
    open_dialog = pyqtSignal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = dict(cfg)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)
        lay.addWidget(QLabel("Save as"))
        self.fmt = QComboBox()
        self.fmt.setToolTip('<b>Image format</b><br>JPEG: universal. PNG / TIFF / BMP: lossless. WebP: smaller, Q100 = lossless. AVIF / JPEG 2000: smallest, slower, less tool support. Only formats this build can write are listed.')
        self.fmt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        _fill_formats(self.fmt, self.cfg["ext"])
        self.fmt.currentIndexChanged.connect(self._on_format)
        lay.addWidget(self.fmt)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setFixedWidth(96)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setValue(int(self.cfg["quality"]))
        self.slider.valueChanged.connect(self._on_quality)
        self.qlabel = QLabel()
        self.qlabel.setMinimumWidth(34)
        lay.addWidget(self.slider)
        lay.addWidget(self.qlabel)
        more = QToolButton()
        more.setIcon(newIcon("layout"))
        more.setText("Naming…")
        more.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        more.setAutoRaise(True)
        more.setToolTip("Image format, quality and file-naming template…")
        more.clicked.connect(self.open_dialog.emit)
        lay.addWidget(more)
        self._sync()

    def set_cfg(self, cfg):
        self.cfg = dict(cfg)
        _fill_formats(self.fmt, self.cfg["ext"])
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.cfg["quality"]))
        self.slider.blockSignals(False)
        self._sync()

    def _sync(self):
        lossy = iio.is_lossy(self.cfg["ext"])
        self.slider.setEnabled(lossy)
        self.qlabel.setText(f"Q {self.cfg['quality']}" if lossy else "lossless")
        tip = (f"Quality {self.cfg['quality']} — {iio.quality_note(self.cfg['ext'], self.cfg['quality'])}"
               if lossy else f"{iio.FORMATS[self.cfg['ext']][0]} is lossless")
        self.slider.setToolTip(tip)
        self.qlabel.setToolTip(tip)
        self.setToolTip("Output: " + summary(self.cfg))

    def _on_format(self, i):
        self.cfg["ext"] = self.fmt.itemData(i)
        self._sync()
        self.changed.emit(dict(self.cfg))

    def _on_quality(self, v):
        self.cfg["quality"] = v
        self._sync()
        self.changed.emit(dict(self.cfg))


class _SizeWorker(QThread):
    done = pyqtSignal(int, int, float)      # token, bytes, ms

    def __init__(self, img, ext, quality, token):
        super().__init__()
        self.img, self.ext, self.quality, self.token = img, ext, quality, token

    def run(self):
        try:
            t = time.perf_counter()
            n = iio.encoded_size(self.img, self.ext, self.quality)
            self.done.emit(self.token, n, (time.perf_counter() - t) * 1000)
        except Exception:
            self.done.emit(self.token, -1, 0.0)


def _fmt_bytes(n):
    return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.2f} MB"


class OutputSettingsDialog(QDialog):
    def __init__(self, parent, cfg, context=None):
        """context: dict(video_path, frame, sec, fps, image (bgr or None), roi, output_dir)"""
        super().__init__(parent)
        self.setWindowTitle("Output format & file naming")
        self.setWindowIcon(newIcon("save"))
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 760, 640)
        self.cfg = dict(cfg)
        self.ctx = context or {}
        self._token = 0
        self._workers = []
        lay = QVBoxLayout(self)

        # ---- image format
        fbox = QGroupBox("Image format")
        g = QGridLayout(fbox)
        self.fmt = QComboBox()
        self.fmt.setToolTip('<b>Image format</b><br>JPEG: universal. PNG / TIFF / BMP: lossless. WebP: smaller, Q100 = lossless. AVIF / JPEG 2000: smallest, slower, less tool support. Only formats this build can write are listed.')
        _fill_formats(self.fmt, self.cfg["ext"])
        self.fmt.currentIndexChanged.connect(self._changed)
        self.fnote = QLabel()
        self.fnote.setWordWrap(True)
        self.fnote.setStyleSheet("color: gray;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.spin = QSpinBox()
        self.spin.setToolTip('Quality 1-100 for lossy formats (default 95). The estimate below shows the real size on the current frame.')
        self.spin.setRange(1, 100)
        self.slider.setValue(int(self.cfg["quality"]))
        self.spin.setValue(int(self.cfg["quality"]))
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self._changed)
        self.qnote = QLabel()
        presets = QHBoxLayout()
        for label, q in (("Max (100)", 100), ("Default · visually lossless (95)", 95), ("Balanced (85)", 85), ("Small (70)", 70)):
            b = QPushButton(label)
            b.setToolTip('Set this quality.')
            b.setFlat(True)
            b.clicked.connect(lambda _=False, q=q: self.spin.setValue(q))
            presets.addWidget(b)
        presets.addStretch(1)
        self.size = QLabel("")
        self.size.setStyleSheet("font-weight: bold;")
        g.addWidget(QLabel("Format:"), 0, 0)
        g.addWidget(self.fmt, 0, 1)
        g.addWidget(self.fnote, 0, 2, 1, 2)
        g.addWidget(QLabel("Quality:"), 1, 0)
        g.addWidget(self.slider, 1, 1, 1, 2)
        g.addWidget(self.spin, 1, 3)
        g.addWidget(self.qnote, 2, 1, 1, 3)
        g.addLayout(presets, 3, 1, 1, 3)
        g.addWidget(QLabel("Current frame:"), 4, 0)
        g.addWidget(self.size, 4, 1, 1, 3)
        g.setColumnStretch(2, 1)
        lay.addWidget(fbox)

        # ---- naming
        nbox = QGroupBox("File naming")
        n = QGridLayout(nbox)
        self.preset = QComboBox()
        self.preset.setToolTip('Ready-made naming templates - pick one, then edit it freely.')
        self.preset.addItem("Custom", "")
        for name, tpl in PRESETS.items():
            self.preset.addItem(f"{name}   {tpl}", tpl)
        self.preset.activated.connect(lambda i: self.preset.itemData(i) and self.tpl.setText(self.preset.itemData(i)))
        self.tpl = QLineEdit(self.cfg["template"])
        self.tpl.setToolTip("File name template (no extension). Tokens like {video} {frame:06d} {time} {tc} {counter}; '/' makes sub-folders. The examples below update as you type.")
        self.tpl.setFont(self._mono())
        self.tpl.textChanged.connect(self._changed)
        tok = QToolButton()
        tok.setToolTip('Insert a token at the cursor.')
        tok.setText("Insert token ▾")
        tok.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(tok)
        for t, desc in TOKENS.items():
            m.addAction(f"{{{t}}}    {desc}").triggered.connect(lambda _=False, t=t: self._insert("{" + t + "}"))
        m.addSeparator()
        for t, desc in (("{frame:06d}", "zero-padded frame"), ("{counter:05d}", "zero-padded counter"),
                        ("{sec:.2f}", "seconds, 2 decimals"), ("/", "sub-folder")):
            m.addAction(f"{t}    {desc}").triggered.connect(lambda _=False, t=t: self._insert(t))
        tok.setMenu(m)
        self.valid = QLabel()
        self.valid.setWordWrap(True)
        self.policy = QComboBox()
        self.policy.setToolTip('When a file with that name exists: overwrite it, keep both (_1, _2 …), or skip the frame.')
        for k, v in ON_EXISTS.items():
            self.policy.addItem(v, k)
        self.policy.setCurrentIndex(max(0, self.policy.findData(self.cfg["on_exists"])))
        self.policy.currentIndexChanged.connect(self._changed)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(self._mono())
        self.preview.setMaximumHeight(130)
        n.addWidget(QLabel("Preset:"), 0, 0)
        n.addWidget(self.preset, 0, 1, 1, 2)
        n.addWidget(QLabel("Template:"), 1, 0)
        n.addWidget(self.tpl, 1, 1)
        n.addWidget(tok, 1, 2)
        n.addWidget(self.valid, 2, 1, 1, 2)
        n.addWidget(QLabel("If the file exists:"), 3, 0)
        n.addWidget(self.policy, 3, 1, 1, 2)
        n.addWidget(QLabel("Examples:"), 4, 0, Qt.AlignmentFlag.AlignTop)
        n.addWidget(self.preview, 4, 1, 1, 2)
        help_ = QLabel("Tokens: " + " ".join("{" + t + "}" for t in TOKENS) +
                       " — format specs like {frame:06d} work; “/” makes sub-folders; illegal characters are replaced. "
                       "Label files (LabelMe / YOLO) always follow the image name.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray; font-size: 8pt;")
        n.addWidget(help_, 5, 0, 1, 3)
        lay.addWidget(nbox)

        row = QHBoxLayout()
        reset = QPushButton("Defaults")
        reset.setToolTip('Back to JPEG · quality 95 · {video}_frame_{frame} · overwrite.')
        reset.clicked.connect(self._defaults)
        row.addWidget(reset)
        row.addStretch(1)
        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        row.addWidget(self.bb)
        lay.addLayout(row)

        self._size_timer = QTimer(self)
        self._size_timer.setSingleShot(True)
        self._size_timer.setInterval(250)
        self._size_timer.timeout.connect(self._estimate)
        self._changed()

    @staticmethod
    def _mono():
        from PyQt6.QtGui import QFontDatabase
        return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

    def _insert(self, text):
        self.tpl.insert(text)
        self.tpl.setFocus()

    def _defaults(self):
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(DEFAULTS["ext"])))
        self.spin.setValue(DEFAULTS["quality"])
        self.tpl.setText(DEFAULTS["template"])
        self.policy.setCurrentIndex(max(0, self.policy.findData(DEFAULTS["on_exists"])))

    def result_cfg(self):
        return dict(ext=self.fmt.currentData(), quality=self.spin.value(), template=self.tpl.text().strip(),
                    on_exists=self.policy.currentData())

    def _changed(self, *_):
        c = self.result_cfg()
        label, lossy, note = iio.FORMATS[c["ext"]]
        self.fnote.setText(note)
        self.slider.setEnabled(lossy)
        self.spin.setEnabled(lossy)
        self.qnote.setText(("<b>" + iio.quality_note(c["ext"], c["quality"]) + "</b>") if lossy
                           else "<i>lossless — quality doesn't apply</i>")
        i = self.preset.findData(c["template"])
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(i if i > 0 else 0)
        self.preset.blockSignals(False)
        errors, warnings = validate(c["template"])
        if errors:
            self.valid.setText("<span style='color:#E53935'>✖ " + "; ".join(errors) + "</span>")
        elif warnings:
            self.valid.setText("<span style='color:#FB8C00'>⚠ " + "; ".join(warnings) + "</span>")
        else:
            self.valid.setText("<span style='color:#43A047'>✔ valid</span>")
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not errors)
        self._update_preview(c, errors)
        self._size_timer.start()

    def _update_preview(self, c, errors):
        if errors:
            self.preview.setPlainText("")
            return
        x = self.ctx
        video = os.path.splitext(os.path.basename(x.get("video_path") or "cam01_north.mp4"))[0]
        frame, fps = int(x.get("frame", 2503)), x.get("fps", 29.97)
        sec = x.get("sec", frame / fps)
        img = x.get("image")
        h, w = (img.shape[:2] if img is not None else (1080, 1920))
        roi = x.get("roi")
        base = x.get("output_dir") or "<output>"
        ext = c["ext"]

        def line(sub, kind, reason="", r=None, counter=1, ww=w, hh=h):
            ctx = NameContext(video, frame, sec, fps, counter, kind, reason, r, ww, hh, x.get("video_path"))
            return os.path.join(base, video, sub, *render(c["template"], ctx).split("/")) + ext

        rw, rh = ((roi[2] - roi[0], roi[3] - roi[1]) if roi else (w, h))
        out = [f"Save frame     {line('', 'frame')}",
               f"Save ROI crop  {line('', 'roi', r=roi or (100, 100, 740, 460), ww=rw, hh=rh)}",
               f"Marker         {line('markers', 'marker', counter=3)}",
               f"Batch          {line('batch_extract', 'batch', counter=17)}",
               f"Batch (mined)  {line('batch_extract', 'mined', 'uncertain', counter=18)}"]
        self.preview.setPlainText("\n".join(out))

    def _estimate(self):
        img = self.ctx.get("image")
        c = self.result_cfg()
        if img is None:
            self.size.setText("— load a video to see the file size per frame")
            return
        self._token += 1
        self.size.setText("estimating…")
        w = _SizeWorker(img, c["ext"], c["quality"], self._token)
        w.done.connect(self._on_size)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        w.start()

    def _on_size(self, token, n, ms):
        if token != self._token:
            return
        img = self.ctx.get("image")
        if n < 0:
            self.size.setText("could not encode")
            return
        raw = img.size
        self.size.setText(f"≈ {_fmt_bytes(n)} per frame ({img.shape[1]}×{img.shape[0]}) · "
                          f"{raw / max(1, n):.1f}:1 · encodes in {ms:.0f} ms · 1,000 frames ≈ {_fmt_bytes(n * 1000)}")

    def closeEvent(self, ev):
        for w in list(self._workers):
            w.wait(3000)
        super().closeEvent(ev)

    def done(self, r):
        for w in list(self._workers):
            w.wait(3000)
        super().done(r)
