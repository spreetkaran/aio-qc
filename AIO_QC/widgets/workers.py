# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Background workers. Nothing here touches widgets; results travel via signals."""
from __future__ import annotations

import base64
import time
import numpy as np
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QRunnable, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QImageReader

from AIO_QC.core import AnnotationError
from AIO_QC.core.io import decode_image


def load_qimage(path, image_data_b64: str | None = None) -> QImage:
    """
    Thread-safe image load (QImage, never QPixmap, off the GUI thread).
    QImageReader honours EXIF orientation; OpenCV is the fallback for codecs
    Qt lacks; LabelMe's embedded imageData is used when the file is missing.
    """
    if path is not None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        img = reader.read()
        if not img.isNull():
            return img
        arr = decode_image(path)
        if arr is not None:
            rgb = np.ascontiguousarray(arr[:, :, ::-1])
            h, w = rgb.shape[:2]
            return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()  # copy: own the buffer
    if image_data_b64:
        try:
            ba = QByteArray(base64.b64decode(image_data_b64))
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.ReadOnly)
            reader = QImageReader(buf)
            reader.setAutoTransform(True)
            img = reader.read()
            buf.close()
            return img
        except Exception:
            pass
    return QImage()


class IndexWorker(QThread):
    """Builds key -> Counter(ORIGINAL labels) for the whole dataset, in batches."""
    batch = pyqtSignal(list)          # [(key, Counter | None, error | None), ...]
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(bool)      # cancelled?

    def __init__(self, adapter, samples, cache=None, namespace: str = ""):
        super().__init__()
        self.adapter, self.samples = adapter, list(samples)
        self.cache, self.namespace = cache, namespace
        self.hits = 0
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        out, last, total = [], time.monotonic(), len(self.samples)
        cache, snap = self.cache, {}
        if cache is not None:
            try:
                cache.ensure_label_namespace(self.namespace)
                snap = cache.load_all_labels()
            except Exception:
                cache = None
        for i, s in enumerate(self.samples, 1):
            if self._stop:
                break
            hit = cache.lookup_labels(snap, s.ann_path) if cache is not None and s.ann_path else None
            if hit is not None:
                self.hits += 1
                out.append((s.key, hit[0], hit[1]))
            else:
                try:
                    ann = self.adapter.load(s, light=True)
                    raw = ann.raw_label_counts()          # shapes + image-level classes, original names
                    out.append((s.key, raw, None))
                    if cache is not None and s.ann_path:
                        cache.put_labels(s.ann_path, raw, None)
                except AnnotationError as exc:
                    out.append((s.key, None, str(exc)))
                    if cache is not None and s.ann_path:
                        cache.put_labels(s.ann_path, None, str(exc))
                except Exception as exc:  # never let one file kill the index (not cached: may be transient)
                    out.append((s.key, None, f"{type(exc).__name__}: {exc}"))
            now = time.monotonic()
            if now - last > 0.25 or i == total:
                self.batch.emit(out)
                self.progress.emit(i, total)
                out, last = [], now
        if out:
            self.batch.emit(out)
        if cache is not None:
            cache.flush()
        self.completed.emit(self._stop)


class PreloadSignals(QObject):
    loaded = pyqtSignal(str, int, QImage)   # key, generation, image


class PreloadTask(QRunnable):
    def __init__(self, signals: PreloadSignals, adapter, sample, generation: int):
        super().__init__()
        self.signals, self.adapter, self.sample, self.generation = signals, adapter, sample, generation

    def run(self):
        try:
            ann = self.adapter.load(self.sample)
            img = load_qimage(ann.image_path, ann.image_data)
        except Exception:
            img = QImage()
        self.signals.loaded.emit(self.sample.key, self.generation, img)


class ThumbSignals(QObject):
    loaded = pyqtSignal(str, int, QImage)


class ThumbTask(QRunnable):
    """Scaled decode (JPEG decodes at reduced size directly) for the filmstrip."""

    def __init__(self, signals: ThumbSignals, adapter, sample, generation: int, size: int = 112):
        super().__init__()
        self.signals, self.adapter, self.sample, self.generation, self.size = signals, adapter, sample, generation, size

    def run(self):
        img = QImage()
        try:
            path = self.sample.image_path
            if path is None or not path.exists():
                path = self.adapter.load(self.sample, light=True).image_path
            if path is not None:
                reader = QImageReader(str(path))
                reader.setAutoTransform(True)
                src = reader.size()
                if src.isValid() and src.width() > 0 and src.height() > 0:
                    k = min(self.size / max(src.width(), src.height()), 1.0)
                    reader.setScaledSize(src * k)
                img = reader.read()
        except Exception:
            img = QImage()
        self.signals.loaded.emit(self.sample.key, self.generation, img)
