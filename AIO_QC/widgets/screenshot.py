# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Screenshots of the image canvas (Image QC).

What is saved is exactly what the canvas shows, cropped to the image: the zoomed-in part at the zoom you are at,
brightness / contrast, the labels you ticked, outlines / fills / tags at their on-screen size, the model
predictions layer, a sketch drawing — at the screen's real resolution (sharp on high-DPI screens). Pre-click
feedback is left out: the hover highlight, the edit handles of a selected shape and a box being drawn. Widgets
floating over the canvas (guideline cards, the sketch / ROI bars, notifications) are never in it.

    E          save to the screenshots folder (no dialog):  <folder>/<dataset>/<sub-folders>/<file name>.png
    Shift+E    save as … (dialog)
    right-click on the image, the toolbar button, File ▸ Screenshot: the same, plus copy to the clipboard
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, QStandardPaths, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QRegion
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

FORMATS = {"png": "PNG", "jpg": "JPEG"}


def default_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation) or str(Path.home())
    return Path(base) / "AIO QC Screenshots"


def visible_image_rect(canvas) -> QRect:
    """The part of the viewport the image covers (viewport pixels), empty without an image."""
    pix = canvas.pixmap_item.pixmap()
    if pix.isNull():
        return QRect()
    poly = canvas.mapFromScene(canvas.pixmap_item.sceneBoundingRect())
    return poly.boundingRect().intersected(canvas.viewport().rect())


def capture_view(canvas) -> QImage:
    """The canvas as you see it, cropped to the image, at the screen's device resolution. Null image when
    nothing is shown."""
    area = visible_image_rect(canvas)
    if area.isEmpty():
        return QImage()
    vp = canvas.viewport()
    item = canvas.pixmap_item
    if item.timer.isActive():                        # the rested high-quality render is still pending: do it now
        item.timer.stop()
        item.refine()
    # pre-click feedback is not part of the picture
    hover = canvas.hover_group
    if hover is not None:
        canvas._set_hover(None)
    hidden = [h for h in list(getattr(canvas, "_handles", [])) + [getattr(canvas, "_draw_item", None)]
              if h is not None and h.isVisible()]
    for h in hidden:
        h.setVisible(False)
    try:
        dpr = vp.devicePixelRatioF()
        img = QImage(max(1, round(area.width() * dpr)), max(1, round(area.height() * dpr)),
                     QImage.Format.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(Qt.GlobalColor.transparent)
        # the viewport's own painting only (no child widgets floating over it)
        vp.render(img, QPoint(), QRegion(area), QWidget.RenderFlag.DrawWindowBackground)
    finally:
        for h in hidden:
            h.setVisible(True)
        if hover is not None and hover in canvas.shape_groups:
            canvas._set_hover(hover)
    img.setDevicePixelRatio(1.0)                     # a file is plain pixels
    return img


PNG_QUALITY = 80        # zlib level ~2: ~7× faster than Qt's default (6.2 s → 0.85 s for 12 MP), ~15 % larger


def save_image(img: QImage, path: Path, fmt: str = "png") -> bool:
    """Thread-safe (Qt releases the GIL while encoding). Screenshots are opaque: 3 channels, never alpha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    if fmt == "jpg":
        return img.save(str(path), "JPEG", 92)
    return img.save(str(path), "PNG", PNG_QUALITY)


def unique_path(path: Path) -> Path:
    """Never overwrite: name.png, name_2.png, name_3.png …"""
    if not path.exists():
        return path
    i = 2
    while True:
        p = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not p.exists():
            return p
        i += 1


class Screenshots:
    """The screenshot commands of one Image QC window."""

    def __init__(self, app):
        self.app = app
        self.last: Path | None = None

    # ---- settings
    @property
    def folder(self) -> Path:
        from AIO_QC.utils.save_dirs import folder_for
        return folder_for("screenshots", self.app.settings)

    @property
    def fmt(self) -> str:
        v = self.app.settings.value("screenshot/format", "png") or "png"
        return v if v in FORMATS else "png"

    def target_for(self, key: str) -> Path:
        """<folder>/<dataset>/<sub-folders of the file>/<file name>.<ext> — the dataset's own layout, so files with
        the same name in different folders never collide."""
        s = self.app.session
        src = Path(key)                              # the annotation (LabelMe) or the image (YOLO, classification)
        root = Path(s.root) if s else src.parent
        try:
            rel = src.parent.relative_to(root)
        except ValueError:
            rel = Path()
        return self.folder / (root.name or "dataset") / rel / f"{src.stem}.{self.fmt}"

    # ---- commands
    def _grab(self) -> QImage | None:
        app = self.app
        if app.session is None or app.current_key is None:
            app.statusBar().showMessage("Open an image first", 4000)
            return None
        img = capture_view(app.canvas)
        if img.isNull():
            app.statusBar().showMessage("Nothing to capture — the image is not shown", 4000)
            return None
        return img

    def save(self) -> Path | None:
        """E: straight into the screenshots folder."""
        img = self._grab()
        if img is None:
            return None
        path = unique_path(self.target_for(self.app.current_key))
        return self._write(img, path)

    def save_as(self) -> Path | None:
        """Shift+E: choose name / folder / format."""
        img = self._grab()
        if img is None:
            return None
        from AIO_QC.utils.save_dirs import ask_save
        target = self.target_for(self.app.current_key)
        f, chosen = ask_save(self.app, "Save screenshot", target.name, "PNG (*.png);;JPEG (*.jpg *.jpeg)",
                             kind="screenshots", fallback=target.parent if target.parent.is_dir() else None)
        if not f:
            return None
        path = Path(f)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path = path.with_suffix(".jpg" if chosen.startswith("JPEG") else ".png")
        return self._write(img, path, fmt="jpg" if path.suffix.lower() in (".jpg", ".jpeg") else "png")

    def copy(self) -> bool:
        img = self._grab()
        if img is None:
            return False
        QApplication.clipboard().setImage(img)
        self.app.statusBar().showMessage(f"Screenshot copied ({img.width()} × {img.height()} px)", 4000)
        return True

    def open_folder(self):
        d = self.last.parent if self.last is not None and self.last.parent.is_dir() else self.folder
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _write(self, img: QImage, path: Path, fmt: str | None = None) -> Path | None:
        app = self.app
        try:
            ok = save_image(img, path, fmt or self.fmt)
        except OSError as exc:
            ok, why = False, str(exc)
        else:
            why = "the file could not be written"
        if not ok:
            QMessageBox.warning(app, "Screenshot", f"Could not save the screenshot:\n{path}\n\n{why}")
            return None
        self.last = path
        app.statusBar().showMessage(f"Screenshot saved: {path}", 6000)
        toast = getattr(app, "toast", None)
        if toast is not None:
            stamp = time.strftime("%H:%M:%S")
            toast.show_message(f"<b>Screenshot saved</b> · {stamp}<br>{path.name} ({img.width()} × {img.height()} px)",
                               kind="success", key="screenshot", timeout_ms=5000,
                               actions=[("Open folder", self.open_folder)])
        return path

    # ---- right-click on the image
    def menu_actions(self) -> list:
        a = self.app.a
        return [a.screenshot, a.screenshot_as, a.screenshot_copy, None, a.screenshot_folder]

    def canvas_menu(self, view_pos):
        """Right-click inside the image (outside it there is no menu). ROI mode keeps right-click for itself."""
        from PyQt6.QtWidgets import QMenu
        app = self.app
        if getattr(getattr(app, "roi", None), "active", False):
            return
        m = QMenu(app)
        for act in self.menu_actions():
            if act is None:
                m.addSeparator()
            else:
                m.addAction(act)
        m.exec(app.canvas.viewport().mapToGlobal(view_pos))

