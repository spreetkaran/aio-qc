# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
App-wide style corrections (a QProxyStyle over the platform style, installed once by the shell):

- Pressed / checked buttons keep their icon and text centred: the platform styles shift the contents 1 px right
  and down while a button is down (the toolbar's Drag / Select / pixel-view buttons looked off-centre).
- Buttons that open a menu and already say so in their text ("Add rule ▾", the zoom presets "▾") get no second,
  drawn menu arrow — one arrow, not two.
- Modal dialogs are freed once they close (release_after_exec): `SomeDialog(self).exec()` used to leave the dialog
  alive as a hidden child of its window for the rest of the session — every Settings / form / preview opened
  added one.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QProxyStyle, QPushButton, QStyle, QStyleOptionButton, \
    QStyleOptionToolButton, QToolButton

ARROWS = ("▾", "▼", "⏷")


def _says_arrow(w) -> bool:
    return isinstance(w, (QToolButton, QPushButton)) and any(a in (w.text() or "") for a in ARROWS)


class AppStyle(QProxyStyle):
    def pixelMetric(self, m, opt=None, w=None):
        if m in (QStyle.PixelMetric.PM_ButtonShiftHorizontal, QStyle.PixelMetric.PM_ButtonShiftVertical):
            return 0
        return super().pixelMetric(m, opt, w)

    # ---- one menu arrow
    @staticmethod
    def _tool_opt(opt, w):
        if _says_arrow(w) and isinstance(opt, QStyleOptionToolButton):
            o = QStyleOptionToolButton(opt)
            o.features &= ~(QStyleOptionToolButton.ToolButtonFeature.HasMenu
                            | QStyleOptionToolButton.ToolButtonFeature.MenuButtonPopup)
            return o
        return opt

    @staticmethod
    def _push_opt(opt, w):
        if _says_arrow(w) and isinstance(opt, QStyleOptionButton):
            o = QStyleOptionButton(opt)
            o.features &= ~QStyleOptionButton.ButtonFeature.HasMenu
            return o
        return opt

    def drawComplexControl(self, cc, opt, p, w=None):
        if cc == QStyle.ComplexControl.CC_ToolButton:
            opt = self._tool_opt(opt, w)
        super().drawComplexControl(cc, opt, p, w)

    def drawControl(self, ce, opt, p, w=None):
        if ce in (QStyle.ControlElement.CE_PushButton, QStyle.ControlElement.CE_PushButtonLabel,
                  QStyle.ControlElement.CE_PushButtonBevel):
            opt = self._push_opt(opt, w)
        elif ce in (QStyle.ControlElement.CE_ToolButtonLabel,):
            opt = self._tool_opt(opt, w)
        super().drawControl(ce, opt, p, w)

    def sizeFromContents(self, ct, opt, size, w=None):
        if ct == QStyle.ContentsType.CT_ToolButton:
            opt = self._tool_opt(opt, w)
        elif ct == QStyle.ContentsType.CT_PushButton:
            opt = self._push_opt(opt, w)
        return super().sizeFromContents(ct, opt, size, w)

    def drawPrimitive(self, pe, opt, p, w=None):
        # style sheets draw the indicator as a primitive of their own
        if pe == QStyle.PrimitiveElement.PE_IndicatorArrowDown and _says_arrow(w):
            return
        super().drawPrimitive(pe, opt, p, w)


_ORIG_EXEC = None


def release(dlg) -> bool:
    """Free a finished dialog that has a parent (deleteLater), unless it sets `keep_after_exec`."""
    try:
        if dlg.parent() is not None and not getattr(dlg, "keep_after_exec", False):
            dlg.deleteLater()
            return True
    except RuntimeError:                       # already deleted
        pass
    return False


def release_after_exec():
    """QDialog.exec() frees a dialog that has a parent once it returns (deleteLater: the caller can still read its
    values right after exec). A dialog meant to be shown again sets `keep_after_exec = True`."""
    global _ORIG_EXEC
    from PyQt6.QtWidgets import QDialog
    if _ORIG_EXEC is not None:
        return
    _ORIG_EXEC = QDialog.exec

    def exec_(self, *a, **k):
        r = _ORIG_EXEC(self, *a, **k)
        release(self)
        return r
    exec_.__doc__ = "QDialog.exec, then deleteLater() (AIO QC: dialogs never pile up in memory)"
    QDialog.exec = exec_


def install(app: QApplication | None = None):
    app = app or QApplication.instance()
    release_after_exec()
    if app is None or isinstance(app.style(), AppStyle):
        return
    base = app.style().name()
    from PyQt6.QtWidgets import QStyleFactory
    inner = QStyleFactory.create(base)
    # the proxy hands standardPalette() and everything else to the same platform style: colours (and following the
    # system's light / dark scheme) are unchanged
    app.setStyle(AppStyle(inner) if inner is not None else AppStyle())
