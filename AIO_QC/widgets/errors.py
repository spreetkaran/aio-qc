# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One way to tell the user that something failed: what happened, why, and a button that fixes it (core.explain).

    show_error(parent, "Model Check", exc, app=window, retry=self.start_scan)

The box shows the plain sentence, the likely cause below it, the original text under "Show Details…", and at most
two buttons besides Close: the fix (Open Models & Environment, Install it, Show the log …) and Try again when the
caller can retry. The original text goes to the log either way.
"""
from __future__ import annotations

from loguru import logger
from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core.explain import Explanation, explain

LAST: list = []                       # the latest explanations shown (tests, diagnostics)


def _find_app(parent, app):
    if app is not None:
        return app
    w = parent
    while w is not None:
        if hasattr(w, "open_model_setup"):
            return w
        w = w.parent() if hasattr(w, "parent") else None
    return getattr(parent, "app", None)


def run_fix(e: Explanation, app, retry=None) -> bool:
    """Do what the fix button promises. -> True when something was done."""
    a = e.action
    if not a:
        return False
    if a == "retry":
        if retry is not None:
            retry()
            return True
        return False
    if app is None:
        return False
    if a == "model_setup" and hasattr(app, "open_model_setup"):
        app.open_model_setup()
        return True
    if a == "open_log" and hasattr(app, "show_log"):
        app.show_log()
        return True
    if a.startswith("install:"):
        pkg = a.split(":", 1)[1]
        fn = getattr(app, "install_into_env", None)
        if fn is not None:
            fn([pkg], retry)
            return True
        if hasattr(app, "open_model_setup"):
            app.open_model_setup()
            return True
    return False


def show_error(parent, title: str, err, app=None, retry=None) -> Explanation:
    e = explain(err)
    LAST.append(e)
    del LAST[:-20]
    logger.warning(f"[{title}] {e.detail or e.what}")
    app = _find_app(parent, app)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(e.what)
    if e.why:
        box.setInformativeText(e.why)
    if e.detail and e.detail != e.what:
        box.setDetailedText(e.detail)
    fix_btn = None
    if e.fix and (e.action != "retry" or retry is not None) and (e.action == "retry" or app is not None):
        fix_btn = box.addButton(e.fix, QMessageBox.ButtonRole.AcceptRole)
    retry_btn = None
    if retry is not None and e.action != "retry":
        retry_btn = box.addButton("Try again", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    clicked = box.clickedButton()
    if fix_btn is not None and clicked is fix_btn:
        run_fix(e, app, retry)
    elif retry_btn is not None and clicked is retry_btn:
        retry()
    return e


def message(err) -> str:
    """One line for a status bar or a toast: what · why."""
    e = explain(err)
    return e.what + (f" — {e.why}" if e.why else "")


def explain_html(err) -> str:
    """For a result label: the plain sentence, the cause in grey, the original text in a tooltip-sized line."""
    import html
    e = explain(err)
    out = f"<b>{html.escape(e.what)}</b>"
    if e.why:
        out += f"<br><span style='color:gray'>{html.escape(e.why)}</span>"
    if e.detail and e.detail != e.what:
        out += f"<br><span style='color:gray;font-size:small'>{html.escape(e.detail[:300])}</span>"
    return out
