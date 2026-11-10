# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Real PyQt6 (offscreen). Every Python exception raised inside a Qt slot / event and every Qt critical message
fails the test that caused it; modal dialogs are answered automatically (and recorded)."""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AIO_QC_TESTING", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

ERRORS: list = []
MESSAGES: list = []
NONBOOL: list = []          # setEnabled / setVisible / setChecked called with something other than a real bool


def _strict_bool_setters():
    """Wrap every Qt class's own setEnabled / setVisible / setChecked (and friends): a non-bool argument is
    recorded (PyQt6 raises TypeError for None / str / list and silently accepts ints)."""
    import inspect
    from PyQt6 import QtGui, QtWidgets
    names = ("setEnabled", "setVisible", "setChecked", "setHidden", "setDisabled")
    for mod in (QtWidgets, QtGui):
        for _n, cls in inspect.getmembers(mod, inspect.isclass):
            for m in names:
                if m not in cls.__dict__ or getattr(cls.__dict__[m], "_strict", False):
                    continue
                orig = getattr(cls, m)

                def wrap(self, v, *a, _o=orig, _m=m):
                    if type(v) is not bool:
                        NONBOOL.append(f"{type(self).__name__}.{_m}({v!r})\n" + "".join(
                            traceback.format_stack(limit=4)[:-1]))
                    return _o(self, v, *a)
                wrap._strict = True
                try:
                    setattr(cls, m, wrap)
                except (TypeError, AttributeError):
                    pass


_strict_bool_setters()


def _hook(t, v, tb):
    ERRORS.append("".join(traceback.format_exception(t, v, tb)))


@pytest.fixture(scope="session", autouse=True)
def _isolated_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    os.environ["HOME"] = str(home)
    os.environ["XDG_CONFIG_HOME"] = str(home / ".config")
    os.environ["XDG_DATA_HOME"] = str(home / ".local" / "share")
    os.environ["AIOQC_NO_LIBMPV_DOWNLOAD"] = "1"     # tests never download libmpv (test_libmpv drives it)
    yield home


@pytest.fixture(scope="session")
def qapp_args():
    return ["aio-qc-tests"]


@pytest.fixture(autouse=True)
def _errors(qapp, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMenu, QMessageBox
    ERRORS.clear()
    MESSAGES.clear()
    NONBOOL.clear()
    monkeypatch.setattr(sys, "excepthook", _hook)

    def box(kind, answer):
        def f(*a, **k):
            MESSAGES.append((kind, str(a[2] if len(a) > 2 else "")[:300]))
            return answer
        return f
    Yes = QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, "information", staticmethod(box("info", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(box("warning", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(box("critical", QMessageBox.StandardButton.Ok)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(box("question", Yes)))
    monkeypatch.setattr(QMessageBox, "about", staticmethod(box("about", None)))
    monkeypatch.setattr(QMessageBox, "aboutQt", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: (MESSAGES.append(("box", self.text()[:300])),
                                                          int(Yes))[1])
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: (MESSAGES.append(("menu", self.title())), next(
        (x for x in self.actions() if x.isEnabled() and not x.isSeparator() and x.menu() is None), None))[1])
    monkeypatch.setattr(QDialog, "exec", lambda self: (MESSAGES.append(("dialog", type(self).__name__)), 0)[1])
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: (a[3][0] if len(a) > 3 and a[3]
                                                                                 else "", False)))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], "")))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))
    yield
    errs = list(ERRORS)
    ERRORS.clear()
    nb = list(dict.fromkeys(NONBOOL))
    NONBOOL.clear()
    crit = [m for m in MESSAGES if m[0] == "critical"]
    assert not errs, "exceptions inside Qt:\n" + "\n".join(errs)
    assert not crit, f"critical message boxes: {crit}"
    assert not nb, "non-bool passed to a bool setter:\n" + "\n".join(nb[:10])
