# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Windows frame integration: no PyQt nativeEvent override (it crashed natively at start-up on Windows), and a
start-up safety net that turns the integration off by itself after a crash inside it, or after three runs in a row
that ended while it was being set up — not after one."""
from PyQt6.QtWidgets import QWidget


def test_no_native_event_override():
    from AIO_QC.widgets import window_frame as wf
    py_classes = [c for c in wf.FrameHost.__mro__ if c.__module__.startswith("AIO_QC")]
    assert py_classes and not any("nativeEvent" in vars(c) for c in py_classes)


def _fake_api(calls, fail=False):
    class U:
        def GetWindowLongPtrW(self, h, i): return 0
        def SetWindowLongPtrW(self, h, i, v): calls.append(("style", v))
        def SetWindowPos(self, *a): calls.append(("pos",))

    class C:
        def SetWindowSubclass(self, *a):
            calls.append(("subclass",))
            return not fail

    class W:
        HWND = int
    return {"W": W, "u": U(), "c": C(), "cb": object(), "MARGINS": lambda *a: None,
            "ctypes": type("ct", (), {"windll": None, "byref": staticmethod(lambda x: x)})}


def test_guard_turns_off_after_a_crashed_setup(qapp, monkeypatch):
    from AIO_QC.utils import app_settings
    from AIO_QC.widgets import window_frame as wf
    monkeypatch.setattr(wf, "_WIN", True)
    monkeypatch.delenv("AIOQC_NATIVE_FRAME", raising=False)
    st = app_settings()
    calls = []
    monkeypatch.setattr(wf, "_win_api", lambda: _fake_api(calls))
    w = QWidget()
    st.setValue(wf.NATIVE_MIGRATED, True)
    st.setValue(wf.NATIVE_STRIKES, 0)
    st.setValue(wf.NATIVE_STATE, "")
    assert wf._win_native_frame(w) and ("subclass",) in calls
    assert st.value(wf.NATIVE_STATE) == "trying"           # until the window has lived a while
    # ... the app died here — twice more: only the third unsettled start turns it off, without touching the window
    for _ in range(2):
        calls.clear()
        assert wf._win_native_frame(w) and ("subclass",) in calls
    calls.clear()
    assert wf._win_native_frame(w) is False and calls == []
    assert st.value(wf.NATIVE_STATE) == "auto-off"
    monkeypatch.setenv("AIOQC_NATIVE_FRAME", "1")          # explicit retry
    assert wf.native_frame_allowed()
    monkeypatch.setenv("AIOQC_NATIVE_FRAME", "0")
    assert not wf.native_frame_allowed()
    st.setValue(wf.NATIVE_STATE, "")
    st.setValue(wf.NATIVE_STRIKES, 0)


def test_guard_settles_and_failure_is_harmless(qapp, qtbot, monkeypatch):
    from AIO_QC.utils import app_settings
    from AIO_QC.widgets import window_frame as wf
    monkeypatch.setattr(wf, "_WIN", True)
    monkeypatch.delenv("AIOQC_NATIVE_FRAME", raising=False)
    st = app_settings()
    st.setValue(wf.NATIVE_STATE, "")
    calls = []
    monkeypatch.setattr(wf, "_win_api", lambda: _fake_api(calls, fail=True))
    w = QWidget()
    assert wf._win_native_frame(w) is False and st.value(wf.NATIVE_STATE) == ""    # failed cleanly: retried next time
    monkeypatch.setattr(wf, "_win_api", lambda: _fake_api(calls))
    import AIO_QC.utils.qt_utils as qu
    monkeypatch.setattr(qu, "later", lambda ms, owner, fn: fn())                   # "the window lived a while"
    assert wf._win_native_frame(w) and st.value(wf.NATIVE_STATE) == "ok"
    st.setValue(wf.NATIVE_STATE, "")


def test_crash_in_the_window_callback_turns_it_off(qapp, monkeypatch):
    from AIO_QC.utils import app_settings
    from AIO_QC.widgets import lifecycle, window_frame as wf
    monkeypatch.setattr(wf, "_WIN", True)
    monkeypatch.delenv("AIOQC_NATIVE_FRAME", raising=False)
    st = app_settings()
    st.setValue(wf.NATIVE_MIGRATED, True)
    st.setValue(wf.NATIVE_STATE, "ok")
    old = 'File "F:\\\\x\\\\AIO_QC\\\\widgets\\\\window_frame.py", line 561 in restore'
    monkeypatch.setitem(lifecycle._STATE, "prev_crash", old)               # the 3.0.0 start-up crash: not this
    assert wf.native_frame_allowed()
    new = 'File "F:\\\\x\\\\AIO_QC\\\\widgets\\\\window_frame.py", line 70 in proc'
    monkeypatch.setitem(lifecycle._STATE, "prev_crash", new)
    assert not wf.native_frame_allowed() and st.value(wf.NATIVE_STATE) == "auto-off"
    st.setValue(wf.NATIVE_STATE, "")
