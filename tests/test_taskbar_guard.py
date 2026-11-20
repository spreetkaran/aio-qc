# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Windows taskbar minimise / restore stays on: the safety net switches the frame integration off only when a crash
happens INSIDE it, or after three unsettled starts in a row (not after one kill or an unrelated crash); the old
guard's automatic "off" is undone once; your own "off" in Settings is kept; the Settings switch turns it back on."""
import pytest


@pytest.fixture
def wf(monkeypatch, qapp):
    from AIO_QC.utils import app_settings
    from AIO_QC.widgets import window_frame as WF
    st = app_settings()
    for k in (WF.NATIVE_STATE, WF.NATIVE_USER, WF.NATIVE_STRIKES, WF.NATIVE_MIGRATED):
        st.remove(k)
    monkeypatch.setattr(WF, "_WIN", True)
    monkeypatch.delenv("AIOQC_NATIVE_FRAME", raising=False)
    dump = {"text": ""}
    import AIO_QC.widgets.lifecycle as LC
    monkeypatch.setattr(LC, "previous_crash", lambda: dump["text"])
    WF._NOTICE["auto_off"] = False
    yield WF, st, dump
    for k in (WF.NATIVE_STATE, WF.NATIVE_USER, WF.NATIVE_STRIKES, WF.NATIVE_MIGRATED):
        st.remove(k)


def test_old_automatic_off_is_undone_once_and_user_off_is_kept(wf):
    WF, st, _d = wf
    st.setValue(WF.NATIVE_STATE, "off")                          # what the old guard left behind
    assert WF.native_frame_allowed() and st.value(WF.NATIVE_STATE) == ""
    setting = WF.native_frame_setting()
    setting.set(False)                                           # you turn it off
    assert not WF.native_frame_allowed() and setting.get() is False
    st.remove(WF.NATIVE_MIGRATED)                                # even a fresh migration keeps your choice
    assert not WF.native_frame_allowed()
    setting.set(True)
    assert WF.native_frame_allowed() and setting.get() is True


def test_unsettled_starts_need_three_strikes(wf):
    WF, st, dump = wf
    st.setValue(WF.NATIVE_MIGRATED, True)
    for n in (1, 2):
        st.setValue(WF.NATIVE_STATE, "trying")                   # the last run was killed before it settled
        assert WF.native_frame_allowed() and int(st.value(WF.NATIVE_STRIKES)) == n
    # an unrelated crash that merely passed through this file while showing a window: not the integration
    dump["text"] = ('File "C:/app/AIO_QC/widgets/window_frame.py", line 1260 in show\n'
                    'File "C:/app/AIO_QC/core/x.py", line 3 in process_batch\n')
    assert not WF._crash_in_integration(dump["text"])
    st.setValue(WF.NATIVE_STATE, "trying")
    assert not WF.native_frame_allowed() and st.value(WF.NATIVE_STATE) == "auto-off"      # third strike
    assert WF._NOTICE["auto_off"] and WF.native_frame_setting().get() is False
    WF.native_frame_setting().set(True)
    assert WF.native_frame_allowed() and int(st.value(WF.NATIVE_STRIKES)) == 0


def test_crash_inside_the_integration_turns_it_off_at_once(wf):
    WF, st, dump = wf
    st.setValue(WF.NATIVE_MIGRATED, True)
    dump["text"] = 'Fatal Python error\n  File "C:/app/AIO_QC/widgets/window_frame.py", line 80 in proc\n'
    st.setValue(WF.NATIVE_STATE, "trying")
    assert not WF.native_frame_allowed() and st.value(WF.NATIVE_STATE) == "auto-off"
    import os
    os.environ["AIOQC_NATIVE_FRAME"] = "1"                        # try again on purpose
    try:
        assert WF.native_frame_allowed()
    finally:
        del os.environ["AIOQC_NATIVE_FRAME"]
