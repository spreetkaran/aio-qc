# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Settings in every mode: the gear and File ▸ Settings…, the window (sections, search, reset, import / export),
every setting bound to the code that did the work before, and menus that keep only commands."""
from __future__ import annotations

import json

import pytest
from PyQt6.QtGui import QAction

from .test_modes import shell3  # noqa: F401  (fixture)

MODES = ("image", "video", "data")


def _dialog(win):
    from AIO_QC.widgets.settings_dialog import SettingsDialog
    return SettingsDialog(win, win.settings_sections(), "Settings")


def _menu_texts(win):
    out = []

    def walk(menu, path):
        for a in menu.actions():
            if a.isSeparator():
                continue
            if a.menu():
                walk(a.menu(), path + [a.text()])
            else:
                out.append(" › ".join(path + [a.text()]))
    for a in win.menuBar().actions():
        if a.menu():
            walk(a.menu(), [a.text()])
    return out


@pytest.mark.parametrize("mode", MODES)
def test_gear_and_file_menu_open_settings(shell3, qtbot, mode, monkeypatch):  # noqa: F811
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog, QToolButton
    from AIO_QC.widgets.settings_dialog import SettingsDialog
    w = shell3.window(mode)
    qtbot.addWidget(w)
    corner = w.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    assert corner.objectName() == "settingsCorner"
    gear = corner.findChild(QToolButton, "settingsButton")
    assert gear is not None and corner.findChild(type(w.mode_switch)) is w.mode_switch
    act = w.findChild(QAction, "open_settings")
    assert act is not None and act.shortcut().toString() == "Ctrl+,"
    file_menu = w.menuBar().actions()[0].menu()
    acts = [a for a in file_menu.actions() if not a.isSeparator()]
    assert act in acts and acts[-1] is not act                  # Exit / Quit stays last
    seen = []
    monkeypatch.setattr(QDialog, "exec", lambda self: seen.append(self) or 0)
    gear.click()
    act.trigger()
    assert len(seen) == 2 and all(isinstance(d, SettingsDialog) for d in seen)


@pytest.mark.parametrize("mode", MODES)
def test_every_setting_reads_and_round_trips(shell3, qtbot, mode):  # noqa: F811
    w = shell3.window(mode)
    qtbot.addWidget(w)
    dlg = _dialog(w)
    qtbot.addWidget(dlg)
    assert dlg.rows and len({s.title for s in dlg.sections}) == len(dlg.sections)
    for key, row in dlg.rows.items():
        s = row.s
        assert key.split(".")[0] in ("app", "review", {"image": "image", "video": "fx", "data": "data"}[mode])
        if s.kind in ("button", "info"):
            continue
        v0 = s.value()
        assert v0 is not None, key
        if s.kind == "bool":
            dlg.apply(s, not v0)
            assert s.value() == (not v0), key
            dlg.apply(s, v0)
            assert s.value() == v0, key
        elif s.kind == "choice" and (s.enabled is None or s.enabled()):
            if key == "fx.renderer":
                continue                                        # swaps the video view: covered in test_fx_main
            others = [c[0] for c in s.choices if c[0] != v0]
            target = others[0]
            dlg.apply(s, target)
            if s.value() == target:                             # (a disabled choice - e.g. PyAV missing - stays)
                dlg.apply(s, v0)
            assert s.value() == v0, key
        elif s.kind == "int":
            dlg.apply(s, int(s.minimum if s.minimum is not None else 0) + 1)
            dlg.apply(s, v0)
            assert s.value() == v0, key


@pytest.mark.parametrize("mode", MODES)
def test_search_reset_export_import(shell3, qtbot, tmp_path, mode, monkeypatch):  # noqa: F811
    from PyQt6.QtWidgets import QFileDialog
    w = shell3.window(mode)
    qtbot.addWidget(w)
    dlg = _dialog(w)
    qtbot.addWidget(dlg)
    dlg.show()
    # search by an alias / description word, activate the first hit: it scrolls there and lights the row up
    word = {"image": "retention", "video": "mpv", "data": "pyspark"}[mode]
    dlg.search.setText(word)
    assert dlg.nav.count() >= 1 and dlg.nav.item(0).data(256)[0] == "setting"
    key = dlg.nav.item(0).data(256)[1]
    dlg._activate(dlg.nav.item(0))
    assert dlg._lit is dlg.rows[key] and dlg.search.text() == ""
    dlg.search.setText("zzqqxx")
    assert dlg.nav.count() == 1 and dlg.nav.item(0).text() == "No matching settings"
    dlg.search.clear()
    assert dlg.nav.count() == len(dlg.sections)
    # change one exportable setting, export, reset all, import -> the change is back
    s = next(r.s for r in dlg.rows.values() if r.s.kind in ("bool", "int") and r.s.has_default and r.s.exportable
             and (r.s.enabled is None or r.s.enabled()))
    other = (not s.default) if s.kind == "bool" else int(s.default) + 1
    dlg.apply(s, other)
    assert dlg.rows[s.key].reset.isVisibleTo(dlg)
    f = tmp_path / "s.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(f), "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(f), "")))
    dlg.export_file()
    data = json.loads(f.read_text())
    assert data["settings"][s.key] == other
    dlg.reset_all()                                            # (the harness answers "Yes")
    assert s.value() == s.default and not dlg.rows[s.key].reset.isVisibleTo(dlg)
    dlg.import_file()
    assert s.value() == other
    dlg.rows[s.key].reset.click()
    assert s.value() == s.default
    as_text = ("on" if other else "off") if s.kind == "bool" else str(other)     # files may hold strings
    n, problems = dlg.load_values({"no.such.key": 1, s.key: as_text})
    assert n == 1 and problems == ["no.such.key: not a setting here"] and s.value() == other
    dlg.apply(s, s.default)


def test_menus_keep_commands_preferences_moved(shell3, qtbot):  # noqa: F811
    for mode in MODES:
        w = shell3.window(mode)
        qtbot.addWidget(w)
        texts = _menu_texts(w)
        assert not any("\n" in t for t in texts), [t for t in texts if "\n" in t]
        assert any(t.endswith("Settings…") for t in texts)
        joined = "\n".join(texts)
        for moved in ("Compact Toolbar", "Watch Dataset", "Guideline Cards", "Auto-advance", "Clear Dataset Cache",
                      "Renderer", "Decoder", "Engine:", "Real-time pipeline", "ffmpeg location", "libmpv…",
                      "Customize keyboard", "Output format", "Animations"):
            assert moved not in joined, (mode, moved)
    fx = shell3.window("video")
    assert all(a.menu() is None or a.menu().title() != "Settings" for a in fx.menuBar().actions())
    names = {aid for aid, _a, _c in fx._registry()}           # palette + shortcut customisation still see them
    assert {"output_settings", "help", "renderer_gpu", "animations_off", "engine_builtin", "decoder_auto",
            "hw_decode"} <= names
    for aid in ("output_settings", "help"):
        act = next(a for i, a, _c in fx._registry() if i == aid)
        assert act in fx.actions()                              # shortcuts work although the menu is hidden


def test_auto_advance_and_animations_persist_and_sync(shell3, qtbot):  # noqa: F811
    from AIO_QC.widgets import motion
    img = shell3.window("image")
    fx = shell3.window("video")
    for w in (img, fx):
        qtbot.addWidget(w)
    d = _dialog(img)
    qtbot.addWidget(d)
    s = d.rows["image.auto_advance"].s
    d.apply(s, False)
    assert img.settings.value("review/auto_advance", True, type=bool) is False and not img.a.auto_advance.isChecked()
    d.apply(s, True)
    a = d.rows["app.animations"].s
    before = motion.mode()
    d.apply(a, "off")
    assert motion.mode() == "off" and not img.a.animations.isChecked()
    assert fx.findChild(QAction, "animations_off").isChecked()
    d.apply(a, before)


def test_studio_tunables_reach_the_code(shell3, qtbot):  # noqa: F811
    from AIO_QC.core.tabular import io, profile, workspace
    from AIO_QC.widgets.studio import chart_view
    w = shell3.window("data")
    qtbot.addWidget(w)
    d = _dialog(w)
    qtbot.addWidget(d)
    d.apply(d.rows["data.big_file"].s, 7)
    d.apply(d.rows["data.sample_rows"].s, 5000)
    d.apply(d.rows["data.profile_rows"].s, 3000)
    d.apply(d.rows["data.chart_points"].s, 4000)
    d.apply(d.rows["data.cache_mb"].s, 200)
    assert io.BIG_BYTES == 7 * 1024 ** 2 and io.SAMPLE_ROWS == 5000 and profile.STAT_ROWS == 3000
    assert chart_view.MAX_POINTS == 4000 and workspace.CACHE_BUDGET == 200 * 1024 ** 2
    import pandas as pd
    p = profile.profile(pd.DataFrame({"x": range(10_000)}))
    assert p is not None
    d.reset_all()
    assert io.BIG_BYTES == 400 * 1024 ** 2 and profile.STAT_ROWS == 200_000


def test_fx_output_folder_and_ffmpeg(shell3, qtbot, tmp_path):  # noqa: F811
    fx = shell3.window("video")
    qtbot.addWidget(fx)
    d = _dialog(fx)
    qtbot.addWidget(d)
    out = tmp_path / "frames"
    d.apply(d.rows["fx.output_dir"].s, str(out))
    assert fx.output_dir == str(out) and out.is_dir() and fx.settings.value("output_dir") == str(out)
    d.apply(d.rows["fx.ffmpeg"].s, "")
    assert fx.settings.value("playback/ffmpeg", "") == ""


def test_setting_definitions_are_checked():
    from AIO_QC.widgets.settings_dialog import Setting
    with pytest.raises(ValueError):
        Setting("x", "X", "slider")
    s = Setting("x", "X", "choice", choices=[(1, "one"), ("b", "bee")])
    assert s.valid("1") == 1 and s.valid("b") == "b"
    with pytest.raises(ValueError):
        s.valid("zz")
    n = Setting("n", "N", "int", minimum=1, maximum=5)
    assert n.valid("3") == 3
    with pytest.raises(ValueError):
        n.valid(9)
