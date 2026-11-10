# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
What each mode's Settings window shows (widgets.settings_dialog builds the window from these).

Every setting is bound to what already did the work before - the window's own actions, handlers and QSettings keys -
so moving a preference out of a menu changes where it is found, never how it behaves. The keys ("image.…", "fx.…",
"data.…", "app.…") are what Export / Import write; keep them stable.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from AIO_QC.widgets.settings_dialog import (Section, Setting, command, from_action, from_actions, from_store,
                                            general_section)


# ------------------------------------------------------------------ Image QC
def image_sections(win) -> list:
    a, st = win.a, win.settings
    has_ds = lambda: win.session is not None                            # noqa: E731
    from AIO_QC.core import backups as B
    from AIO_QC.core import storage as S

    def exact(v):
        B.EXACT_IMAGE_DATA = bool(v)
    days = dict(minimum=0, maximum=3650, step=1, suffix=" days")
    return [
        general_section(),
        Section("Interface", "layout", [
            from_action("image.compact_toolbar", a.compact_toolbar, "Compact toolbar (icons only)", default=True,
                        aliases=("toolbar", "icons", "text")),
            from_action("image.guide_cards", a.guide_cards, "Guideline cards", default=True,
                        description="Show a class's guideline next to the selected shape.",
                        aliases=("guidelines", "examples")),
            from_store("image.get_started", st, "get_started/at_start", "bool", False, "Show Get Started at start",
                       description="Open the guided overview every time the app starts (it always opens the first "
                                   "time).", aliases=("tour", "welcome")),
        ]),
        Section("Dataset", "folders", [
            from_action("image.watch", a.watch, "Watch the dataset for outside changes", default=True,
                        description="Pick up files changed, added or removed by other programs (re-read only, never "
                                    "written).", aliases=("reload", "refresh", "file system")),
            command("image.clear_cache", "Dataset cache",
                    "Clear…", win.clear_cache, enabled=has_ds,
                    description="Delete the open dataset's cache (.aio_qc/cache.sqlite: labels, image hashes, "
                                "model predictions). It is rebuilt when needed.", aliases=("sqlite", "reset")),
        ], "Applies to the dataset open in Image QC."),
        Section("Review", "review-ok", [
            from_action("image.auto_advance", a.auto_advance, "Go to the next file after marking", default=True,
                        description="After ✓ Mark OK or ⚑ Flag, the next file opens.",
                        aliases=("auto-advance", "next", "mark")),
        ]),
        Section("Storage & backups", "save", [
            from_store("image.storage_auto", st, "storage/auto", "bool", S.DEFAULTS["auto"],
                       "Clean up automatically", description="Run housekeeping after a dataset is indexed.",
                       aliases=("housekeeping", "retention")),
            from_store("image.backup_days", st, "storage/backup_days", "int", S.DEFAULTS["backup_days"],
                       "Keep backups for", description="Older backups are deleted (0 = keep forever).",
                       aliases=("history", "revert"), **days),
            from_store("image.session_days", st, "storage/session_days", "int", S.DEFAULTS["session_days"],
                       "Keep finished editor sessions for", description="Manifests of finished editor sessions.",
                       **days),
            from_store("image.result_days", st, "storage/result_days", "int", S.DEFAULTS["result_days"],
                       "Keep unused model / OCR results for", description="Cached results not used for this long "
                                                                          "are dropped (0 = keep).", **days),
            from_store("image.exact_image_data", st, "storage/exact_image_data", "bool", False,
                       "Keep embedded images byte-exact in backups", apply=exact,
                       description="Off: a backup leaves out a JSON's embedded image when it can be rebuilt from the "
                                   "image file (restores are checked by SHA-256). On: about the image size per file.",
                       aliases=("imageData",)),
            command("image.storage", "What is stored", "Storage…", win.open_storage, enabled=has_ds,
                    description="Size of the cache, backups and sessions of the open dataset - export, delete."),
        ], "Housekeeping of each dataset's .aio_qc folder."),
        Section("Models", "tool-model-setup", [
            command("image.models", "Models & Environment", "Open…", win.open_model_setup,
                    description="The Python environment, models, class mapping, device and image size used by "
                                "Model Check, Testing and the Frame Extractor.",
                    aliases=("python", "environment", "yolo", "onnx", "device", "gpu")),
            from_store("image.delete_temp_env", st, "inference/delete_temp_on_exit", "bool", True,
                       "Delete a temporary model environment on exit",
                       apply=lambda v: setattr(win.inference, "delete_temp_on_exit", bool(v)),
                       aliases=("temp", "venv")),
        ]),
        review_section(),
        Section("Keyboard", "keyboard", [
            command("image.shortcuts", "Keyboard shortcuts", "Show…", a.shortcuts.trigger,
                    description="Every key of every mode."),
            command("image.keymap", "Key map", "Show…", a.keymap.trigger),
        ]),
    ]


# ------------------------------------------------------------------ Frame Extractor
def fx_sections(win) -> list:
    st, a = win.settings, win.actions_
    from AIO_QC.tools.frame_extractor.core import preview_pipe as PP

    def ffmpeg_path():
        return st.value("playback/ffmpeg", "") or ""

    def set_ffmpeg(p):
        st.setValue("playback/ffmpeg", p)
        if win.engine:
            win.configure_realtime(force=True)

    return [
        general_section(),
        Section("Display", "eye", [
            from_actions("fx.renderer", win.renderer_actions, "Video renderer", default="auto",
                         labels={"auto": "Auto (GPU when available)", "gpu": "GPU (OpenGL)", "software": "Software"},
                         description="GPU: smooth, full-quality scaling and libmpv. Software: for computers whose "
                                     "OpenGL driver misbehaves.", aliases=("opengl", "cpu", "canvas")),
        ]),
        Section("Playback", "run", [
            from_actions("fx.engine", win.engine_actions, "Playback engine", default="auto",
                         labels={"auto": "libmpv when available (recommended)", "mpv": "libmpv",
                                 "builtin": "Built-in"},
                         description="libmpv: GPU decoding and display-synced timing. Pausing, stepping, saving and "
                                     "extraction always use the exact decoder.", aliases=("mpv", "player")),
            from_store("fx.libmpv_auto", st, "playback/libmpv_auto", "bool", True,
                       "Get libmpv automatically when it is missing",
                       description="Windows / macOS: downloads it in the background. Linux: asks once, then uses "
                                   "your package manager.", aliases=("download", "install", "mpv")),
            command("fx.libmpv", "libmpv", "Status & install…", win.open_libmpv_dialog,
                    description="Is libmpv in use, why not, get it now, or choose a file.", aliases=("mpv",)),
            from_actions("fx.realtime", win.playback_actions, "Real-time pipeline", default="auto",
                         labels={"auto": "Auto (videos ≥ 2560×1440)", "on": "Always", "off": "Off (exact decoder)"},
                         description="ffmpeg decodes on the GPU in a separate process and scales the picture "
                                     "(built-in engine).", aliases=("ffmpeg", "4k", "8k", "large")),
            Setting("fx.ffmpeg", "ffmpeg", "file", get=ffmpeg_path, set=set_ffmpeg, default="",
                    description="Empty: the bundled ffmpeg (imageio-ffmpeg) or the one on PATH (now: "
                                f"{PP.find_ffmpeg('') or 'not found'}).",
                    file_filter="ffmpeg (ffmpeg ffmpeg.exe);;All files (*)", aliases=("location", "path")),
        ]),
        Section("Decoding", "image", [
            from_actions("fx.decoder", win.decoder_actions, "Exact decoder", default="auto",
                         labels={"auto": "Auto (PyAV for VFR / rotated, else OpenCV)", "opencv": "OpenCV",
                                 "pyav": "PyAV (exact timestamps)"},
                         description="Used for pausing, stepping, saving and extraction.",
                         aliases=("opencv", "pyav", "vfr", "backend")),
            from_action("fx.hw_decode", win.hw_action, "Hardware decoding in the exact decoder", default=False,
                        description="Faster exact frames; rare driver crashes.", aliases=("gpu", "cuda", "dxva")),
        ]),
        Section("Output", "save", [
            Setting("fx.output_dir", "Output folder", "folder", get=lambda: win.output_dir,
                    set=lambda p: win.set_output_dir(p),
                    description="Where saved and extracted frames go (one sub-folder per video).",
                    aliases=("directory", "save", "frames")),
            command("fx.output_format", "Format & file names", "Edit…", a.output_settings.trigger,
                    description="Image format, quality and the file-naming template.",
                    aliases=("png", "jpg", "jpeg", "webp", "quality", "naming", "template")),
        ]),
        Section("Models", "tool-model-setup", [
            command("fx.models", "Detection models", "Models…", a.models.trigger,
                    description="YOLO models (.onnx / .pt / .engine) for detections, mining and pre-annotations.",
                    aliases=("yolo", "onnx", "detect")),
            from_action("fx.timeline_models", a.timeline_models, "Run models in timeline analysis", default=False,
                        aliases=("timeline", "mining")),
        ]),
        review_section(),
        Section("Keyboard", "keyboard", [
            command("fx.customize_keys", "Keyboard shortcuts", "Customize…", a.help.trigger,
                    description="Change any key of the Frame Extractor.", aliases=("keys", "hotkeys")),
            command("fx.key_reference", "Shortcut reference", "Show…", a.shortcut_ref.trigger),
        ]),
    ]


# ------------------------------------------------------------------ video review (Testing → Frame Extractor)
REVIEW_DEFAULTS = dict(mode="balanced", stride=2, lead_s=3.0, conf=0.25)


def review_defaults() -> dict:
    """The analysis settings a new review starts with (Settings ▸ Video review; Testing can change them per review)."""
    from AIO_QC.utils.fx_settings import testing_settings
    st = testing_settings()
    out = {}
    for k, v in REVIEW_DEFAULTS.items():
        try:
            out[k] = type(v)(st.value(f"review/{k}", v))
        except (TypeError, ValueError):
            out[k] = v
    if out["mode"] not in ("balanced", "best", "fixed"):
        out["mode"] = "balanced"
    return out


def review_section() -> Section:
    from AIO_QC.utils.fx_settings import testing_settings
    st = testing_settings()
    d = REVIEW_DEFAULTS
    return Section("Video review", "tool-testing", [
        from_store("review.mode", st, "review/mode", "choice", d["mode"], "Analysis mode",
                   choices=[("balanced", "Balanced - smooth real-time playback"),
                            ("best", "Best quality - every frame, playback waits"),
                            ("fixed", "Fixed step - every N-th frame")],
                   description="Balanced: every frame live when the models keep up; otherwise the frame step "
                               "adapts, frames in between are interpolated, and spare time fills them in.",
                   aliases=("detections", "playlist", "testing", "interpolate", "stride", "real-time")),
        from_store("review.stride", st, "review/stride", "int", d["stride"], "Fixed step",
                   description="Fixed step mode: every N-th frame is analysed.", minimum=2, maximum=30),
        from_store("review.lead", st, "review/lead_s", "float", d["lead_s"], "Look-ahead",
                   description="How far ahead of the playhead detections are kept ready.", minimum=0.5,
                   maximum=30.0, step=0.5, suffix=" s", decimals=1),
        from_store("review.conf", st, "review/conf", "float", d["conf"], "Show boxes from confidence",
                   description="The starting threshold; the player's slider changes it live.", minimum=0.01,
                   maximum=0.99, step=0.05, decimals=2),
    ], "Defaults for Models ▸ Testing ▸ Review videos (each review can change them).")


# ------------------------------------------------------------------ Data Studio
STUDIO_TUNABLES = (   # (key, QSettings name, module, attribute, default, factor, title, description, min, max, suffix)
    ("data.big_file", "studio/big_file_mb", "AIO_QC.core.tabular.io", "BIG_BYTES", 400, 1024 ** 2,
     "Open larger text files as a sample", "CSV / JSONL files above this size open as their first rows; exports and "
     "Spark jobs always use the whole file.", 1, 1_000_000, " MB"),
    ("data.sample_rows", "studio/sample_rows", "AIO_QC.core.tabular.io", "SAMPLE_ROWS", 1_000_000, 1,
     "Rows in that sample", "", 1_000, 100_000_000, " rows"),
    ("data.profile_rows", "studio/profile_rows", "AIO_QC.core.tabular.profile", "STAT_ROWS", 200_000, 1,
     "Rows the profile looks at", "A random sample of this many rows is profiled.", 1_000, 100_000_000, " rows"),
    ("data.hist_bins", "studio/hist_bins", "AIO_QC.core.tabular.profile", "HIST_BINS", 24, 1,
     "Histogram bars", "", 4, 200, ""),
    ("data.chart_points", "studio/chart_points", "AIO_QC.widgets.studio.chart_view", "MAX_POINTS", 20_000, 1,
     "Points drawn in a chart", "Larger tables are drawn as a random sample of this many points.", 1_000, 2_000_000,
     " points"),
    ("data.cache_mb", "studio/cache_mb", "AIO_QC.core.tabular.workspace", "CACHE_BUDGET", 1_500, 1024 ** 2,
     "Memory for intermediate results", "Older step results beyond this are recomputed when needed.", 100,
     1_000_000, " MB"),
)


def apply_studio_tunables(settings):
    """At start and on change: the stored values into the modules that use them."""
    import importlib
    for key, name, mod, attr, default, factor, *_ in STUDIO_TUNABLES:
        try:
            v = int(settings.value(name, default, type=int))
        except (TypeError, ValueError):
            v = default
        setattr(importlib.import_module(mod), attr, v * factor)


def studio_sections(win) -> list:
    st = win.st
    rows = []
    for key, name, mod, attr, default, factor, title, desc, lo, hi, suffix in STUDIO_TUNABLES:
        rows.append(from_store(key, st, name, "int", default, title, description=desc, minimum=lo, maximum=hi,
                               suffix=suffix, apply=lambda _v: apply_studio_tunables(st)))
    return [
        general_section(),
        Section("Large tables", "data-profile", rows[:2]),
        Section("Profile & charts", "data-chart", rows[2:5]),
        Section("Performance", "sliders", rows[5:]),
        Section("Spark", "data-spark", [
            from_store("data.spark_python", st, "studio/spark_python", "file", "",
                       "Python with PySpark", description="The environment Run on Spark… uses by default.",
                       aliases=("pyspark", "environment")),
            from_store("data.spark_url", st, "studio/spark_url", "text", "", "Spark master",
                       description="Empty: local. Example: spark://host:7077, yarn.", aliases=("cluster", "master")),
            from_store("data.spark_out", st, "studio/spark_out", "folder", "", "Spark output folder"),
        ]),
    ]


def windows_with_settings():
    return [w for w in QApplication.topLevelWidgets() if hasattr(w, "settings_sections")]
