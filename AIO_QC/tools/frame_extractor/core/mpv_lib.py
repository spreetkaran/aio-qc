# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
libmpv playback engine - minimal ctypes binding of mpv's client + OpenGL render API (no python-mpv needed).

Why libmpv: it is the engine of mpv (and of many VLC-class players) and gives the best possible playback here:
  * hardware decoding on the GPU that drives the display (hwdec=auto-safe): zero-copy through the OpenGL
    interop where the driver offers it (DXVA2/WGL_NV_DX_interop, NVDEC/CUDA-GL, VAAPI-EGL, VideoToolbox),
    otherwise the fastest copy-back mode (d3d11va-copy, nvdec-copy ...). Falls back to multithreaded software
    decoding for anything the GPU can't decode (e.g. 8K H.264).
  * video-sync=display-resample: frames are timed to the display's vsync, so 59.94 fps on 60 Hz (or 29.97,
    23.976 ...) plays with even cadence and no periodic repeat / skip.
  * frame-exact seeking (hr-seek).
It renders straight into our OpenGL canvas, so ROI, detections and other overlays are still drawn on top.
Pausing, stepping, saving and analysis keep using FrameExtractor's own exact decoder.

Getting libmpv (details: packaging/LIBMPV.md, and Settings > Playback > libmpv... in the app):
  Windows - libmpv-2.dll from an mpv-dev build (zhongfly/mpv-winbuild "mpv-dev-lgpl-x86_64-*.7z" is LGPL);
  Linux   - the distribution's package (apt install libmpv2 / libmpv1, dnf install mpv-libs, pacman -S mpv);
  macOS   - brew install mpv (or a bundled copy in the app).
Searched: the chosen file, $AIOQC_LIBMPV, $FX_LIBMPV_DIR, frame_extractor/lib/, the per-user libmpv folder,
next to the app / in the app bundle, Homebrew / MacPorts (macOS), then the system library path.

Everything here is Qt-free; the GL canvas supplies get_proc_address and the target FBO.
"""
import ctypes as C
import ctypes.util
import os
import sys
import threading

from loguru import logger

# ------------------------------------------------------------------ constants (client.h / render.h)
FMT_NONE, FMT_STRING, FMT_FLAG, FMT_INT64, FMT_DOUBLE = 0, 1, 3, 4, 5
EV_NONE, EV_SHUTDOWN, EV_LOG, EV_END_FILE, EV_FILE_LOADED, EV_PLAYBACK_RESTART = 0, 1, 2, 7, 8, 21
END_FILE_ERROR = 4
R_INVALID, R_API_TYPE, R_GL_INIT, R_GL_FBO, R_FLIP_Y = 0, 1, 2, 3, 4
R_BLOCK_FOR_TARGET_TIME = 12
UPDATE_FRAME = 1


class _RenderParam(C.Structure):
    _fields_ = [("type", C.c_int), ("data", C.c_void_p)]


GET_PROC = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_char_p)
UPDATE_CB = C.CFUNCTYPE(None, C.c_void_p)


class _GLInit(C.Structure):
    # extra_exts was removed in client API 2.0; a trailing NULL is ignored by newer versions and needed by older
    _fields_ = [("get_proc_address", GET_PROC), ("get_proc_address_ctx", C.c_void_p), ("extra_exts", C.c_char_p)]


class _GLFbo(C.Structure):
    _fields_ = [("fbo", C.c_int), ("w", C.c_int), ("h", C.c_int), ("internal_format", C.c_int)]


class _Event(C.Structure):
    _fields_ = [("event_id", C.c_int), ("error", C.c_int), ("reply_userdata", C.c_uint64), ("data", C.c_void_p)]


class _LogMsg(C.Structure):
    _fields_ = [("prefix", C.c_char_p), ("level", C.c_char_p), ("text", C.c_char_p), ("log_level", C.c_int)]


class _EndFile(C.Structure):
    _fields_ = [("reason", C.c_int), ("error", C.c_int)]


# ------------------------------------------------------------------ library
_LIB = None
_LIB_ERR = None
_NAMES = {"win32": ["libmpv-2.dll", "mpv-2.dll", "mpv-1.dll", "libmpv.dll"],
          "darwin": ["libmpv.2.dylib", "libmpv.dylib"]}.get(sys.platform, ["libmpv.so.2", "libmpv.so.1", "libmpv.so"])
MIN_API = (1, 107)                       # mpv 0.33 (render API with the fields used here)


def user_lib_dir():
    """A per-user folder that is searched too and survives app updates (portable builds: inside data/)."""
    try:
        from AIO_QC.core.appdirs import data_home
        return str(data_home() / "libmpv")
    except Exception:
        return ""


def search_dirs():
    """(folder, why) in search order; the system's own library path comes after these."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # tools/frame_extractor/
    exe_dir = os.path.dirname(sys.executable)
    out = [(os.environ.get("FX_LIBMPV_DIR", ""), "FX_LIBMPV_DIR"),
           (os.path.join(here, "lib"), "bundled (frame_extractor/lib)"),
           (user_lib_dir(), "your libmpv folder"),
           (os.path.join(user_lib_dir(), "env", "lib") if user_lib_dir() else "", "downloaded (conda-forge)"),
           (getattr(sys, "_MEIPASS", ""), "app bundle"),
           (exe_dir, "next to the app")]
    if sys.platform == "darwin":
        out += [(os.path.normpath(os.path.join(exe_dir, "..", "Frameworks")), "app bundle Frameworks"),
                ("/opt/homebrew/lib", "Homebrew (Apple silicon)"), ("/usr/local/lib", "Homebrew (Intel)"),
                ("/opt/local/lib", "MacPorts")]
    elif sys.platform.startswith("linux"):
        out += [("/app/lib", "Flatpak")]
    seen, res = set(), []
    for d, why in out:
        if d and d not in seen:
            seen.add(d)
            res.append((d, why))
    return res


def _candidates(user_path=""):
    out = []
    env = os.environ.get("AIOQC_LIBMPV", "")
    for p in (user_path, env):
        if p:
            out.append(p)
    for d, _why in search_dirs():
        out += [os.path.join(d, n) for n in _NAMES]
    out += _NAMES                                             # system search path (ldconfig / PATH / dyld)
    found = ctypes.util.find_library("mpv")
    if found:
        out.append(found)
    return out


def _hint(err):
    e = str(err)
    if "undefined symbol" in e or "version `" in e or "Symbol not found" in e:
        return (" - this libmpv needs newer versions of libraries the app has already loaded (often GLib on "
                "Linux); use your system's libmpv package or one built for this system")
    if sys.platform == "win32" and ("193" in e or "not a valid Win32" in e):
        return " - wrong architecture (a 32-bit or ARM DLL in a 64-bit app, or the reverse)"
    if sys.platform == "win32" and ("126" in e or "could not be found" in e):
        return " - a DLL it depends on is missing; use a self-contained mpv-dev build (only libmpv-2.dll needed)"
    if "wrong ELF class" in e or "incompatible architecture" in e or "mach-o file, but is an incompatible" in e:
        return " - wrong architecture for this app"
    return ""


def probe(path):
    """Try one file. (lib or None, message)."""
    try:
        if sys.platform == "win32" and os.path.isabs(path):
            os.add_dll_directory(os.path.dirname(path))     # its own dependencies (if any) sit next to it
        lib = C.CDLL(path)
    except OSError as e:
        return None, f"{path}: {e}{_hint(e)}"
    try:
        _declare(lib)
    except AttributeError as e:
        return None, f"{path}: not a usable libmpv ({e})"
    ver = lib.mpv_client_api_version()
    major, minor = ver >> 16, ver & 0xFFFF
    if (major, minor) < MIN_API:
        return None, f"{path}: client API {major}.{minor} is too old (need mpv 0.33 or newer)"
    lib.path, lib.api = path, (major, minor)
    return lib, f"{path}: OK (client API {major}.{minor})"


def _promote_pending():
    """A DLL installed while the old one was in use waits as <name>.new: put it in place before loading."""
    d = user_lib_dir()
    if not d or not os.path.isdir(d):
        return
    for n in os.listdir(d):
        if n.endswith(".new"):
            try:
                os.replace(os.path.join(d, n), os.path.join(d, n[:-4]))
            except OSError:
                pass


def load_library(user_path=""):
    """The libmpv CDLL, or None (reason in last_error())."""
    global _LIB, _LIB_ERR
    if _LIB is not None:
        return _LIB
    _promote_pending()
    errors, user_err = [], None
    if user_path and os.path.isabs(user_path) and not os.path.isfile(user_path):
        user_err = f"{user_path}: the chosen file no longer exists"
    for cand in _candidates(user_path):
        if os.path.isabs(cand) and not os.path.isfile(cand):
            continue
        lib, msg = probe(cand)
        if lib is None:
            errors.append(msg)
            if user_path and cand == user_path:
                user_err = msg
            continue
        _LIB = lib
        _LIB_ERR = None
        logger.info(f"libmpv: {cand} (client API {lib.api[0]}.{lib.api[1]})")
        if user_err:
            logger.warning(f"libmpv: {user_err} - using {cand} instead")
        return _LIB
    real = [e for e in errors if "cannot open shared object" not in e and "could not be found" not in e
            and "image not found" not in e and "no such file" not in e.lower()]
    _LIB_ERR = user_err or (real[0] if real else "libmpv not found")
    return None


def reset():
    """Forget the loaded library choice (a new location was chosen). The old one stays mapped in the process."""
    global _LIB, _LIB_ERR
    _LIB, _LIB_ERR = None, None


def diagnose(user_path=""):
    """Every place looked at, for the libmpv dialog: [(path, status)] (status None = not there)."""
    rows = []
    for cand in _candidates(user_path):
        if os.path.isabs(cand):
            rows.append((cand, None if not os.path.isfile(cand) else ("loaded" if _LIB is not None and
                                                                     _LIB.path == cand else "found")))
    return rows


def last_error():
    return _LIB_ERR


def _declare(lib):
    p, i, d, s = C.c_void_p, C.c_int, C.c_double, C.c_char_p
    sig = {
        "mpv_client_api_version": ([], C.c_ulong),
        "mpv_create": ([], p),
        "mpv_initialize": ([p], i),
        "mpv_terminate_destroy": ([p], None),
        "mpv_set_option_string": ([p, s, s], i),
        "mpv_set_property_string": ([p, s, s], i),
        "mpv_set_property": ([p, s, i, p], i),
        "mpv_get_property": ([p, s, i, p], i),
        "mpv_command": ([p, C.POINTER(s)], i),
        "mpv_command_async": ([p, C.c_uint64, C.POINTER(s)], i),
        "mpv_wait_event": ([p, d], C.POINTER(_Event)),
        "mpv_wakeup": ([p], None),
        "mpv_request_log_messages": ([p, s], i),
        "mpv_error_string": ([i], s),
        "mpv_free": ([p], None),
        "mpv_render_context_create": ([C.POINTER(p), p, C.POINTER(_RenderParam)], i),
        "mpv_render_context_set_update_callback": ([p, UPDATE_CB, p], None),
        "mpv_render_context_update": ([p], C.c_uint64),
        "mpv_render_context_render": ([p, C.POINTER(_RenderParam)], i),
        "mpv_render_context_report_swap": ([p], None),
        "mpv_render_context_free": ([p], None),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res


def _b(x):
    return x if isinstance(x, bytes) else str(x).encode("utf-8")


# ------------------------------------------------------------------ player
class MpvPlayer:
    """One mpv instance. Thread rules: the event thread owns mpv_wait_event; GL calls (create_render, render,
    free_render, report_swap) happen on the GUI thread with the canvas' context current; everything else is
    thread-safe per libmpv."""

    def __init__(self, lib, hwdec="auto-safe", log_level="warn", basic=False):
        self.lib = lib
        self.h = lib.mpv_create()
        if not self.h:
            raise RuntimeError("mpv_create failed")
        opts = {
            "vo": "libmpv", "hwdec": hwdec, "video-sync": "display-resample", "interpolation": "no",
            "keep-open": "always", "pause": "yes", "idle": "yes", "audio": "no", "sid": "no",
            "hr-seek": "yes", "hr-seek-framedrop": "no", "osc": "no", "osd-level": "0",
            "input-default-bindings": "no", "input-vo-keyboard": "no", "terminal": "no",
            "msg-level": f"all={log_level}", "keepaspect": "yes", "video-unscaled": "no",
            "background-color": "#000000", "cache": "yes", "demuxer-max-bytes": "150MiB",
            "vd-lavc-threads": "0", "reset-on-next-file": "all",
        }
        if basic:                         # mpv's simple renderer: 8-bit intermediates, no fancy scaling - works
            opts["gpu-dumb-mode"] = "yes"   # on drivers whose float framebuffers misbehave
        for k, v in opts.items():
            r = lib.mpv_set_option_string(self.h, _b(k), _b(v))
            if r < 0 and k not in ("background-color", "osc"):
                logger.debug(f"mpv option {k}={v}: {self.err(r)}")
        from collections import deque
        self.log_tail = deque(maxlen=6)   # mpv's last warnings / errors (shown when a file can't be played)
        lib.mpv_request_log_messages(self.h, _b(os.environ.get("AIOQC_MPV_LOG", "warn")))
        r = lib.mpv_initialize(self.h)
        if r < 0:
            lib.mpv_terminate_destroy(self.h)
            self.h = None
            raise RuntimeError(f"mpv_initialize: {self.err(r)}")
        self.render_ctx = None
        self._gpa = self._ucb = None
        self.on_update = None             # called from mpv threads -> must only post (Qt signal)
        self.loaded = False
        self.failed = None
        self.closing = False              # being shut down: an end-of-file error then is expected
        self.restarts = 0                 # PLAYBACK_RESTART counter (a seek finished, frame ready)
        self.t0 = None                    # time-pos of the first frame (container start offset)
        self.path = None
        self._view = None
        self._quit = False
        self._ev = threading.Thread(target=self._events, name="mpv-events", daemon=True)
        self._ev.start()

    def err(self, code):
        try:
            return self.lib.mpv_error_string(code).decode()
        except Exception:
            return str(code)

    # -------------------------------------------------------------- events
    def _events(self):
        lib = self.lib
        while not self._quit:
            ev = lib.mpv_wait_event(self.h, 0.5)
            if not ev:
                continue
            e = ev.contents
            eid = e.event_id
            if eid == EV_NONE:
                continue
            if eid == EV_SHUTDOWN:
                break
            if eid == EV_LOG and e.data:
                m = C.cast(e.data, C.POINTER(_LogMsg)).contents
                txt = f"{(m.prefix or b'').decode('utf-8', 'replace')}: {(m.text or b'').decode('utf-8', 'replace').strip()}"
                self.log_tail.append(txt)
                logger.debug(f"[mpv] {txt}")
                continue
            if eid == EV_FILE_LOADED:
                self.loaded = True
                self.failed = None
            elif eid == EV_END_FILE and e.data:
                ef = C.cast(e.data, C.POINTER(_EndFile)).contents
                if ef.reason == END_FILE_ERROR and not self.closing:
                    why = "; ".join(self.log_tail)
                    self.failed = f"mpv could not play the file ({self.err(ef.error)})" + (f" - {why}" if why else "")
                    self.loaded = False
                    logger.warning(self.failed)
            elif eid == EV_PLAYBACK_RESTART:
                if self.t0 is None and self.loaded:
                    self.t0 = self.get_double("time-pos")
                self.restarts += 1

    # -------------------------------------------------------------- commands / properties
    def command(self, *args):
        arr = (C.c_char_p * (len(args) + 1))(*[_b(a) for a in args], None)
        r = self.lib.mpv_command(self.h, arr)
        if r < 0:
            logger.debug(f"mpv {args[0]}: {self.err(r)}")
        return r

    def set(self, name, value):
        if isinstance(value, bool):
            value = "yes" if value else "no"
        return self.lib.mpv_set_property_string(self.h, _b(name), _b(value))

    def set_double(self, name, value):
        v = C.c_double(value)
        return self.lib.mpv_set_property(self.h, _b(name), FMT_DOUBLE, C.byref(v))

    def get_double(self, name):
        v = C.c_double()
        return v.value if self.lib.mpv_get_property(self.h, _b(name), FMT_DOUBLE, C.byref(v)) >= 0 else None

    def get_flag(self, name):
        v = C.c_int()
        return bool(v.value) if self.lib.mpv_get_property(self.h, _b(name), FMT_FLAG, C.byref(v)) >= 0 else None

    def get_int(self, name):
        v = C.c_int64()
        return v.value if self.lib.mpv_get_property(self.h, _b(name), FMT_INT64, C.byref(v)) >= 0 else None

    def get_str(self, name):
        v = C.c_char_p()
        if self.lib.mpv_get_property(self.h, _b(name), FMT_STRING, C.byref(v)) < 0 or not v.value:
            return None
        s = v.value.decode("utf-8", "replace")
        self.lib.mpv_free(C.cast(v, C.c_void_p))
        return s

    # -------------------------------------------------------------- playback
    def load(self, path, aspect=None):
        """Open paused at the start. aspect 'w:h' makes mpv use the same (square-pixel) geometry as our canvas."""
        self.loaded, self.failed, self.t0, self.restarts, self.path = False, None, None, 0, path
        self.log_tail.clear()
        self.set("pause", True)
        if aspect:
            self.set("video-aspect-override", aspect)
        return self.command("loadfile", path, "replace") >= 0

    def set_display_fps(self, hz):
        for name in ("display-fps-override", "override-display-fps", "display-fps"):   # newest name first
            if self.set(name, f"{hz:.6f}") >= 0:
                return

    def seek(self, t, exact=True):
        return self.command("seek", f"{max(0.0, t):.6f}", "absolute+exact" if exact else "absolute+keyframes")

    def play(self):
        self.set("pause", False)

    def pause(self):
        self.set("pause", True)

    @property
    def time_pos(self):
        return self.get_double("time-pos")

    def set_view(self, zoom_log2, pan_x, pan_y):
        v = (round(zoom_log2, 5), round(pan_x, 6), round(pan_y, 6))
        if v != self._view:
            self._view = v
            self.set_double("video-zoom", v[0])
            self.set_double("video-pan-x", v[1])
            self.set_double("video-pan-y", v[2])

    def stats(self):
        g = self.get_str
        return dict(hwdec=g("hwdec-current") or "no", fps=self.get_double("estimated-vf-fps"),
                    display_fps=self.get_double("display-fps"), drops=self.get_int("frame-drop-count"),
                    decoder_drops=self.get_int("decoder-frame-drop-count"),
                    delayed=self.get_int("vo-delayed-frame-count"), sync=g("video-sync"),
                    codec=g("video-codec"))

    # -------------------------------------------------------------- render (GUI thread, GL context current)
    def create_render(self, get_proc_address):
        """get_proc_address(name: bytes) -> int address."""
        if self.render_ctx is not None:
            return True

        def gpa(_ctx, name):
            try:
                return int(get_proc_address(name) or 0)
            except Exception:
                return 0
        self._gpa = GET_PROC(gpa)
        init = _GLInit(self._gpa, None, None)
        api = C.c_char_p(b"opengl")
        params = (_RenderParam * 3)(_RenderParam(R_API_TYPE, C.cast(api, C.c_void_p)),
                                    _RenderParam(R_GL_INIT, C.cast(C.pointer(init), C.c_void_p)),
                                    _RenderParam(R_INVALID, None))
        ctx = C.c_void_p()
        r = self.lib.mpv_render_context_create(C.byref(ctx), self.h, params)
        if r < 0:
            raise RuntimeError(f"mpv_render_context_create: {self.err(r)}")
        self.render_ctx = ctx

        def upd(_):
            cb = self.on_update
            if cb is not None:
                try:
                    cb()
                except Exception:
                    pass
        self._ucb = UPDATE_CB(upd)
        self.lib.mpv_render_context_set_update_callback(self.render_ctx, self._ucb, None)
        return True

    def render(self, fbo, w, h, flip_y=True):
        if self.render_ctx is None:
            return False
        self.lib.mpv_render_context_update(self.render_ctx)
        f = _GLFbo(int(fbo), int(w), int(h), 0)
        flip = C.c_int(1 if flip_y else 0)
        params = (_RenderParam * 3)(_RenderParam(R_GL_FBO, C.cast(C.pointer(f), C.c_void_p)),
                                    _RenderParam(R_FLIP_Y, C.cast(C.pointer(flip), C.c_void_p)),
                                    _RenderParam(R_INVALID, None))
        return self.lib.mpv_render_context_render(self.render_ctx, params) >= 0

    def report_swap(self):
        if self.render_ctx is not None:
            self.lib.mpv_render_context_report_swap(self.render_ctx)

    def free_render(self):
        """Must run with the same GL context current; before terminate()."""
        if self.render_ctx is not None:
            self.lib.mpv_render_context_free(self.render_ctx)
            self.render_ctx = None

    def terminate(self):
        """Stop the event thread first (it must not wait on a destroyed handle), then destroy."""
        if not self.h:
            return
        self.on_update = None
        self._quit = True
        self.lib.mpv_wakeup(self.h)
        self._ev.join(2.0)
        if self.render_ctx is not None:
            logger.warning("mpv render context still alive at terminate (GL context gone?) - leaking it")
            self.render_ctx = None           # freeing without its GL context would crash
        self.lib.mpv_terminate_destroy(self.h)
        self.h = None


def view_params(zoom, cx, cy, vw, vh):
    """AIO QC's view (zoom relative to fit, centre in video pixels) -> mpv (video-zoom log2,
    video-pan-x/y as fractions of the scaled video size). Both fit the video to the window and centre it."""
    import math
    if not vw or not vh:
        return 0.0, 0.0, 0.0
    return math.log2(max(zoom, 1e-6)), -(cx / vw - 0.5), -(cy / vh - 0.5)
