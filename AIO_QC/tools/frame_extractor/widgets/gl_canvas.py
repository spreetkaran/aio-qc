# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
GPU video canvas (OpenGL via QOpenGLWidget) - the media-player path.

  * Full-resolution frames go to the GPU as they are: NV12 from the real-time pipeline (1.5 bytes / pixel,
    YUV->RGB in the shader) or BGR from the exact decoder. Zooming during playback therefore shows original
    pixels, and scaling costs the CPU nothing.
  * Presentation is paced by the display: while playing, every swap (vsync) asks for the frame that is due
    at the *next* refresh, so frames land on vsync with even cadence, like mpv / VLC.
  * High-quality scaling: multi-tap area filter when minifying (8K in a 1080p window without aliasing),
    bilinear when magnifying, nearest from 3x so single pixels can be inspected.
  * Overlays (ROI, detections, OSD, timecode) are the shared QPainter code, drawn over the GL picture.
  * Safety: if OpenGL can't be initialised, a shader fails, an upload raises, or the first frame reads back
    wrong (self-test), `failed` is emitted and the app swaps in the software canvas.
"""

import numpy as np
from loguru import logger
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QSurfaceFormat, QVector2D, QVector3D, QVector4D
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import (QOpenGLShaderProgram, QOpenGLShader, QOpenGLBuffer, QOpenGLVersionFunctionsFactory,
                            QOpenGLVersionProfile)

from AIO_QC.tools.frame_extractor.widgets.video_canvas import CanvasMixin
from AIO_QC.tools.frame_extractor.widgets.gl_shaders import VERTEX, FRAGMENT, yuv_uniforms, taps_for
from AIO_QC.utils.qt_utils import later   # owner-bound timer: never fires after `self` is gone

GL_TEXTURE_2D, GL_UNSIGNED_BYTE, GL_FLOAT = 0x0DE1, 0x1401, 0x1406
GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB = 0x1909, 0x190A, 0x1907
GL_LINEAR, GL_NEAREST, GL_CLAMP_TO_EDGE = 0x2601, 0x2600, 0x812F
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T = 0x2801, 0x2800, 0x2802, 0x2803
GL_UNPACK_ALIGNMENT, GL_TEXTURE0, GL_TRIANGLE_STRIP, GL_COLOR_BUFFER_BIT = 0x0CF5, 0x84C0, 0x0005, 0x4000
GL_MAX_TEXTURE_SIZE = 0x0D33
GL_BGRA, GL_RGBA = 0x80E1, 0x1908


def request_vsync_format():
    """Call before QApplication creates windows: swap interval 1 (vsync) and a compatibility context."""
    fmt = QSurfaceFormat.defaultFormat()
    fmt.setSwapInterval(1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)


class GLVideoCanvas(CanvasMixin, QOpenGLWidget):
    resized = pyqtSignal()
    wheel_step = pyqtSignal(int)
    roi_changed = pyqtSignal(object)
    zoom_changed = pyqtSignal(float)
    cursor_moved = pyqtSignal(object)
    failed = pyqtSignal(str)            # GPU path unusable -> the app switches to the software canvas
    mpv_failed = pyqtSignal(str)        # libmpv rendering unusable -> the app plays with the built-in engine
    _mpv_frame = pyqtSignal()           # posted from mpv's threads: a new frame is ready
    gpu = True

    def __init__(self, text="Load a video", parent=None):
        QOpenGLWidget.__init__(self, parent)
        self._init_canvas(text)
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
        self._ok = False
        self._dead = False
        self._tex = {}                  # name -> (id, w, h, fmt)
        self._uploaded = None           # id() of the frame currently in the textures
        self._mode = 0
        self._tex_size = (0, 0)
        self._color = yuv_uniforms("601", False)
        self._self_tested = False
        self._upload_mode = None        # how this PyQt build accepts pixel data
        self.present_hook = None        # callable(present_at) while playing; returns nothing
        self.playing = False
        self.frameSwapped.connect(self._on_swapped)
        self._mpv = None                # core.mpv_lib.MpvPlayer when libmpv plays
        self._mpv_on = False            # show mpv's picture (playing, or paused until the exact frame arrives)
        self._mpv_wait = None           # show it only after this many PLAYBACK_RESTARTs (fresh seek)
        self._bind_fbo = None
        self._mpv_frame.connect(self.update)

    # ------------------------------------------------------------------ API
    def set_colorspace(self, space, full_range):
        self._color = yuv_uniforms(space, full_range)
        self.update()

    # ------------------------------------------------------------------ libmpv
    def attach_mpv(self, player):
        self._mpv = player
        player.on_update = self._mpv_frame.emit        # thread-safe: queued to the GUI thread

    def set_mpv_active(self, on, after_restarts=None):
        self._mpv_on = bool(on)
        self._mpv_wait = after_restarts
        self.update()

    @property
    def mpv_showing(self):
        m = self._mpv
        return bool(m is not None and self._mpv_on and m.loaded and not self._dead
                    and (self._mpv_wait is None or m.restarts > self._mpv_wait))

    def release_mpv(self):
        """Free mpv's render context with our GL context current (must precede MpvPlayer.terminate)."""
        m = self._mpv
        self._mpv, self._mpv_on = None, False
        if m is None or m.render_ctx is None:
            return
        try:
            m.closing = True                   # unload first: freeing the context under a loaded file makes mpv
            m.command("stop")                  # try (and fail) to re-open its video output
            self.makeCurrent()
            m.free_render()
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def _mpv_init_render(self, m):
        """mpv's OpenGL render context on our GL context (which must be current)."""
        ctx = self.context()

        def gpa(name):
            f = ctx.getProcAddress(name)
            return int(f) if f else 0
        m.create_render(gpa)
        import ctypes
        addr = gpa(b"glBindFramebuffer")
        self._bind_fbo = ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_uint)(addr) if addr else None
        vao = gpa(b"glBindVertexArray")
        self._bind_vao = ctypes.CFUNCTYPE(None, ctypes.c_uint)(vao) if vao else None
        self._after_mpv_gl()

    def _after_mpv_gl(self):
        """mpv shares our GL context and leaves its own state behind (pixel-unpack buffer, row length, vertex array,
        program, blending ...). Our uploads and QPainter assume GL defaults - a bound unpack buffer or a row length
        turns the next exact frame into garbage colours - so put the defaults back after every mpv GL call."""
        g = getattr(self, "gl", None)
        if g is None:
            return
        try:
            for target in (0x88EC, 0x88EB, 0x8892):              # PIXEL_UNPACK / PIXEL_PACK / ARRAY buffer
                g.glBindBuffer(target, 0)
            for pname in (0x0CF2, 0x0CF3, 0x0CF4, 0x0D02, 0x0D03, 0x0D04):   # (UN)PACK_ROW_LENGTH / SKIP_*
                g.glPixelStorei(pname, 0)
            g.glPixelStorei(0x0D05, 4)                           # PACK_ALIGNMENT
            g.glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
            if getattr(self, "_bind_vao", None) is not None:
                self._bind_vao(0)
            g.glUseProgram(0)
            for cap in (0x0BE2, 0x0C11, 0x0B71, 0x0B90, 0x0B44, 0x8DB9):   # BLEND SCISSOR DEPTH STENCIL CULL SRGB
                g.glDisable(cap)
            g.glColorMask(True, True, True, True)
            for unit in range(4):
                g.glActiveTexture(GL_TEXTURE0 + unit)
                g.glBindTexture(GL_TEXTURE_2D, 0)
            g.glActiveTexture(GL_TEXTURE0)
            while g.glGetError():                                # e.g. SRGB unknown on this context
                pass
        except Exception as e:
            logger.debug(f"GL state reset after mpv: {e}")

    def prepare_mpv(self):
        """Create mpv's render context now, before a file is loaded: with vo=libmpv, mpv can only open its video
        output while a render context exists - loading first fails ('no audio or video data played').
        True when ready, False while OpenGL is not initialised yet (call again later); raises if mpv refuses."""
        m = self._mpv
        if m is None or m.render_ctx is not None:
            return m is not None
        ctx = self.context()
        if not self._ok or self._dead or ctx is None or not ctx.isValid():
            return False
        self.makeCurrent()
        try:
            self._mpv_init_render(m)
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass
        return True

    def verify_mpv(self, ref_bgr):
        """Does mpv's picture look like the video? Renders mpv's current (paused) frame into a small offscreen
        framebuffer and compares it with ref_bgr, our exact decoder's picture of the same frame.
        True / False, or None when it can't tell (a nearly uniform frame, OpenGL not ready).
        Some drivers accept everything mpv asks for and still draw nothing (seen: Mesa llvmpipe draws black with
        mpv's default 16-bit float intermediates); this catches that on any platform instead of guessing."""
        import cv2
        m = self._mpv
        ctx = self.context()
        if m is None or ref_bgr is None or not self._ok or self._dead or ctx is None or not ctx.isValid():
            return None
        h0, w0 = ref_bgr.shape[:2]
        sc = min(1.0, 256.0 / max(w0, h0))
        w, h = max(16, int(round(w0 * sc))), max(16, int(round(h0 * sc)))
        ref = cv2.resize(ref_bgr, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        if float(ref.std()) < 6.0:
            return None                                   # flat frame: black-vs-black proves nothing
        from PyQt6.QtGui import QImage
        from PyQt6.QtOpenGL import QOpenGLFramebufferObject
        self.makeCurrent()
        try:
            if m.render_ctx is None:
                self._mpv_init_render(m)
            fbo = QOpenGLFramebufferObject(w, h)
            m.set_view(0.0, 0.0, 0.0)                     # whole frame, fitted: it fills the w x h target
            self._after_mpv_gl()
            m.render(fbo.handle(), w, h, flip_y=False)
            self._after_mpv_gl()
            img = fbo.toImage().convertToFormat(QImage.Format.Format_RGB32)
            fbo.release()
            if self._bind_fbo is not None:
                self._bind_fbo(0x8D40, self.defaultFramebufferObject())
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        got = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3].astype(np.float32)
        k = (5, 5)
        g, r = cv2.blur(got, k), cv2.blur(ref, k)
        err = min(float(np.abs(g - r).mean()), float(np.abs(g[::-1] - r).mean()))   # either row order
        logger.info(f"libmpv picture check: mean difference {err:.1f} (0-255)")
        self.update()
        return err < 24.0

    def _draw_mpv(self):
        m = self._mpv
        if m.render_ctx is None:
            self._mpv_init_render(m)
        if self._vw:
            from AIO_QC.tools.frame_extractor.core.mpv_lib import view_params
            m.set_view(*view_params(self._zoom, self._cx, self._cy, self._vw, self._vh))
        dpr = self.devicePixelRatioF()
        fbo = self.defaultFramebufferObject()
        self._after_mpv_gl()                   # mpv expects default GL state too (QPainter leaves its own)
        m.render(fbo, int(self.width() * dpr), int(self.height() * dpr), flip_y=True)
        if self._bind_fbo is not None:
            self._bind_fbo(0x8D40, fbo)        # GL_FRAMEBUFFER: hand Qt its FBO back for the overlays
        self._after_mpv_gl()

    def show_frame(self, frame):
        self._mpv_on = False                   # an exact / own frame replaces mpv's picture
        self._frame = frame
        self._image, self._bgr, self._text = frame.image, frame.bgr, None
        self._dets = []
        self.update()

    def set_frame(self, image, bgr=None):          # compatibility with the software canvas API
        self._image, self._bgr, self._text = image, bgr, None
        self._dets = []
        self.update()

    def set_playing(self, on):
        self.playing = bool(on)
        if on:
            self.update()

    def refresh_interval(self):
        scr = self.screen()
        hz = scr.refreshRate() if scr is not None else 60.0
        return 1.0 / (hz if hz and hz > 20 else 60.0)

    # ------------------------------------------------------------------ GL setup
    def _fail(self, why):
        if not self._dead:
            self._dead = True
            logger.warning(f"GPU canvas disabled: {why}")
            later(0, self, lambda: self.failed.emit(why))

    def initializeGL(self):
        try:
            prof = QOpenGLVersionProfile()
            prof.setVersion(2, 0)
            self.gl = QOpenGLVersionFunctionsFactory.get(prof, self.context())
            if self.gl is None:
                raise RuntimeError("OpenGL 2.0 functions are not available")
            self.gl.initializeOpenGLFunctions()
            self.prog = QOpenGLShaderProgram(self)
            if not self.prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX):
                raise RuntimeError("vertex shader: " + self.prog.log())
            if not self.prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT):
                raise RuntimeError("fragment shader: " + self.prog.log())
            self.prog.bindAttributeLocation("a_pos", 0)
            if not self.prog.link():
                raise RuntimeError("link: " + self.prog.log())
            self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self.vbo.create()
            self.vbo.bind()
            quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], np.float32).tobytes()
            self.vbo.allocate(quad, len(quad))
            self.vbo.release()
            try:
                v = self.gl.glGetIntegerv(GL_MAX_TEXTURE_SIZE)
                self._max_tex = int(v[0] if isinstance(v, (list, tuple)) else v)
            except Exception:
                self._max_tex = 8192
            self.context().aboutToBeDestroyed.connect(self.release_mpv)
            self._ok = True
            ctx = self.context().format()
            logger.info(f"GPU canvas: OpenGL {ctx.majorVersion()}.{ctx.minorVersion()}, "
                        f"swap interval {ctx.swapInterval()}")
        except Exception as e:
            self._ok = False
            self._fail(f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ textures
    def _texture(self, name, w, h, fmt):
        t = self._tex.get(name)
        g = self.gl
        if t is None or t[1:] != (w, h, fmt):
            if t is not None:
                g.glDeleteTextures(1, [t[0]])
            tid = g.glGenTextures(1)
            tid = tid[0] if isinstance(tid, (list, tuple)) else int(tid)
            g.glBindTexture(GL_TEXTURE_2D, tid)
            for p, v in ((GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                         (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)):
                g.glTexParameteri(GL_TEXTURE_2D, p, v)
            g.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            if max(w, h) > getattr(self, "_max_tex", 8192):
                raise RuntimeError(f"{w}x{h} exceeds this GPU's max texture size {self._max_tex}")
            ch = {GL_LUMINANCE: 1, GL_LUMINANCE_ALPHA: 2, GL_RGB: 3, GL_BGRA: 4}[fmt]
            internal = GL_RGBA if fmt == GL_BGRA else fmt
            self._pixels(lambda data: g.glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, GL_UNSIGNED_BYTE, data),
                         np.zeros((h, w * ch), np.uint8), w * h * ch)
            t = (tid, w, h, fmt)
            self._tex[name] = t
        return t[0]

    def _pixels(self, call, arr, expected=None):
        """Pass pixel data in whatever form this PyQt build's OpenGL wrappers accept (learned once).
        The buffer must be exactly what GL will read: a short buffer would make the driver read past it."""
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        if expected is not None and arr.nbytes != expected:
            raise ValueError(f"pixel buffer is {arr.nbytes} bytes, GL expects {expected}")
        modes = [self._upload_mode] if self._upload_mode else ["array", "bytes"]
        err = None
        for m in modes:
            try:
                call(arr if m == "array" else arr.tobytes())
                self._upload_mode = m
                return
            except TypeError as e:
                err = e
        raise err

    def _upload(self, frame):
        g = self.gl
        g.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        nv12 = getattr(frame, "nv12", None)
        if nv12 is not None:
            y, uv = nv12
            h, w = y.shape[:2]
            for unit, name, arr, fmt, tw, th in ((0, "y", y, GL_LUMINANCE, w, h),
                                                 (1, "uv", uv, GL_LUMINANCE_ALPHA, w // 2, h // 2)):
                g.glActiveTexture(GL_TEXTURE0 + unit)
                g.glBindTexture(GL_TEXTURE_2D, self._texture(name, tw, th, fmt))
                self._pixels(lambda d: g.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, th, fmt, GL_UNSIGNED_BYTE, d), arr,
                             tw * th * (1 if fmt == GL_LUMINANCE else 2))
            self._mode, self._tex_size = 0, (w, h)
            return True
        src = frame.bgr if frame.bgr is not None else getattr(frame, "small", None)
        if src is None:
            return False
        h, w = src.shape[:2]
        g.glActiveTexture(GL_TEXTURE0 + 2)
        g.glBindTexture(GL_TEXTURE_2D, self._texture("rgb", w, h, GL_RGB))
        if src.ndim != 3 or src.shape[2] != 3:
            return False
        if getattr(self, "_bgra_ok", True):
            # 4-byte BGRA is the format drivers upload fastest (3-byte RGB often takes a slow, converting path)
            try:
                import cv2
                bgra = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
                g.glGetError()
                g.glBindTexture(GL_TEXTURE_2D, self._texture("bgra", w, h, GL_BGRA))
                self._pixels(lambda d: g.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, d),
                             bgra, w * h * 4)
                if g.glGetError() == 0:
                    self._mode, self._tex_size = 2, (w, h)
                    return True
            except Exception:
                pass
            self._bgra_ok = False
            logger.info("GPU canvas: BGRA upload unavailable, using RGB")
        g.glBindTexture(GL_TEXTURE_2D, self._texture("rgb", w, h, GL_RGB))
        self._pixels(lambda d: g.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, d), src,
                     w * h * 3)
        self._mode, self._tex_size = 1, (w, h)
        return True

    # ------------------------------------------------------------------ paint
    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        p = QPainter(self)
        try:
            p.beginNativePainting()
            if self.mpv_showing:
                try:
                    self._draw_mpv()
                except Exception as e:           # mpv unusable here: built-in engine takes over
                    self._mpv_on = False
                    later(0, self, lambda e=e: self.mpv_failed.emit(f"{type(e).__name__}: {e}"))
                    self._draw_video()
            else:
                self._draw_video()
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
        finally:
            self._restore_gl_state()      # before QPainter draws text: see _restore_gl_state
            p.endNativePainting()
        self._paint_overlays(p)
        p.end()
        if self._ok and not self._self_tested and self._uploaded is not None:
            self._self_tested = True
            QTimer.singleShot(50, self._self_test)

    def _restore_gl_state(self):
        """Put back the OpenGL state QPainter relies on. Crucially GL_UNPACK_ALIGNMENT = 4: Qt uploads text
        glyphs into its glyph-cache texture assuming the default alignment; leaving our value (1, needed for
        odd-width video planes) makes every glyph row start a few bytes off - skewed / sheared overlay text."""
        g = getattr(self, "gl", None)
        if g is None or not self._ok:
            return
        try:
            g.glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
            for unit in (2, 1, 0):
                g.glActiveTexture(GL_TEXTURE0 + unit)
                g.glBindTexture(GL_TEXTURE_2D, 0)
            g.glActiveTexture(GL_TEXTURE0)
            if getattr(self, "prog", None) is not None:
                self.prog.release()
            if getattr(self, "vbo", None) is not None:
                self.vbo.release()
        except Exception:
            pass

    def _draw_video(self):
        g = self.gl if self._ok else None
        dpr = self.devicePixelRatioF()
        W, H = int(self.width() * dpr), int(self.height() * dpr)
        if g is None:
            return
        g.glViewport(0, 0, W, H)
        g.glClearColor(0.0, 0.0, 0.0, 1.0)
        g.glClear(GL_COLOR_BUFFER_BIT)
        f = self._frame
        if f is None or not self._vw or self._dead:
            return
        if self._uploaded != id(f):
            import time as _t
            t_up = _t.perf_counter()
            if not self._upload(f):
                return
            self._uploaded = id(f)
            self.upload_ms = 0.9 * getattr(self, "upload_ms", 0.0) + 0.1 * (_t.perf_counter() - t_up) * 1000
        # geometry: image rect in NDC
        tl, br = self.v2w(0, 0), self.v2w(self._vw, self._vh)
        lw, lh = max(1.0, self.width()), max(1.0, self.height())
        x0, x1 = tl.x() / lw * 2 - 1, br.x() / lw * 2 - 1
        y_top, y_bot = 1 - tl.y() / lh * 2, 1 - br.y() / lh * 2
        # filtering: source texels per output (physical) pixel
        tw, th = self._tex_size
        out_px_per_tex = (br.x() - tl.x()) * dpr / max(1, tw)
        src_per_px = 1.0 / max(1e-6, out_px_per_tex)
        taps = taps_for(src_per_px)
        filt = GL_NEAREST if out_px_per_tex >= 3.0 else GL_LINEAR
        units = (0, 1) if self._mode == 0 else (2,)
        for unit in units:
            g.glActiveTexture(GL_TEXTURE0 + unit)
            name = {0: "y", 1: "uv", 2: "bgra" if self._mode == 2 else "rgb"}[unit]
            g.glBindTexture(GL_TEXTURE_2D, self._tex[name][0])
            g.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
            g.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
        pr = self.prog
        pr.bind()
        pr.setUniformValue("u_y", 0)
        pr.setUniformValue("u_uv", 1)
        pr.setUniformValue("u_rgb", 2)
        pr.setUniformValue("u_mode", int(self._mode))
        pr.setUniformValue("u_taps", int(taps))
        pr.setUniformValue("u_rect", QVector4D(x0, y_bot, x1, y_top))
        pr.setUniformValue("u_footprint", QVector2D(src_per_px / max(1, tw), src_per_px / max(1, th)))
        r, gg, b, y0 = self._color
        pr.setUniformValue("u_r", QVector3D(*r))
        pr.setUniformValue("u_g", QVector3D(*gg))
        pr.setUniformValue("u_b", QVector3D(*b))
        pr.setUniformValue("u_y0", float(y0))
        self.vbo.bind()
        pr.enableAttributeArray(0)
        pr.setAttributeBuffer(0, GL_FLOAT, 0, 2, 0)
        g.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        pr.disableAttributeArray(0)
        self.vbo.release()
        pr.release()
        g.glActiveTexture(GL_TEXTURE0)

    def _self_test(self):
        """Read back the centre of the picture and compare with the source frame."""
        f = self._frame
        if f is None or self._dead or not self._ok:
            return
        try:
            img = self.grabFramebuffer()
            c = self.v2w(self._vw / 2, self._vh / 2)
            dpr = img.devicePixelRatio() or 1.0
            px = img.pixelColor(int(c.x() * dpr), int(c.y() * dpr))
            got = np.array([px.blue(), px.green(), px.red()], float)
            ref = self._reference_pixel(f)
            if ref is None:
                return
            err = float(np.abs(got - ref).max())
            if err > 60:
                self._fail(f"self-test: centre pixel {got.astype(int).tolist()} vs expected {ref.astype(int).tolist()}")
            else:
                logger.info(f"GPU canvas self-test passed (max channel error {err:.0f})")
        except Exception as e:
            logger.debug(f"GPU self-test skipped: {e}")

    @staticmethod
    def _reference_pixel(f):
        def patch(img):
            h, w = img.shape[:2]
            return img[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3].reshape(-1, 3).mean(0)
        if getattr(f, "nv12", None) is not None:
            from AIO_QC.tools.frame_extractor.widgets.video_engine import nv12_small_bgr
            return patch(nv12_small_bgr(*f.nv12, max_w=640))
        src = f.bgr if f.bgr is not None else getattr(f, "small", None)
        return None if src is None else patch(src)

    # ------------------------------------------------------------------ vsync pacing
    def _on_swapped(self):
        if self.mpv_showing:
            self._mpv.report_swap()            # mpv's display-resample needs the vsync timing
        if self.playing and self.present_hook is not None and not self._dead:
            # the frame drawn now becomes visible at the next refresh
            self.present_hook(self.refresh_interval())
            self.update()
