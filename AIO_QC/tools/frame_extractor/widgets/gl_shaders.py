# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Shaders for the GPU video canvas (no Qt imports: also exercised by the test suite on Mesa).

The GPU does what a media player does:
  * NV12 frames (from the real-time pipeline) are converted YUV -> RGB in the fragment shader with the
    video's own matrix (BT.601 / 709 / 2020, limited / full range) - identical colours to the exact
    decoder (OpenCV follows the file's tag, BT.601 when untagged), so pausing never shifts colours.
  * BGR frames (exact decoder) are sampled directly.
  * Minification uses a multi-tap box filter over the pixel footprint (area-average quality, no aliasing
    when an 8K frame is shown in a 1080p window); magnification is bilinear, or nearest at >= 3x so
    individual pixels can be inspected.
Written for GLSL 1.10 (desktop, what Qt's compatibility context gives) and GLSL ES 1.00.
"""

VERTEX = """
attribute vec2 a_pos;
uniform vec4 u_rect;
varying vec2 v_tex;
void main() {
    v_tex = vec2(a_pos.x, 1.0 - a_pos.y);
    gl_Position = vec4(mix(u_rect.x, u_rect.z, a_pos.x), mix(u_rect.y, u_rect.w, a_pos.y), 0.0, 1.0);
}
"""

FRAGMENT = """
#ifdef GL_ES
precision highp float;
#endif
uniform sampler2D u_y;
uniform sampler2D u_uv;
uniform sampler2D u_rgb;
uniform int u_mode;          // 0 = NV12 (u_y luma, u_uv chroma as luminance-alpha), 1 = BGR bytes in u_rgb,
                             // 2 = BGRA upload (fast path) in u_rgb
uniform vec3 u_r;            // yuv -> rgb rows, applied to (Y - y0, U - 0.5, V - 0.5)
uniform vec3 u_g;
uniform vec3 u_b;
uniform float u_y0;
uniform vec2 u_footprint;    // texture-space size of one output pixel
uniform int u_taps;          // taps per axis for minification (1..4)
varying vec2 v_tex;

vec3 fetch(vec2 t) {
    if (u_mode == 0) {
        float y = texture2D(u_y, t).r;
        vec4 c = texture2D(u_uv, t);
        vec3 yuv = vec3(y - u_y0, c.r - 0.5, c.a - 0.5);
        return clamp(vec3(dot(u_r, yuv), dot(u_g, yuv), dot(u_b, yuv)), 0.0, 1.0);
    }
    if (u_mode == 2) {
        return texture2D(u_rgb, t).rgb;     // uploaded as BGRA: the driver already stored it as RGBA
    }
    return texture2D(u_rgb, t).bgr;
}

void main() {
    vec3 acc = vec3(0.0);
    if (u_taps <= 1) {
        acc = fetch(v_tex);
    } else {
        float n = float(u_taps);
        for (int i = 0; i < 4; i++) {
            if (i >= u_taps) break;
            for (int j = 0; j < 4; j++) {
                if (j >= u_taps) break;
                vec2 o = (vec2(float(i), float(j)) + 0.5) / n - 0.5;
                acc += fetch(v_tex + o * u_footprint);
            }
        }
        acc /= n * n;
    }
    gl_FragColor = vec4(acc, 1.0);
}
"""

# (Kr-based) rows for (Y', U, V) with limited-range luma scaling folded in
_MATRICES = {
    "601": ((1.0, 0.0, 1.4020), (1.0, -0.344136, -0.714136), (1.0, 1.7720, 0.0)),
    "709": ((1.0, 0.0, 1.5748), (1.0, -0.187324, -0.468124), (1.0, 1.8556, 0.0)),
    "2020": ((1.0, 0.0, 1.4746), (1.0, -0.164553, -0.571353), (1.0, 1.8814, 0.0)),
}


def yuv_uniforms(space="601", full_range=False):
    """(row_r, row_g, row_b, y0) for the fragment shader."""
    rows = _MATRICES.get(space, _MATRICES["601"])
    if full_range:
        ys, cs, y0 = 1.0, 1.0, 0.0
    else:
        ys, cs, y0 = 255.0 / 219.0, 255.0 / 224.0, 16.0 / 255.0
    out = [(r[0] * ys, r[1] * cs, r[2] * cs) for r in rows]
    return out[0], out[1], out[2], y0


def colorspace_from_probe(color_space, color_range):
    """ffprobe / PyAV tags -> (space, full_range), mirroring what the exact decoder (OpenCV) does."""
    cs = (color_space or "").lower()
    space = "709" if "709" in cs else "2020" if "2020" in cs else "601"
    return space, (color_range or "").lower() in ("pc", "jpeg", "full")


def taps_for(footprint_px):
    """Taps per axis for a minification factor (source pixels per output pixel)."""
    if footprint_px <= 1.25:
        return 1
    return int(min(4, max(2, round(footprint_px))))
