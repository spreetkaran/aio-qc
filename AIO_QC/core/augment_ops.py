# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image augmentation engine — ONE standalone file (NumPy + OpenCV only), so the same code runs in AIO QC (preview,
generating files) and inside a training environment (copied next to the training runner: augmentation on the fly).

    CATALOG / FAMILIES          every effect: key, title, family, parameters (ranges), default probability, tip
    default_recipe() · PRESETS  a recipe is plain JSON: mode, per-effect on / probability / ranges, label rules
    Augmenter(recipe).apply(image_bgr, shapes, seed) -> (image, shapes, log)
        shapes: [{"shape_type": rectangle|polygon|obb|circle|line|linestrip|point, "points": [[x, y], …], …}] in
        pixels — they follow every geometric effect (clipped to the image; dropped when too little stays visible)
    UltralyticsTransform(recipe)   the same recipe as one step of Ultralytics' training pipeline (labels dict)
    libraries()                 albumentations / imgaug found here (their extra effects switch on when installed)
    from_legacy(cfg)            a recipe from the older imgaug pipeline's aug_parameters.yaml

Built-in effects cover geometry, colour, weather / lighting, blur, noise, occlusion. Geometry is ONE warp per image
(every matrix effect composed first: one interpolation, exact label mapping); elastic / piecewise / lens distortion
are displacement fields whose label points are solved by fixed-point iteration. albumentations and imgaug add their
own effects (pixel-only: labels are unaffected) when the environment has them; nothing here needs them.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = 1
FAMILIES = [("geometry", "Geometry"), ("color", "Colour & exposure"), ("weather", "Weather & lighting"),
            ("blur", "Blur & focus"), ("noise", "Noise & camera artifacts"), ("occlusion", "Occlusion"),
            ("albumentations", "albumentations extras"), ("imgaug", "imgaug extras")]


def _p(name, label, lo, hi, dlo, dhi, neutral, dec=2, unit=""):
    return {"name": name, "label": label, "lo": lo, "hi": hi, "default": [dlo, dhi], "neutral": neutral,
            "dec": dec, "unit": unit}


def _op(key, title, family, kind, p, tip, params=(), needs=None, cls=None):
    return {"key": key, "title": title, "family": family, "kind": kind, "p": p, "tip": tip, "params": list(params),
            "needs": needs, "cls": cls}


CATALOG = [
    # ---------------------------------------------------------------- geometry (labels follow)
    _op("fliplr", "Mirror left–right", "geometry", "matrix", 0.5, "Not for text (plates, signs)"),
    _op("flipud", "Upside-down", "geometry", "matrix", 0.0, "Only for top-down views (aerial, conveyor)"),
    _op("rot90", "Quarter turns", "geometry", "matrix", 0.0, "0 / 90 / 180 / 270° — top-down views"),
    _op("rotate", "Rotate", "geometry", "matrix", 0.5, "Camera tilt",
        [_p("deg", "Angle", -180, 180, -15, 15, 0, 1, "°")]),
    _op("shear", "Shear", "geometry", "matrix", 0.3, "Slanted view",
        [_p("deg", "Angle", -45, 45, -10, 10, 0, 1, "°")]),
    _op("scale", "Zoom", "geometry", "matrix", 0.5, "Objects nearer / farther",
        [_p("factor", "Factor", 0.3, 2.0, 0.8, 1.2, 1.0, 2, "×")]),
    _op("translate", "Shift", "geometry", "matrix", 0.3, "Objects elsewhere in the frame",
        [_p("frac", "Share of the image", -0.5, 0.5, -0.1, 0.1, 0.0, 2)]),
    _op("perspective", "Perspective", "geometry", "matrix", 0.3, "Another viewpoint (corners pulled)",
        [_p("scale", "Corner move", 0.0, 0.3, 0.02, 0.08, 0.0, 3)]),
    _op("crop", "Random crop", "geometry", "matrix", 0.3, "A part of the image, enlarged to the full size",
        [_p("keep", "Part kept", 0.3, 1.0, 0.7, 0.95, 1.0, 2)]),
    _op("elastic", "Elastic warp", "geometry", "dense", 0.2, "Soft local stretching (cloth, soft objects)",
        [_p("alpha", "Strength", 0, 200, 30, 60, 0, 0, " px"), _p("sigma", "Smoothness", 3, 30, 8, 12, 12, 1)]),
    _op("piecewise", "Piecewise warp", "geometry", "dense", 0.2, "A grid of points moved a little (uneven lens, "
        "paper)", [_p("scale", "Grid move", 0.0, 0.1, 0.01, 0.03, 0.0, 3)]),
    _op("lens", "Lens distortion", "geometry", "dense", 0.2, "Barrel / pincushion (wide-angle cameras)",
        [_p("k", "Curvature", -0.5, 0.5, -0.2, 0.2, 0.0, 2)]),
    # ---------------------------------------------------------------- colour & exposure
    _op("brightness", "Brightness", "color", "pixel", 0.5, "Exposure: multiply and add",
        [_p("mul", "Multiply", 0.2, 2.5, 0.7, 1.3, 1.0, 2, "×"), _p("add", "Add", -80, 80, -10, 10, 0, 0)]),
    _op("contrast", "Contrast", "color", "pixel", 0.5, "Flat … harsh",
        [_p("factor", "Factor", 0.2, 3.0, 0.75, 1.35, 1.0, 2, "×")]),
    _op("gamma", "Gamma", "color", "pixel", 0.3, "Shadows lifted / crushed",
        [_p("gamma", "Gamma", 0.3, 3.0, 0.7, 1.5, 1.0, 2)]),
    _op("hue", "Hue", "color", "pixel", 0.3, "Colour shift — keep small when colour tells classes apart",
        [_p("deg", "Shift", -90, 90, -15, 15, 0, 0, "°")]),
    _op("saturation", "Saturation", "color", "pixel", 0.4, "Washed out … vivid",
        [_p("factor", "Factor", 0.0, 3.0, 0.6, 1.4, 1.0, 2, "×")]),
    _op("gray", "Grayscale", "color", "pixel", 0.1, "Towards black and white (IR / mono cameras)",
        [_p("alpha", "Amount", 0.0, 1.0, 0.3, 1.0, 0.0, 2)]),
    _op("temperature", "Colour temperature", "color", "pixel", 0.3, "Warm (tungsten) … cold (shade) light",
        [_p("kelvin", "Kelvin", 1500, 12000, 3000, 9000, 6600, 0, " K")]),
    _op("rgb_shift", "RGB shift", "color", "pixel", 0.2, "Each colour channel moved a little (white balance error)",
        [_p("amount", "Amount", 0, 80, 5, 25, 0, 0)]),
    _op("clahe", "CLAHE", "color", "pixel", 0.2, "Local contrast (as cameras' HDR / WDR)",
        [_p("clip", "Clip limit", 1, 16, 2, 8, 1, 1)]),
    _op("equalize", "Equalize", "color", "pixel", 0.05, "Histogram equalised"),
    _op("posterize", "Posterize", "color", "pixel", 0.05, "Fewer colour levels",
        [_p("bits", "Bits kept", 1, 7, 3, 6, 8, 0)]),
    _op("solarize", "Solarize", "color", "pixel", 0.0, "Bright areas inverted",
        [_p("threshold", "Threshold", 0, 255, 100, 200, 255, 0)]),
    _op("invert", "Invert", "color", "pixel", 0.0, "Negative (thermal / X-ray style polarity)"),
    _op("channel_shuffle", "Channel shuffle", "color", "pixel", 0.0, "RGB channels in another order"),
    _op("kmeans", "Colour quantization", "color", "pixel", 0.05, "Only N colours (k-means), like a cheap encoder",
        [_p("colors", "Colours", 2, 64, 8, 32, 64, 0)]),
    # ---------------------------------------------------------------- weather & lighting
    _op("fog", "Fog", "weather", "pixel", 0.15, "Haze: washed-out, low contrast",
        [_p("density", "Density", 0.0, 1.0, 0.2, 0.6, 0.0, 2)]),
    _op("rain", "Rain", "weather", "pixel", 0.15, "Streaks, a darker and softer image",
        [_p("amount", "Amount", 0.0, 1.0, 0.2, 0.6, 0.0, 2)]),
    _op("snow", "Snow", "weather", "pixel", 0.1, "Flakes and a brighter scene",
        [_p("amount", "Amount", 0.0, 1.0, 0.2, 0.5, 0.0, 2)]),
    _op("glare", "Glare (bright ground)", "weather", "pixel", 0.1, "Darker areas lifted towards white: snow cover, "
        "wet ground, over-exposure", [_p("threshold", "Up to lightness", 50, 220, 100, 150, 50, 0),
                                      _p("mul", "Multiply", 1.0, 3.0, 1.2, 2.0, 1.0, 2, "×")]),
    _op("sun_flare", "Sun flare", "weather", "pixel", 0.1, "A bright spot with rings",
        [_p("strength", "Strength", 0.0, 1.0, 0.3, 0.7, 0.0, 2)]),
    _op("shadow", "Shadow", "weather", "pixel", 0.2, "A darker region with a soft edge (buildings, trees)",
        [_p("darkness", "Darkness", 0.0, 0.9, 0.3, 0.6, 0.0, 2)]),
    _op("vignette", "Vignette", "weather", "pixel", 0.1, "Darker corners",
        [_p("amount", "Amount", 0.0, 1.0, 0.2, 0.6, 0.0, 2)]),
    _op("low_light", "Low light", "weather", "pixel", 0.15, "Dark and noisy: dusk, night, indoors",
        [_p("darkness", "Darkness", 0.0, 0.95, 0.4, 0.75, 0.0, 2)]),
    # ---------------------------------------------------------------- blur & focus
    _op("gauss_blur", "Gaussian blur", "blur", "pixel", 0.2, "Out of focus / soft lens",
        [_p("sigma", "Sigma", 0.0, 10.0, 0.5, 2.5, 0.0, 1, " px")]),
    _op("motion_blur", "Motion blur", "blur", "pixel", 0.2, "A moving camera or object",
        [_p("kernel", "Length", 3, 45, 5, 15, 3, 0, " px"), _p("angle", "Direction", 0, 360, 0, 360, 0, 0, "°")]),
    _op("defocus", "Defocus", "blur", "pixel", 0.1, "Lens out of focus (disc blur)",
        [_p("radius", "Radius", 1, 15, 2, 5, 1, 0, " px")]),
    _op("ghost", "Ghosting", "blur", "pixel", 0.0, "The image blended with a slightly turned copy (double "
        "exposure, reflections in glass)", [_p("alpha", "Amount", 0.0, 1.0, 0.2, 0.5, 0.0, 2),
                                            _p("rotate", "Turn", -30, 30, -10, 10, 0, 1, "°")]),
    # ---------------------------------------------------------------- noise & camera artifacts
    _op("noise", "Sensor noise", "noise", "pixel", 0.3, "Gaussian, Laplace, Poisson or speckle — one at random",
        [_p("scale", "Strength", 0, 80, 5, 25, 0, 0)]),
    _op("shot_noise", "Shot noise", "noise", "pixel", 0.1, "Grain that grows with brightness (few photons)",
        [_p("severity", "Severity", 1, 5, 1, 3, 1, 0)]),
    _op("salt_pepper", "Salt & pepper", "noise", "pixel", 0.1, "Dead / hot pixels",
        [_p("amount", "Share of pixels", 0.0, 0.3, 0.01, 0.05, 0.0, 3)]),
    _op("jpeg", "JPEG compression", "noise", "pixel", 0.3, "Blocky artifacts of cheap streams",
        [_p("quality", "Quality", 5, 95, 20, 60, 95, 0)]),
    _op("pixelate", "Pixelate", "noise", "pixel", 0.1, "Big square pixels (far objects, digital zoom)",
        [_p("block", "Block", 2, 32, 2, 6, 1, 0, " px")]),
    _op("downscale", "Low resolution", "noise", "pixel", 0.2, "Shrunk and enlarged again",
        [_p("factor", "Scale", 0.1, 1.0, 0.3, 0.7, 1.0, 2, "×")]),
    # ---------------------------------------------------------------- occlusion
    _op("cutout", "Cutout (coarse dropout)", "occlusion", "pixel", 0.3, "Rectangles hidden: teaches partly hidden "
        "objects", [_p("holes", "Rectangles", 1, 30, 2, 8, 1, 0), _p("size", "Size", 0.01, 0.5, 0.03, 0.12, 0.01, 2)]),
    _op("pixel_dropout", "Pixel dropout", "occlusion", "pixel", 0.1, "Single pixels blacked out",
        [_p("share", "Share of pixels", 0.0, 0.5, 0.01, 0.08, 0.0, 3)]),
    _op("channel_dropout", "Channel dropout", "occlusion", "pixel", 0.0, "One colour channel missing"),
    # ---------------------------------------------------------------- albumentations extras (pixel only)
    _op("a_iso", "ISO noise", "albumentations", "lib", 0.2, "Camera sensor noise at high ISO", needs="albumentations",
        cls="ISONoise"),
    _op("a_rain", "Rain (albumentations)", "albumentations", "lib", 0.0, "Rain drops and streaks",
        needs="albumentations", cls="RandomRain"),
    _op("a_snow", "Snow (albumentations)", "albumentations", "lib", 0.0, "Snow cover", needs="albumentations",
        cls="RandomSnow"),
    _op("a_fog", "Fog (albumentations)", "albumentations", "lib", 0.0, "Patchy fog", needs="albumentations",
        cls="RandomFog"),
    _op("a_flare", "Sun flare (albumentations)", "albumentations", "lib", 0.0, "Lens flare with rings",
        needs="albumentations", cls="RandomSunFlare"),
    _op("a_shadow", "Shadows (albumentations)", "albumentations", "lib", 0.0, "Polygon shadows",
        needs="albumentations", cls="RandomShadow"),
    _op("a_plasma_shadow", "Plasma shadow", "albumentations", "lib", 0.0, "Soft, cloud-like light and shade",
        needs="albumentations", cls="PlasmaShadow"),
    _op("a_illumination", "Uneven illumination", "albumentations", "lib", 0.1, "A light gradient across the image",
        needs="albumentations", cls="Illumination"),
    _op("a_spatter", "Spatter on the lens", "albumentations", "lib", 0.0, "Rain or mud drops on the lens",
        needs="albumentations", cls="Spatter"),
    _op("a_gravel", "Gravel", "albumentations", "lib", 0.0, "Stones / debris on the ground",
        needs="albumentations", cls="RandomGravel"),
    _op("a_glass", "Glass blur", "albumentations", "lib", 0.0, "Seen through frosted glass (slow)",
        needs="albumentations", cls="GlassBlur"),
    _op("a_zoom_blur", "Zoom blur", "albumentations", "lib", 0.0, "Moving towards the scene",
        needs="albumentations", cls="ZoomBlur"),
    _op("a_chromatic", "Chromatic aberration", "albumentations", "lib", 0.1, "Colour fringes of cheap lenses",
        needs="albumentations", cls="ChromaticAberration"),
    _op("a_planckian", "Planckian jitter", "albumentations", "lib", 0.0, "Physically based light colour",
        needs="albumentations", cls="PlanckianJitter"),
    _op("a_tone", "Tone curve", "albumentations", "lib", 0.1, "A random camera tone curve", needs="albumentations",
        cls="RandomToneCurve"),
    _op("a_sharpen", "Sharpen", "albumentations", "lib", 0.1, "In-camera sharpening", needs="albumentations",
        cls="Sharpen"),
    _op("a_ringing", "Ringing", "albumentations", "lib", 0.0, "Halos next to edges (over-sharpened video)",
        needs="albumentations", cls="RingingOvershoot"),
    _op("a_emboss", "Emboss", "albumentations", "lib", 0.0, "Relief look", needs="albumentations", cls="Emboss"),
    # ---------------------------------------------------------------- imgaug extras (pixel only)
    _op("i_clouds", "Clouds", "imgaug", "lib", 0.0, "Cloud layers over the image", needs="imgaug", cls="Clouds"),
    _op("i_fog", "Fog (imgaug)", "imgaug", "lib", 0.0, "Dense, uneven fog", needs="imgaug", cls="Fog"),
    _op("i_snowflakes", "Snowflakes", "imgaug", "lib", 0.0, "Falling snow", needs="imgaug", cls="Snowflakes"),
    _op("i_rain", "Rain (imgaug)", "imgaug", "lib", 0.0, "Falling rain", needs="imgaug", cls="Rain"),
    _op("i_cartoon", "Cartoon", "imgaug", "lib", 0.0, "Flat colours and edges (synthetic-looking data)",
        needs="imgaug", cls="Cartoon"),
    _op("i_superpixels", "Superpixels", "imgaug", "lib", 0.0, "Regions replaced by their average colour",
        needs="imgaug", cls="Superpixels"),
    _op("i_canny", "Edge overlay", "imgaug", "lib", 0.0, "Edges drawn over the image", needs="imgaug", cls="Canny"),
    _op("i_emboss", "Emboss (imgaug)", "imgaug", "lib", 0.0, "Relief look", needs="imgaug", cls="Emboss"),
]
OPS = {o["key"]: o for o in CATALOG}
_FAMILY_ORDER = {f: i for i, (f, _t) in enumerate(FAMILIES)}
_ORDER = {o["key"]: (0 if o["kind"] == "matrix" else 1 if o["kind"] == "dense" else 2, i)
          for i, o in enumerate(CATALOG)}
BORDERS = {"gray": "Grey", "black": "Black", "edge": "Stretched edge", "reflect": "Mirrored"}


# ====================================================================================================== recipes
def default_recipe() -> dict:
    """Everything in the catalog with its default ranges; a balanced set switched on."""
    on = {"fliplr", "rotate", "scale", "brightness", "contrast", "hue", "saturation", "gauss_blur", "noise", "jpeg",
          "shadow", "cutout"}
    return {"version": VERSION, "name": "Balanced", "mode": "some", "min": 1, "max": 3, "strength": 1.0,
            "border": "gray", "min_visible": 0.3, "min_size": 3,
            "ops": {o["key"]: _op_default(o, o["key"] in on) for o in CATALOG}}


def _op_default(o: dict, on: bool = False) -> dict:
    d = {"on": bool(on), "p": o["p"] if o["p"] > 0 else 0.3}
    for p in o["params"]:
        d[p["name"]] = list(p["default"])
    return d


def _with(on: dict, **top) -> dict:
    r = default_recipe()
    for k in r["ops"]:
        r["ops"][k]["on"] = False
    for k, v in on.items():
        r["ops"][k]["on"] = True
        r["ops"][k].update(v)
    r.update(top)
    return r


def presets() -> dict:
    """name -> (recipe, what it is for)."""
    return {
        "Light": (_with({"fliplr": {}, "brightness": {"mul": [0.85, 1.15], "add": [-5, 5]},
                         "contrast": {"factor": [0.85, 1.15]}, "saturation": {"factor": [0.8, 1.2]},
                         "scale": {"factor": [0.9, 1.1]}, "jpeg": {"quality": [50, 85]}},
                        name="Light", mode="some", min=1, max=2),
                  "Small, safe changes — large datasets, fine-tuning"),
        "Balanced": (default_recipe(), "A good start for most detection datasets"),
        "Strong": (_with({"fliplr": {}, "rotate": {"deg": [-25, 25]}, "shear": {}, "scale": {"factor": [0.6, 1.4]},
                          "translate": {}, "perspective": {}, "crop": {}, "brightness": {"mul": [0.5, 1.5]},
                          "contrast": {"factor": [0.6, 1.6]}, "gamma": {}, "hue": {"deg": [-25, 25]},
                          "saturation": {"factor": [0.4, 1.6]}, "gray": {}, "shadow": {}, "fog": {},
                          "gauss_blur": {}, "motion_blur": {}, "noise": {"scale": [5, 40]}, "jpeg": {},
                          "downscale": {}, "cutout": {}},
                         name="Strong", mode="some", min=2, max=4),
                   "Much variety — small datasets that over-fit"),
        "Outdoor camera & weather": (_with({"brightness": {}, "contrast": {}, "temperature": {}, "shadow": {},
                                            "fog": {"p": 0.2}, "rain": {"p": 0.2}, "snow": {"p": 0.1},
                                            "sun_flare": {"p": 0.15}, "glare": {"p": 0.1}, "low_light": {"p": 0.2},
                                            "motion_blur": {}, "gauss_blur": {}, "noise": {}, "jpeg": {},
                                            "scale": {}, "rotate": {"deg": [-5, 5]}},
                                           name="Outdoor camera & weather", mode="some", min=1, max=3),
                                     "Fixed and moving cameras outdoors: light, weather, motion"),
        "Night & low light": (_with({"low_light": {"p": 0.7}, "gamma": {"gamma": [1.2, 2.5]},
                                     "noise": {"scale": [10, 45]}, "shot_noise": {"p": 0.3},
                                     "motion_blur": {}, "gray": {"p": 0.2}, "sun_flare": {"p": 0.15},
                                     "temperature": {"kelvin": [2500, 5000]}, "jpeg": {}},
                                    name="Night & low light", mode="some", min=1, max=3),
                              "Dusk, night and indoor scenes from day-time data"),
        "Poor camera": (_with({"gauss_blur": {}, "motion_blur": {}, "defocus": {}, "jpeg": {"quality": [8, 40]},
                               "pixelate": {}, "downscale": {}, "noise": {}, "salt_pepper": {},
                               "rgb_shift": {}, "kmeans": {"p": 0.1}, "posterize": {"p": 0.1}},
                              name="Poor camera", mode="some", min=1, max=3),
                        "Cheap sensors and streams: blur, compression, low resolution"),
        "Text & plates": (_with({"rotate": {"deg": [-8, 8]}, "shear": {"deg": [-12, 12]},
                                 "perspective": {"scale": [0.01, 0.05]}, "scale": {"factor": [0.8, 1.25]},
                                 "brightness": {}, "contrast": {}, "gamma": {}, "shadow": {}, "glare": {"p": 0.1},
                                 "motion_blur": {"kernel": [3, 11]}, "gauss_blur": {"sigma": [0.3, 1.8]},
                                 "noise": {}, "jpeg": {}, "downscale": {"factor": [0.4, 0.8]},
                                 "low_light": {"p": 0.15}},
                                name="Text & plates", mode="some", min=1, max=3),
                          "Readable text stays readable: no flips, mild geometry, real camera faults"),
        "Classic (one effect per image)": (from_legacy({}), "The older imgaug pipeline: one family, then one of "
                                                            "its effects, per image"),
    }


def clean(recipe: dict | None) -> dict:
    """A valid recipe from anything stored / imported: unknown keys dropped, ranges ordered and clamped."""
    base = default_recipe()
    r = dict(recipe or {})
    out = {"version": VERSION, "name": str(r.get("name") or "Custom")[:80]}
    out["mode"] = r.get("mode") if r.get("mode") in ("one", "some", "each") else base["mode"]
    out["min"] = int(min(max(int(r.get("min", 1) or 0), 0), 12))
    out["max"] = int(min(max(int(r.get("max", 3) or 1), max(1, out["min"])), 12))
    out["strength"] = float(min(max(float(r.get("strength", 1.0) or 1.0), 0.1), 2.0))
    out["border"] = r.get("border") if r.get("border") in BORDERS else "gray"
    out["min_visible"] = float(min(max(float(r.get("min_visible", 0.3)), 0.0), 1.0))
    out["min_size"] = float(min(max(float(r.get("min_size", 3)), 0.0), 1000.0))
    ops = r.get("ops") or {}
    out["ops"] = {}
    for o in CATALOG:
        d = _op_default(o)
        s = ops.get(o["key"]) or {}
        d["on"] = bool(s.get("on", False))
        try:
            d["p"] = float(min(max(float(s.get("p", d["p"])), 0.0), 1.0))
        except (TypeError, ValueError):
            pass
        for p in o["params"]:
            v = s.get(p["name"], d[p["name"]])
            try:
                a, b = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), float(v))
            except (TypeError, ValueError, IndexError):
                a, b = p["default"]
            a, b = min(a, b), max(a, b)
            d[p["name"]] = [min(max(a, p["lo"]), p["hi"]), min(max(b, p["lo"]), p["hi"])]
        out["ops"][o["key"]] = d
    return out


def from_legacy(cfg: dict) -> dict:
    """The older pipeline's aug_parameters.yaml (apply_geo_transformation, apply_color_transformation,
    apply_noise_artifact, apply_lightning_effects with apply / prob / ranges) as a recipe. An empty config gives
    that pipeline's stock settings."""
    def rng(d, k, default):
        v = (d or {}).get(k, default)
        return list(v) if isinstance(v, (list, tuple)) and len(v) == 2 else [v, v]
    geo = cfg.get("apply_geo_transformation") or {}
    col = cfg.get("apply_color_transformation") or {}
    noi = cfg.get("apply_noise_artifact") or {}
    lit = cfg.get("apply_lightning_effects") or {}
    stock = not cfg
    on: dict = {}

    def add(key, src, name, **vals):
        e = src.get(name) or {}
        if stock or (src.get("enabled", True) and e.get("apply", False)):
            on[key] = dict(vals, p=float(e.get("prob", 1.0)))
    g = lambda n: geo.get(n) or {}     # noqa: E731
    c = lambda n: col.get(n) or {}     # noqa: E731
    n_ = lambda n: noi.get(n) or {}    # noqa: E731
    l_ = lambda n: lit.get(n) or {}    # noqa: E731
    add("rotate", geo, "rotate", deg=rng(g("rotate"), "range", [-30, 30]))
    add("shear", geo, "shear", deg=[max(-45, rng(g("shear"), "range", [-50, 50])[0]),
                                    min(45, rng(g("shear"), "range", [-50, 50])[1])])
    add("perspective", geo, "perspectiveTransform", scale=rng(g("perspectiveTransform"), "scale", [0.01, 0.15]))
    add("piecewise", geo, "pieceWiseAffine", scale=rng(g("pieceWiseAffine"), "scale", [0.02, 0.04]))
    add("elastic", geo, "ElasticTransformation", alpha=rng(g("ElasticTransformation"), "scale", [40, 100]),
        sigma=rng(g("ElasticTransformation"), "sigma", [8, 15]))
    add("fliplr", geo, "fliplr")
    add("flipud", geo, "flipud")
    add("contrast", col, "linearContrast", factor=rng(c("linearContrast"), "range", [0.75, 1.35]))
    add("hue", col, "hue", deg=rng(c("hue"), "range", [-30, 30]))
    add("brightness", col, "addBrightness", mul=rng(c("addBrightness"), "mul", [0.5, 1.5]),
        add=rng(c("addBrightness"), "add", [-10, 10]))
    add("invert", col, "invert")
    add("gray", col, "GrayScale", alpha=rng(c("GrayScale"), "alpha", [0.3, 1.0]))
    add("clahe", col, "clahe", clip=rng(c("clahe"), "clip_limit", [2, 8]))
    add("pixelate", col, "pixelate", block=[2 * v for v in rng(c("pixelate"), "scale", [1, 5])])
    add("kmeans", col, "KMeansColor", colors=[min(64, v) for v in rng(c("KMeansColor"), "scale", [50, 100])])
    add("salt_pepper", noi, "salt_pepper", amount=rng(n_("salt_pepper"), "scale", [0.01, 0.1]))
    add("noise", noi, "noise", scale=rng(n_("noise"), "scale", [0, 40]))
    add("shot_noise", noi, "shotnoise", severity=rng(n_("shotnoise"), "scale", [1, 5]))
    add("cutout", noi, "CoarseDropout")
    add("gauss_blur", noi, "gausBlur", sigma=rng(n_("gausBlur"), "scale", [0.5, 5.0]))
    add("fog", noi, "fog", density=[v / 5.0 for v in rng(n_("fog"), "scale", [1, 4])])
    add("channel_dropout", noi, "dropout2d")
    add("jpeg", noi, "JpegCompression", quality=sorted(100 - v for v in rng(n_("JpegCompression"), "compression",
                                                                            [10, 50])))
    add("glare", lit, "flare", threshold=rng(l_("flare"), "threshold", [100, 150]),
        mul=rng(l_("flare"), "multiplier", [1.2, 2.0]))
    add("motion_blur", lit, "motionBlur", kernel=rng(l_("motionBlur"), "kernel", [8, 15]),
        angle=rng(l_("motionBlur"), "angle", [150, 360]))
    if stock or (lit.get("enabled", True) and l_("contrastNormalization").get("apply")):
        on.setdefault("contrast", {"factor": rng(l_("contrastNormalization"), "alpha", [0.5, 2.0]),
                                   "p": float(l_("contrastNormalization").get("prob", 1.0))})
    add("temperature", lit, "colorTemperature", kelvin=rng(l_("colorTemperature"), "temperature", [2000, 5000]))
    add("ghost", lit, "blend_alpha", alpha=rng(l_("blend_alpha"), "alpha", [0.0, 1.0]),
        rotate=rng(l_("blend_alpha"), "rotate", [-10, 10]))
    if stock:
        on["invert"]["p"] = 0.2
        for k in ("fliplr", "flipud"):
            on[k]["p"] = 0.3
    return clean(_with(on, name="Classic (one effect per image)", mode="one", min=1, max=1))


# ====================================================================================================== libraries
_LIBS: dict = {}


def _numpy2_shim():
    """imgaug 0.4 predates NumPy 2: the names it still uses, put back (harmless aliases)."""
    if not hasattr(np, "sctypes"):
        np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64],
                      "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                      "float": [np.float16, np.float32, np.float64], "complex": [np.complex64, np.complex128],
                      "others": [bool, object, bytes, str, np.void]}
    for name, val in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object),
                      ("str", str), ("float_", np.float64), ("complex_", np.complex128), ("product", np.prod),
                      ("cumproduct", np.cumprod), ("sometrue", np.any), ("alltrue", np.all), ("in1d", np.isin),
                      ("row_stack", np.vstack)):
        if not hasattr(np, name):
            try:
                setattr(np, name, val)
            except Exception:
                pass


def _lib(name: str):
    """(module or None, version or the reason it is missing) — imported once."""
    if name not in _LIBS:
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "imgaug":
                    _numpy2_shim()
                    import imgaug.augmenters as mod
                    import imgaug
                    ver = imgaug.__version__
                else:
                    import os
                    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")      # no version check over the network
                    import albumentations as mod
                    ver = mod.__version__
            _LIBS[name] = (mod, ver)
        except Exception as exc:
            _LIBS[name] = (None, f"{type(exc).__name__}: {str(exc)[:160]}")
    return _LIBS[name]


_BROKEN: dict = {}
_INSTANCES: dict = {}


def libraries(test: bool = True) -> dict:
    """{"albumentations": {"version" | "error", "effects": [...], "broken": {key: why}}, "imgaug": {…}} — with
    `test`, every library effect is tried once on a small image, so one that does not work here is never offered."""
    out = {}
    probe = np.full((48, 64, 3), 120, np.uint8)
    probe[10:30, 20:40] = (30, 160, 220)
    for name in ("albumentations", "imgaug"):
        mod, ver = _lib(name)
        keys = [o["key"] for o in CATALOG if o["needs"] == name]
        info = {"version": ver if mod is not None else None, "error": None if mod is not None else ver,
                "effects": [], "broken": {}}
        if mod is not None:
            for k in keys:
                if test and k not in _BROKEN:
                    try:
                        r = _lib_apply(k, probe, np.random.default_rng(0))
                        _BROKEN[k] = None if r.shape == probe.shape and r.dtype == np.uint8 else "wrong output"
                    except Exception as exc:
                        _BROKEN[k] = f"{type(exc).__name__}: {str(exc)[:100]}"
                if _BROKEN.get(k):
                    info["broken"][k] = _BROKEN[k]
                else:
                    info["effects"].append(k)
        out[name] = info
    return out


def available(key: str) -> bool:
    o = OPS.get(key)
    if o is None:
        return False
    if not o["needs"]:
        return True
    return _lib(o["needs"])[0] is not None and not _BROKEN.get(key)


def _lib_apply(key, img, rng):
    o = OPS[key]
    mod, _ver = _lib(o["needs"])
    seed = int(rng.integers(0, 2**31 - 1))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = _INSTANCES.get(key)
    if o["needs"] == "albumentations":
        if t is None:
            t = _INSTANCES[key] = getattr(mod, o["cls"])(p=1.0)
        if hasattr(t, "set_random_seed"):
            t.set_random_seed(seed)
        res = t(image=rgb)["image"]
    else:
        if t is None:
            kw = {"Superpixels": {"p_replace": (0.3, 0.8), "n_segments": (40, 160)}, "Canny": {"alpha": (0.2, 0.6)},
                  "Emboss": {"alpha": (0.2, 0.8), "strength": (0.5, 1.5)}}.get(o["cls"], {})
            t = _INSTANCES[key] = getattr(mod, o["cls"])(**kw)
        t.seed_(seed)
        res = t(image=rgb)
    res = np.ascontiguousarray(res)
    if res.dtype != np.uint8:
        res = np.clip(res * (255.0 if res.dtype.kind == "f" and res.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


# ====================================================================================================== geometry
def _T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def _about_centre(A, w, h):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return _T(cx, cy) @ A @ _T(-cx, -cy)


def _m_fliplr(rng, w, h):
    return np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64)


def _m_flipud(rng, w, h):
    return np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], np.float64)


def _m_rotate(rng, w, h, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return _about_centre(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64), w, h)


def _m_shear(rng, w, h, deg):
    t = math.tan(math.radians(deg))
    A = np.array([[1, t, 0], [0, 1, 0], [0, 0, 1]], np.float64) if rng.random() < 0.5 else \
        np.array([[1, 0, 0], [t, 1, 0], [0, 0, 1]], np.float64)
    return _about_centre(A, w, h)


def _m_scale(rng, w, h, factor):
    f = max(factor, 0.05)
    return _about_centre(np.diag([f, f, 1.0]), w, h)


def _m_translate(rng, w, h, frac):
    other = float(rng.uniform(-abs(frac), abs(frac)))
    return _T(frac * w, other * h)


def _m_perspective(rng, w, h, scale):
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    jit = rng.uniform(-scale, scale, (4, 2)).astype(np.float32) * np.array([w, h], np.float32)
    return cv2.getPerspectiveTransform(src, src + jit).astype(np.float64)


def _m_crop(rng, w, h, keep):
    f = min(max(keep, 0.05), 1.0)
    cw, ch = f * w, f * h
    x0, y0 = float(rng.uniform(0, w - cw)), float(rng.uniform(0, h - ch))
    return np.diag([1 / f, 1 / f, 1.0]) @ _T(-x0, -y0)


def _d_elastic(rng, w, h, alpha, sigma):
    """Smooth random displacement: built at ¼ size (fast on large images), enlarged."""
    k = 4 if min(w, h) >= 256 else 1
    sw, sh = max(2, w // k), max(2, h // k)
    f = rng.uniform(-1, 1, (2, sh, sw)).astype(np.float32)
    s = max(sigma / k, 0.5)
    out = []
    for a in f:
        a = cv2.GaussianBlur(a, (0, 0), s)
        m = float(np.abs(a).max()) or 1.0
        out.append(cv2.resize(a / m * alpha * 0.25, (w, h), interpolation=cv2.INTER_CUBIC))
    return out[0], out[1]


def _d_piecewise(rng, w, h, scale):
    g = int(rng.integers(4, 8))
    f = rng.normal(0, 1, (2, g, g)).astype(np.float32)
    f[:, 0, :] = f[:, -1, :] = 0
    f[:, :, 0] = f[:, :, -1] = 0                          # the border stays where it is
    return (cv2.resize(f[0] * scale * w, (w, h), interpolation=cv2.INTER_CUBIC),
            cv2.resize(f[1] * scale * h, (w, h), interpolation=cv2.INTER_CUBIC))


def _d_lens(rng, w, h, k):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    xs = (np.arange(w, dtype=np.float32) - cx)[None, :]
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r2 = (xs * xs + ys * ys) / float(cx * cx + cy * cy + 1e-6)
    g = ((1 + k * r2) / (1 + k) - 1).astype(np.float32)   # corners stay at the corners
    return xs * g, ys * g


def _sample(field, P):
    h, w = field.shape
    x = np.clip(P[:, 0], 0, w - 1.001)
    y = np.clip(P[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    fx, fy = x - x0, y - y0
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)


def _forward_dense(P, dx, dy):
    """Where the points go under a backward map (out(x) = in(x + d(x))): q + d(q) = p, by fixed-point steps."""
    q = P.astype(np.float64).copy()
    for _ in range(8):
        q = P - np.stack([_sample(dx, q), _sample(dy, q)], 1)
    return q


def _clip_polygon(P, w, h):
    """Sutherland–Hodgman against the image rectangle."""
    pts = [tuple(p) for p in P]
    for axis, bound, keep_less in ((0, 0.0, False), (0, float(w), True), (1, 0.0, False), (1, float(h), True)):
        if not pts:
            break
        out = []
        prev = pts[-1]
        pin = prev[axis] <= bound if keep_less else prev[axis] >= bound
        for cur in pts:
            cin = cur[axis] <= bound if keep_less else cur[axis] >= bound
            if cin != pin:
                t = (bound - prev[axis]) / ((cur[axis] - prev[axis]) or 1e-12)
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
            prev, pin = cur, cin
        pts = out
    return np.array(pts, np.float64).reshape(-1, 2)


def _area(P):
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _densify(P, step=24.0):
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        k = int(max(1, math.ceil(float(np.hypot(*(b - a))) / step)))
        out.extend(a + (b - a) * (j / k) for j in range(k))
    return np.array(out, np.float64)


def _box_ring(x1, y1, x2, y2, n=6):
    t = np.linspace(0, 1, n, endpoint=False)
    return np.concatenate([np.stack([x1 + (x2 - x1) * t, np.full(n, y1)], 1),
                           np.stack([np.full(n, x2), y1 + (y2 - y1) * t], 1),
                           np.stack([x2 - (x2 - x1) * t, np.full(n, y2)], 1),
                           np.stack([np.full(n, x1), y2 - (y2 - y1) * t], 1)])


def map_shapes(shapes, fwd, w, h, min_visible=0.3, min_size=3.0, dense=False):
    """Shapes (pixels) through a forward point mapper, into a w × h image. -> the shapes that are still there."""
    out = []
    for s in shapes or []:
        st = s.get("shape_type") or "polygon"
        try:
            P = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
        except ValueError:
            continue
        if not len(P):
            continue
        new = None
        if st == "rectangle" and len(P) >= 2:
            x1, y1, x2, y2 = min(P[0][0], P[1][0]), min(P[0][1], P[1][1]), max(P[0][0], P[1][0]), max(P[0][1], P[1][1])
            Q = fwd(_box_ring(x1, y1, x2, y2))
            fx1, fy1, fx2, fy2 = Q[:, 0].min(), Q[:, 1].min(), Q[:, 0].max(), Q[:, 1].max()
            cx1, cy1, cx2, cy2 = max(fx1, 0), max(fy1, 0), min(fx2, w), min(fy2, h)
            full = max(fx2 - fx1, 1e-9) * max(fy2 - fy1, 1e-9)
            if cx2 - cx1 >= min_size and cy2 - cy1 >= min_size and (cx2 - cx1) * (cy2 - cy1) / full >= min_visible:
                new = [[cx1, cy1], [cx2, cy2]]
        elif st == "polygon" and len(P) >= 3:
            Q = fwd(_densify(P) if dense else P)
            full = _area(Q)
            C = _clip_polygon(Q, w, h)
            if len(C) >= 3 and full > 0 and _area(C) / full >= min_visible and \
                    np.ptp(C[:, 0]) >= min_size and np.ptp(C[:, 1]) >= min_size:
                new = C.tolist()
        elif st == "obb" and len(P) == 4:
            Q = fwd(P)
            full = _area(Q)
            C = _clip_polygon(Q, w, h)
            if full > 0 and len(C) >= 3 and _area(C) / full >= min_visible:
                box = cv2.boxPoints(cv2.minAreaRect(Q.astype(np.float32)))      # a rectangle again after shear
                i = int(np.argmin(((box - Q[0]) ** 2).sum(1)))                  # the same starting corner
                box = np.roll(box, -i, axis=0)
                if ((box[1] - Q[1]) ** 2).sum() > ((box[-1] - Q[1]) ** 2).sum():
                    box = np.concatenate([box[:1], box[:0:-1]])                # … and the same direction
                new = box.astype(np.float64).tolist()
        elif st == "circle" and len(P) >= 2:
            Q = fwd(P[:2])
            if 0 <= Q[0][0] <= w and 0 <= Q[0][1] <= h:
                new = Q.tolist()
        elif st == "point":
            Q = fwd(P[:1])
            if 0 <= Q[0][0] <= w and 0 <= Q[0][1] <= h:
                new = Q.tolist()
        else:                                              # line, linestrip, anything else with points
            Q = fwd(P)
            inside = (Q[:, 0] >= 0) & (Q[:, 0] <= w) & (Q[:, 1] >= 0) & (Q[:, 1] <= h)
            if inside.any():
                Q[:, 0] = np.clip(Q[:, 0], 0, w)
                Q[:, 1] = np.clip(Q[:, 1], 0, h)
                new = Q.tolist()
        if new is not None:
            s2 = dict(s)
            s2["points"] = [[float(x), float(y)] for x, y in new]
            out.append(s2)
    return out


# ====================================================================================================== pixel effects
def _cv_seed(rng):
    cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))


def _add_noise16(img, noise16):
    return np.clip(img.astype(np.int16) + noise16, 0, 255).astype(np.uint8)


def _x_brightness(img, rng, mul, add):
    return cv2.convertScaleAbs(img, alpha=float(mul), beta=float(add))


def _x_contrast(img, rng, factor):
    return cv2.convertScaleAbs(img, alpha=float(factor), beta=127.5 * (1 - float(factor)))


def _x_gamma(img, rng, gamma):
    lut = (np.linspace(0, 1, 256) ** max(gamma, 0.05) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _x_hue(img, rng, deg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(round(deg / 2.0))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _x_saturation(img, rng, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=float(factor))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _x_gray(img, rng, alpha):
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(g, float(alpha), img, 1 - float(alpha), 0)


def _kelvin_rgb(k):
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return [min(max(v, 1.0), 255.0) for v in (r, g, b)]


def _x_temperature(img, rng, kelvin):
    r, g, b = _kelvin_rgb(kelvin)
    r0, g0, b0 = _kelvin_rgb(6600)
    gains = np.array([b / b0, g / g0, r / r0], np.float32)
    gains /= float(gains.mean())                          # the colour changes, the exposure does not
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def _x_rgb_shift(img, rng, amount):
    sh = rng.uniform(-amount, amount, 3).astype(np.float32)
    return np.clip(img.astype(np.float32) + sh, 0, 255).astype(np.uint8)


def _x_clahe(img, rng, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=float(max(clip, 0.1)), tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _x_equalize(img, rng):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _x_posterize(img, rng, bits):
    sh = 8 - int(min(max(round(bits), 1), 8))
    return ((img >> sh) << sh).astype(np.uint8)


def _x_solarize(img, rng, threshold):
    return np.where(img >= int(threshold), 255 - img, img).astype(np.uint8)


def _x_invert(img, rng):
    return 255 - img


def _x_channel_shuffle(img, rng):
    return np.ascontiguousarray(img[..., rng.permutation(3)])


def _x_kmeans(img, rng, colors):
    """N colours: k-means on a sample, then every pixel through a 32³ lookup table (fast at any size)."""
    k = int(min(max(round(colors), 2), 64))
    flat = img.reshape(-1, 3)
    idx = rng.choice(len(flat), size=min(len(flat), 20000), replace=False)
    _cv_seed(rng)
    _c, _l, centres = cv2.kmeans(flat[idx].astype(np.float32), k, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0), 1,
                                 cv2.KMEANS_PP_CENTERS)
    g = (np.arange(32, dtype=np.float32) * 8 + 4)
    cells = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d = ((cells[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    lut = centres[d.argmin(1)].astype(np.uint8).reshape(32, 32, 32, 3)
    q = img >> 3
    return lut[q[..., 0], q[..., 1], q[..., 2]]


def _lowfreq(rng, w, h, cells=6):
    f = rng.random((cells, cells)).astype(np.float32)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)


def _x_fog(img, rng, density):
    h, w = img.shape[:2]
    m = np.clip(float(density) * (0.6 + 0.6 * _lowfreq(rng, w, h)), 0, 0.97)[..., None]
    return np.clip(img.astype(np.float32) * (1 - m) + 235.0 * m, 0, 255).astype(np.uint8)


def _x_rain(img, rng, amount):
    h, w = img.shape[:2]
    n = int(amount * w * h / 900.0) + 1
    length = max(6, int(min(w, h) * (0.02 + 0.03 * amount)))
    ang = math.radians(float(rng.uniform(-20, 20)))
    dx, dy = int(length * math.sin(ang)), int(length * math.cos(ang))
    layer = np.zeros((h, w), np.uint8)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    for x, y in zip(xs.tolist(), ys.tolist()):
        cv2.line(layer, (x, y), (x + dx, y + dy), 200, 1)
    layer = cv2.blur(layer, (3, 3))
    out = cv2.blur(img, (3, 3)) if amount > 0.3 else img
    out = out.astype(np.float32) * (1 - 0.3 * amount) + layer[..., None].astype(np.float32) * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)


def _x_snow(img, rng, amount):
    h, w = img.shape[:2]
    n = int(amount * w * h / 600.0) + 1
    layer = np.zeros((h, w), np.uint8)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    rs = rng.integers(1, max(2, int(min(w, h) / 180) + 2), n)
    for x, y, r in zip(xs.tolist(), ys.tolist(), rs.tolist()):
        cv2.circle(layer, (x, y), r, 255, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.2)
    out = img.astype(np.float32) * (1 + 0.25 * amount) + 20 * amount + layer[..., None].astype(np.float32) * 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def _x_glare(img, rng, threshold, mul):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls[..., 1].astype(np.float32)
    hls[..., 1] = np.clip(np.where(L < threshold, L * mul, L), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def _x_sun_flare(img, rng, strength):
    h, w = img.shape[:2]
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h * 0.5))
    R = float(min(w, h)) * (0.25 + 0.5 * strength)
    k = 4 if min(w, h) >= 256 else 1
    ys, xs = np.mgrid[0:h:k, 0:w:k].astype(np.float32)
    glow = np.clip(1 - np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / R, 0, 1) ** 2
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR) * (255.0 * strength)
    over = np.zeros((h, w, 3), np.float32)
    tx, ty = (w / 2 - cx), (h / 2 - cy)
    for t in rng.uniform(0.3, 1.8, int(rng.integers(2, 6))):
        col = rng.uniform(120, 255, 3).tolist()
        tmp = np.zeros((h, w, 3), np.uint8)
        cv2.circle(tmp, (int(cx + tx * t), int(cy + ty * t)), int(R * rng.uniform(0.04, 0.16)), col, -1)
        over += tmp.astype(np.float32) * (0.25 * strength)
    return np.clip(img.astype(np.float32) + glow[..., None] + over, 0, 255).astype(np.uint8)


def _x_shadow(img, rng, darkness):
    h, w = img.shape[:2]
    k = 8 if min(w, h) >= 256 else 1                     # the soft mask is built small (fast on large images)
    sw, sh = max(2, w // k), max(2, h // k)
    mask = np.zeros((sh, sw), np.float32)
    n = int(rng.integers(3, 6))
    cx, cy = float(rng.uniform(0, sw)), float(rng.uniform(0, sh))
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    rad = rng.uniform(0.25, 0.9, n) * max(sw, sh)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.resize(cv2.GaussianBlur(mask, (0, 0), max(1.0, min(sw, sh) * 0.02)), (w, h),
                      interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.float32) * (1 - float(darkness) * mask[..., None]), 0, 255).astype(np.uint8)


def _x_vignette(img, rng, amount):
    h, w = img.shape[:2]
    xs = ((np.arange(w, dtype=np.float32) - w / 2) / (w / 2)) ** 2
    ys = ((np.arange(h, dtype=np.float32) - h / 2) / (h / 2)) ** 2
    m = 1 - float(amount) * np.clip((xs[None, :] + ys[:, None]) / 2.0, 0, 1)
    return np.clip(img.astype(np.float32) * m[..., None], 0, 255).astype(np.uint8)


def _x_low_light(img, rng, darkness):
    out = _x_gamma(img, rng, 1 + 1.5 * darkness)
    out = cv2.convertScaleAbs(out, alpha=1 - 0.6 * float(darkness))
    return _x_noise(out, rng, 4 + 18 * float(darkness), kind=0)


def _x_gauss_blur(img, rng, sigma):
    return img if sigma < 0.05 else cv2.GaussianBlur(img, (0, 0), float(sigma))


def _x_motion_blur(img, rng, kernel, angle):
    k = int(max(3, round(kernel))) | 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(angle), 1.0), (k, k))
    s = float(ker.sum())
    return img if s <= 0 else cv2.filter2D(img, -1, ker / s)


def _x_defocus(img, rng, radius):
    r = int(max(1, round(radius)))
    ker = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(ker, (r, r), r, 1.0, -1)
    return cv2.filter2D(img, -1, ker / float(ker.sum()))


def _x_ghost(img, rng, alpha, rotate):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rotate), 1.0)
    other = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return cv2.addWeighted(other, float(alpha), img, 1 - float(alpha), 0)


def _x_noise(img, rng, scale, kind=None):
    """Gaussian (0), Laplace (1), Poisson-like (2) or speckle (3) — one at random, as the older pipeline did."""
    if scale <= 0:
        return img
    kind = int(rng.integers(0, 4)) if kind is None else kind
    _cv_seed(rng)
    mono = rng.random() < 0.3 and kind != 3
    shape = img.shape[:2] + ((1,) if mono else (3,))
    n = np.empty(shape, np.int16)
    cv2.randn(n, 0, float(scale) * (0.7 if kind == 1 else 1.0))
    if kind == 1:                                          # heavier tails
        n2 = np.empty(shape, np.int16)
        cv2.randn(n2, 0, float(scale) * 0.7)
        n = (n.astype(np.int32) * np.abs(n2) // max(1, int(scale * 0.56))).clip(-255, 255).astype(np.int16)
    elif kind == 2:                                        # grows with brightness
        n = (n.astype(np.float32) * np.sqrt(img.mean(-1, keepdims=True) / 128.0 + 0.05)).astype(np.int16)
    elif kind == 3:                                        # multiplicative
        n = (n.astype(np.float32) * (img.astype(np.float32) / 128.0)).astype(np.int16)
    return _add_noise16(img, n)


def _x_shot_noise(img, rng, severity):
    return _x_noise(img, rng, [0, 8, 14, 22, 32, 46][int(min(max(round(severity), 1), 5))], kind=2)


def _x_salt_pepper(img, rng, amount):
    _cv_seed(rng)
    r = np.empty(img.shape[:2], np.uint8)
    cv2.randu(r, 0, 256)
    t = int(min(max(amount, 0.0), 1.0) * 127.5)
    out = img.copy()
    out[r < t] = 0
    out[r > 255 - t] = 255
    return out


def _x_jpeg(img, rng, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(min(max(round(quality), 1), 100))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _x_pixelate(img, rng, block):
    h, w = img.shape[:2]
    b = int(max(1, round(block)))
    if b <= 1:
        return img
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _x_downscale(img, rng, factor):
    h, w = img.shape[:2]
    f = min(max(factor, 0.02), 1.0)
    if f >= 0.999:
        return img
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _x_cutout(img, rng, holes, size):
    h, w = img.shape[:2]
    out = img.copy()
    mean = img.reshape(-1, 3).mean(0)
    for _ in range(int(max(1, round(holes)))):
        bw = max(1, int(w * size * rng.uniform(0.5, 1.5)))
        bh = max(1, int(h * size * rng.uniform(0.5, 1.5)))
        x, y = int(rng.integers(0, max(1, w - bw + 1))), int(rng.integers(0, max(1, h - bh + 1)))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            out[y:y + bh, x:x + bw] = 0
        elif mode == 1:
            out[y:y + bh, x:x + bw] = 114
        elif mode == 2:
            out[y:y + bh, x:x + bw] = mean.astype(np.uint8)
        else:
            out[y:y + bh, x:x + bw] = rng.integers(0, 256, (out[y:y + bh, x:x + bw].shape), dtype=np.uint8)
    return out


def _x_pixel_dropout(img, rng, share):
    _cv_seed(rng)
    r = np.empty(img.shape[:2], np.uint8)
    cv2.randu(r, 0, 256)
    out = img.copy()
    out[r < int(min(max(share, 0.0), 1.0) * 255)] = 0
    return out


def _x_channel_dropout(img, rng):
    out = img.copy()
    out[..., int(rng.integers(0, 3))] = 0
    return out


_FN = {k[3:]: v for k, v in list(globals().items()) if k[:3] in ("_m_", "_d_", "_x_")}
_INT = {("posterize", "bits"), ("kmeans", "colors"), ("motion_blur", "kernel"), ("defocus", "radius"),
        ("shot_noise", "severity"), ("jpeg", "quality"), ("pixelate", "block"), ("cutout", "holes"),
        ("solarize", "threshold")}
_BORDER = {"gray": (cv2.BORDER_CONSTANT, (114, 114, 114, 0)), "black": (cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
           "edge": (cv2.BORDER_REPLICATE, None), "reflect": (cv2.BORDER_REFLECT_101, None)}


# ====================================================================================================== the augmenter
class Augmenter:
    def __init__(self, recipe: dict | None = None, keep_size: bool = True, only: set | None = None):
        """`only`: {"pixel"} / {"matrix", "dense", "pixel", "lib"} — the kinds of effect allowed here."""
        self.recipe = clean(recipe)
        self.keep_size = keep_size
        self.skipped = [k for k, v in self.recipe["ops"].items() if v["on"] and not available(k)]
        self.active = [k for k, v in self.recipe["ops"].items()
                       if v["on"] and v["p"] > 0 and available(k) and (only is None or OPS[k]["kind"] in only)]
        self.active.sort(key=lambda k: _ORDER[k])

    # ------------------------------------------------------------------ what happens to one image
    def _values(self, key, rng) -> dict:
        o, cfg, s = OPS[key], self.recipe["ops"][key], self.recipe["strength"]
        vals = {}
        for p in o["params"]:
            lo, hi = cfg[p["name"]]
            v = float(lo if hi <= lo else rng.uniform(lo, hi))
            v = min(max(p["neutral"] + (v - p["neutral"]) * s, p["lo"]), p["hi"]) if s != 1.0 else v
            vals[p["name"]] = int(round(v)) if (key, p["name"]) in _INT else v
        return vals

    def plan(self, rng) -> list:
        """[(effect, {parameter: value})] for one image, in the order they are applied."""
        act, ops = self.active, self.recipe["ops"]
        if not act:
            return []
        mode = self.recipe["mode"]
        if mode == "one":                                  # one family at random, one of its effects, by its chance
            fams = sorted({OPS[k]["family"] for k in act}, key=_FAMILY_ORDER.get)
            fam = fams[int(rng.integers(0, len(fams)))]
            pool = [k for k in act if OPS[k]["family"] == fam]
            k = pool[int(rng.integers(0, len(pool)))]
            chosen = [k] if rng.random() < ops[k]["p"] else []
        elif mode == "each":
            chosen = [k for k in act if rng.random() < ops[k]["p"]]
        else:                                              # some: N different effects, likelier ones more often
            n = int(rng.integers(self.recipe["min"], self.recipe["max"] + 1))
            n = min(n, len(act))
            w = np.array([ops[k]["p"] for k in act], np.float64)
            chosen = [act[i] for i in rng.choice(len(act), size=n, replace=False, p=w / w.sum())] if n else []
        chosen.sort(key=lambda k: _ORDER[k])
        return [(k, self._values(k, rng)) for k in chosen]

    def run(self, img, plan, rng):
        """-> (image, forward point mapper or None, [(effect, values or error text)])."""
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = img.shape[:2]
        log, M, fields = [], None, []
        bmode, bval = _BORDER[self.recipe["border"]]
        for key, vals in plan:
            if OPS[key]["kind"] != "matrix":
                continue
            try:
                if key == "rot90":
                    k = int(rng.integers(1, 4))
                    if w != h and self.keep_size:
                        k = 2                              # a quarter turn would change the image size
                    m = _m_rotate(rng, w, h, 90 * k)
                    vals = {"deg": 90 * k}
                else:
                    m = _FN[key](rng, w, h, **vals)
                M = m if M is None else m @ M
                log.append((key, vals))
            except Exception as exc:
                log.append((key, f"failed: {exc}"))
        if M is not None:
            img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=bmode,
                                      borderValue=bval or 0)
        for key, vals in plan:
            if OPS[key]["kind"] != "dense":
                continue
            try:
                dx, dy = _FN[key](rng, w, h, **vals)
                xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                img = cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=bmode, borderValue=bval or 0)
                fields.append((dx, dy))
                log.append((key, vals))
            except Exception as exc:
                log.append((key, f"failed: {exc}"))
        pix = [(k, v) for k, v in plan if OPS[k]["kind"] in ("pixel", "lib")]
        if pix:
            gray = img.ndim == 2
            alpha = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else None
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if gray else (img[..., :3] if alpha is not None else img)
            bgr = np.ascontiguousarray(bgr)
            for key, vals in pix:
                try:
                    res = _lib_apply(key, bgr, rng) if OPS[key]["kind"] == "lib" else _FN[key](bgr, rng, **vals)
                    if res.shape == bgr.shape:
                        bgr = np.ascontiguousarray(res)
                        log.append((key, vals))
                except Exception as exc:
                    log.append((key, f"failed: {exc}"))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if gray else \
                (np.dstack([bgr, alpha]) if alpha is not None else bgr)
        mapper = None
        if M is not None or fields:
            def _map(P, M=M, fields=tuple(fields)):
                P = np.asarray(P, np.float64).reshape(-1, 2)
                if M is not None:
                    Q = np.concatenate([P, np.ones((len(P), 1))], 1) @ M.T
                    P = Q[:, :2] / np.where(np.abs(Q[:, 2:3]) < 1e-9, 1e-9, Q[:, 2:3])
                for dx, dy in fields:
                    P = _forward_dense(P, dx, dy)
                return P
            _map.dense = bool(fields)
            mapper = _map
        return img, mapper, log

    def apply(self, img, shapes=None, seed=None, rng=None):
        """-> (augmented image, shapes that are still in it, [(effect, values)])."""
        rng = rng if rng is not None else np.random.default_rng(seed)
        plan = self.plan(rng)
        out, fwd, log = self.run(img, plan, rng)
        if shapes is not None and fwd is not None:
            h, w = out.shape[:2]
            shapes = map_shapes(shapes, fwd, w, h, self.recipe["min_visible"], self.recipe["min_size"], fwd.dense)
        elif shapes is not None:
            shapes = [dict(s) for s in shapes]
        return out, shapes, log


def describe(log) -> str:
    """'Rotate 12.3° · JPEG 34' — what was done to one image."""
    bits = []
    for key, vals in log:
        o = OPS.get(key, {"title": key, "params": []})
        if isinstance(vals, str):
            bits.append(f"{o['title']} ({vals})")
            continue
        units = {p["name"]: (p["dec"], p["unit"]) for p in o["params"]}
        txt = " ".join(f"{v:.{units.get(k, (2, ''))[0]}f}{units.get(k, (2, ''))[1]}" if isinstance(v, float)
                       else f"{v}{units.get(k, (0, ''))[1]}" for k, v in vals.items())
        bits.append(f"{o['title']} {txt}".strip())
    return " · ".join(bits) if bits else "unchanged"


# ====================================================================================================== training
class UltralyticsTransform:
    """One step of Ultralytics' training pipeline (placed just before its Format step): `labels` is its dict with
    "img" (BGR uint8), "instances" (boxes / segments / keypoints) and "cls". Nothing is written anywhere: every
    image is augmented in memory, differently each epoch. A failure never stops training — the image goes through
    unchanged (the first error is printed once)."""

    def __init__(self, recipe: dict, share: float = 1.0):
        self.recipe = clean(recipe)
        self.share = float(share)                          # the part of the images that is augmented at all
        self._aug = None
        self._rng = None
        self._warned = False

    def __getstate__(self):                                # workers get the recipe, each builds its own generator
        return {"recipe": self.recipe, "share": self.share, "_aug": None, "_rng": None, "_warned": False}

    def __call__(self, labels):
        try:
            return self._run(labels)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"AIO QC augmentation skipped an image: {type(exc).__name__}: {exc}", flush=True)
            return labels

    def _run(self, labels):
        if self._aug is None:
            self._aug = Augmenter(self.recipe, keep_size=True)
            self._rng = np.random.default_rng()            # fresh entropy in every worker
        rng = self._rng
        if self.share < 1.0 and rng.random() > self.share:
            return labels
        img = labels["img"]
        inst = labels.get("instances")
        plan = self._aug.plan(rng)
        n = len(inst) if inst is not None else 0
        if n and getattr(inst, "keypoints", None) is not None:
            plan = [(k, v) for k, v in plan if OPS[k]["kind"] not in ("matrix", "dense")]   # pose: pixels only
        if not plan:
            return labels
        out, fwd, _log = self._aug.run(img, plan, rng)
        if fwd is not None and n:
            h, w = out.shape[:2]
            inst.convert_bbox("xyxy")
            inst.denormalize(w, h)
            boxes = np.asarray(inst.bboxes, np.float32).copy()
            segs = inst.segments
            has_seg = segs is not None and len(segs) == n
            new_segs = np.asarray(segs, np.float32).copy() if has_seg else segs
            keep = np.zeros(n, bool)
            for i in range(n):
                Q = fwd(new_segs[i]) if has_seg else fwd(_box_ring(*boxes[i].tolist()))
                x1, y1, x2, y2 = Q[:, 0].min(), Q[:, 1].min(), Q[:, 0].max(), Q[:, 1].max()
                full = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
                cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                if cx2 - cx1 >= max(2.0, self.recipe["min_size"]) and cy2 - cy1 >= max(2.0, self.recipe["min_size"]) \
                        and (cx2 - cx1) * (cy2 - cy1) / full >= self.recipe["min_visible"]:
                    keep[i] = True
                    boxes[i] = (cx1, cy1, cx2, cy2)
                    if has_seg:
                        Q[:, 0] = np.clip(Q[:, 0], 0, w)
                        Q[:, 1] = np.clip(Q[:, 1], 0, h)
                        new_segs[i] = Q.astype(np.float32)
            labels["instances"] = type(inst)(boxes[keep], new_segs[keep] if has_seg else segs, None,
                                             bbox_format="xyxy", normalized=False)
            if labels.get("cls") is not None and len(labels["cls"]) == n:
                labels["cls"] = labels["cls"][keep]
        labels["img"] = out
        return labels


class ImageRecipe:
    """A classifier's view of a recipe: one whole image in, the augmented image out (no boxes to move). Takes and
    gives a PIL image (RGB) or a numpy array (BGR), so it sits in front of any torchvision / timm transform chain.
    Every effect kind is allowed (geometric ones too: a class label does not move). Picklable — each data-loader
    worker builds its own random generator. A failure never stops training: the image goes through unchanged."""

    def __init__(self, recipe: dict, share: float = 1.0):
        self.recipe = clean(recipe)
        self.share = float(share)
        self._aug = None
        self._rng = None
        self._warned = False

    def __getstate__(self):
        return {"recipe": self.recipe, "share": self.share, "_aug": None, "_rng": None, "_warned": False}

    def __call__(self, im):
        try:
            if self._aug is None:
                self._aug = Augmenter(self.recipe, keep_size=True)
                self._rng = np.random.default_rng()
            rng = self._rng
            if self.share < 1.0 and rng.random() > self.share:
                return im
            plan = self._aug.plan(rng)
            if not plan:
                return im
            pil = not isinstance(im, np.ndarray)
            arr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR) if pil else im
            out, _fwd, _log = self._aug.run(arr, plan, rng)
            if not pil:
                return out
            from PIL import Image
            if out.ndim == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            return Image.fromarray(cv2.cvtColor(np.ascontiguousarray(out[..., :3]), cv2.COLOR_BGR2RGB))
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"AIO QC augmentation skipped an image: {type(exc).__name__}: {exc}", flush=True)
            return im


class Then:
    """first, then second (both picklable): the recipe in front of a library's own transform chain."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def __call__(self, x):
        return self.second(self.first(x))


def recipe_line(aug: "Augmenter") -> str:
    names = ", ".join(OPS[k]["title"] for k in aug.active[:12]) + (" …" if len(aug.active) > 12 else "")
    skipped = f" · not in this environment: {', '.join(OPS[k]['title'] for k in aug.skipped)}" if aug.skipped else ""
    return f"Augmentation recipe “{aug.recipe['name']}” on the fly ({len(aug.active)} effects: {names}){skipped}"


def _install_cls(D, recipe: dict, share: float) -> bool:
    """Ultralytics classifiers: ClassificationDataset's training transforms get the recipe in front."""
    C = getattr(D, "ClassificationDataset", None)
    if C is None or getattr(C.__init__, "_aioqc", False):
        return False
    original = C.__init__

    def __init__(self, *a, **kw):
        original(self, *a, **kw)
        augment = kw.get("augment", a[2] if len(a) > 2 else False)
        try:
            if augment and getattr(self, "torch_transforms", None) is not None:
                self.torch_transforms = Then(ImageRecipe(recipe, share), self.torch_transforms)
        except Exception as exc:
            print(f"AIO QC augmentation could not be added: {type(exc).__name__}: {exc}", flush=True)
    __init__._aioqc = True
    C.__init__ = __init__
    return True


def install_in_ultralytics(recipe: dict, share: float = 1.0) -> str:
    """Make every Ultralytics training dataset built from now on (in this process) apply `recipe` on the fly:
    YOLODataset.build_transforms is wrapped, so the step is there from the first epoch and after "close mosaic"
    rebuilds the pipeline; a classifier's ClassificationDataset gets it in front of its training transforms.
    -> a line for the log."""
    from ultralytics.data import dataset as D
    aug = Augmenter(recipe)
    if not aug.active:
        return "Augmentation recipe: nothing switched on"
    try:
        _install_cls(D, recipe, share)
    except Exception as exc:
        print(f"AIO QC augmentation (classifiers) could not be installed: {type(exc).__name__}: {exc}", flush=True)
    if getattr(D.YOLODataset.build_transforms, "_aioqc", False):
        return "Augmentation recipe: already installed"
    original = D.YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        t = original(self, hyp)
        try:
            if getattr(self, "augment", False):
                steps = t.transforms
                steps.insert(max(0, len(steps) - 1), UltralyticsTransform(recipe, share))
        except Exception as exc:
            print(f"AIO QC augmentation could not be added: {type(exc).__name__}: {exc}", flush=True)
        return t
    build_transforms._aioqc = True
    D.YOLODataset.build_transforms = build_transforms
    names = ", ".join(OPS[k]["title"] for k in aug.active[:12]) + (" …" if len(aug.active) > 12 else "")
    skipped = f" · not in this environment: {', '.join(OPS[k]['title'] for k in aug.skipped)}" if aug.skipped else ""
    return f"Augmentation recipe “{aug.recipe['name']}” on the fly ({len(aug.active)} effects: {names}){skipped}"
