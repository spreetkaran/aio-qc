# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Sample data to learn with — made on this computer, never downloaded, never written over your own files.

    sample_dataset()  -> folder   a small LabelMe dataset (street scenes: car · person · plate) with a few planted
                                  problems for the checks to find: a label spelt two ways ("Car"), the same box twice,
                                  a box too small to be real, a box leaving the image, a file with no shapes, an image
                                  with no annotation file, two identical images
    sample_table()    -> file     a CSV of the same images (sizes, counts, a few gaps and odd values) for Data Studio
    sample_video()    -> file     a 12-second video of a moving scene for the Frame Extractor (None if PyAV is missing)

Everything goes under <app data>/samples/ and is made again only when missing.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from AIO_QC.core.diag import ignored as _ignored

W, H = 640, 400
N_IMAGES = 24
PLANTED = {
    "street_0004": "label spelt two ways: “Car” next to “car”",
    "street_0007": "the same box twice",
    "street_0010": "a box only 3 px wide",
    "street_0013": "a box leaving the image",
    "street_0016": "an annotation file without shapes",
    "street_0019": "an image without an annotation file",
    "street_0022": "an exact copy of street_0021",
}


def samples_root() -> Path:
    from AIO_QC.core.appdirs import data_home
    p = Path(str(data_home())) / "samples"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scene(rng: random.Random, i: int):
    """One street picture (PIL image) and its objects [(label, (x1, y1, x2, y2))]."""
    from PIL import Image, ImageDraw
    sky = (150 + rng.randint(-20, 20), 180 + rng.randint(-20, 20), 215)
    img = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 230, W, H], fill=(92, 96, 100))                       # road
    d.rectangle([0, 222, W, 232], fill=(170, 170, 160))                   # kerb
    for k in range(0, W, 80):                                             # lane marks
        d.rectangle([k + (i * 7) % 80, 318, k + (i * 7) % 80 + 40, 324], fill=(235, 235, 220))
    for b in range(4):                                                    # buildings
        x = b * 170 + rng.randint(-20, 20)
        top = rng.randint(40, 120)
        d.rectangle([x, top, x + 140, 222], fill=(rng.randint(120, 200), rng.randint(110, 170), rng.randint(100, 150)))
        for wy in range(top + 14, 210, 28):
            for wx in range(x + 12, x + 130, 30):
                d.rectangle([wx, wy, wx + 14, wy + 14], fill=(230, 220, 160) if rng.random() < .3 else (70, 80, 95))
    objs = []
    slots = rng.sample([10, 225, 440], rng.randint(1, 3))                 # cars with a number plate, apart
    for x0 in slots:
        x = x0 + rng.randint(0, 12)
        y = rng.randint(262, 300)
        col = rng.choice([(200, 40, 40), (40, 90, 200), (230, 230, 235), (30, 30, 35), (240, 190, 40)])
        d.rounded_rectangle([x, y, x + 170, y + 70], 14, fill=col)
        d.rounded_rectangle([x + 30, y - 30, x + 130, y + 6], 10, fill=col)
        d.rectangle([x + 40, y - 24, x + 120, y], fill=(170, 200, 220))
        for wx in (x + 22, x + 128):
            d.ellipse([wx, y + 52, wx + 30, y + 82], fill=(20, 20, 20))
        d.rectangle([x + 62, y + 42, x + 108, y + 56], fill=(245, 245, 245))
        d.text((x + 66, y + 43), f"{rng.randint(10, 99)}AB", fill=(10, 10, 10))
        objs.append(("car", (x, y - 30, x + 170, y + 82)))
        objs.append(("plate", (x + 62, y + 42, x + 108, y + 56)))
    for p in range(rng.randint(0, 2)):                                    # people on the pavement
        x = rng.randint(10, W - 40) if p == 0 else (rng.randint(10, W - 40) + W // 2) % (W - 40)
        y = rng.randint(140, 148)                                         # on the pavement, above the cars
        d.ellipse([x + 8, y, x + 24, y + 16], fill=(220, 180, 150))
        d.rectangle([x + 4, y + 16, x + 28, y + 52], fill=rng.choice([(60, 120, 60), (120, 60, 140), (40, 40, 90)]))
        d.rectangle([x + 6, y + 52, x + 26, y + 74], fill=(50, 50, 60))
        objs.append(("person", (x + 4, y, x + 28, y + 74)))
    return img, objs


def _rect(label, box):
    x1, y1, x2, y2 = box
    return {"label": label, "points": [[float(x1), float(y1)], [float(x2), float(y2)]], "shape_type": "rectangle",
            "flags": {}, "group_id": None, "description": ""}


def sample_dataset(force: bool = False) -> Path:
    root = samples_root() / "street-dataset"
    done = root / ".complete"
    if done.exists() and not force:
        return root
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    previous = None
    for i in range(N_IMAGES):
        name = f"street_{i:04d}"
        if name == "street_0022" and previous is not None:
            img, objs = previous                                          # an exact duplicate image
        else:
            img, objs = _scene(rng, i)
        img.save(root / f"{name}.jpg", quality=88)
        previous = (img, objs)
        shapes = [_rect(l, b) for l, b in objs]
        if name == "street_0004" and shapes:
            shapes[0]["label"] = "Car"
        if name == "street_0007" and shapes:
            shapes.append(dict(shapes[0]))
        if name == "street_0010":
            shapes.append(_rect("plate", (300, 330, 303, 333)))
        if name == "street_0013":
            shapes.append(_rect("car", (560, 260, 700, 380)))
        if name == "street_0016":
            shapes = []
        if name == "street_0019":
            continue
        with open(root / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({"version": "5.11.4", "flags": {}, "shapes": shapes, "imagePath": f"{name}.jpg",
                       "imageData": None, "imageHeight": H, "imageWidth": W}, f, indent=1)
    (root / "classes.txt").write_text("car\nperson\nplate\n", encoding="utf-8")
    (root / "README.txt").write_text(
        "A sample dataset made by AIO QC to learn with. It has a few problems planted on purpose:\n\n"
        + "\n".join(f"  {k}: {v}" for k, v in PLANTED.items())
        + "\n\nChange or delete anything here — it is made again when you pick “Try a sample” and it is missing.\n",
        encoding="utf-8")
    done.write_text("ok")
    return root


def sample_table(force: bool = False) -> Path:
    path = samples_root() / "street-objects.csv"
    if path.exists() and not force:
        return path
    rng = random.Random(11)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "camera", "time", "cars", "people", "plates", "width", "height", "blur", "reviewed_by"])
        for i in range(240):
            cars = rng.randint(0, 4)
            w.writerow([f"street_{i:04d}.jpg", rng.choice(["north", "south", "gate", "Gate "]),
                        f"2026-03-{1 + i // 24:02d} {8 + i % 12:02d}:{(i * 7) % 60:02d}",
                        cars, rng.randint(0, 3), cars if rng.random() > .1 else "",
                        640, 400 if i % 37 else 0, round(rng.uniform(40, 400), 1) if i % 19 else "",
                        rng.choice(["asha", "ravi", "", "Asha"])])
    return path


def sample_video(force: bool = False) -> Path | None:
    path = samples_root() / "street-drive.mp4"
    if path.exists() and not force:
        return path
    try:
        import av
        import numpy as np
    except ImportError:
        return None
    tmp = path.with_suffix(".part.mp4")
    rng = random.Random(3)
    frames = []
    for k in range(6):                                                    # a few scenes, panned across
        img, _ = _scene(rng, k)
        frames.append(np.asarray(img.resize((W * 2, H)), dtype=np.uint8))
    try:
        with av.open(str(tmp), "w") as out:
            codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
            st = out.add_stream(codec, rate=25)
            st.width, st.height = W, H
            st.pix_fmt = "yuv420p"
            for n in range(12 * 25):
                scene = frames[(n // 50) % len(frames)]
                off = int((n % 50) / 50 * W)
                fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(scene[:, off:off + W]), format="rgb24")
                for pkt in st.encode(fr):
                    out.mux(pkt)
            for pkt in st.encode():
                out.mux(pkt)
        tmp.replace(path)
        return path
    except Exception as _exc:
        _ignored(_exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
