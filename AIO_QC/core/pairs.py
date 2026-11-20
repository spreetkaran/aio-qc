# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image ↔ label pairing, for the Image–Label Pairs check (Qt-free; one walk of the dataset).

LabelMe (a JSON next to — or pointing at — its image):
    json_no_image     a JSON whose image is not found (its imagePath, or an image with its name next to it)
    image_only_in_json  the image exists only inside the JSON (imageData) — no image file
    image_no_json     an image no JSON uses
    shared_image      two or more JSONs use the same image
    several_images    no imagePath to decide, and several images share the JSON's name (a.jpg and a.png)
    path_other_name   imagePath names an image with another name than the JSON
    not_labelme       a .json that is not a LabelMe file (what it looks like is said)
YOLO (images/… and labels/… mirror each other; one .txt per image):
    label_no_image    a .txt in labels/ without the image at the mirrored place in images/
    image_no_label    an image without its .txt (a background image — or a missing label)
    case_mismatch     the .txt exists, but its name differs in upper / lower case (pairs on Windows, not on Linux)
    label_in_images   the .txt lies next to the image in images/ instead of in labels/
    several_images    several images (a.jpg and a.png) share one .txt
    not_yolo          a .txt that is not a YOLO label file (what it looks like is said)

For each finding a likely partner is searched (same name elsewhere in the dataset: another folder, another split,
other upper / lower case) and a fix proposed where one is safe: point imagePath at the image found, move the label
next to its image, create an empty label (YOLO background) or an empty LabelMe JSON for an image.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .formats import CLASS_LIST_NAMES, labelme_report, yolo_report
from .io import IMAGE_EXTENSIONS, walk_files
from AIO_QC.core.diag import ignored as _ignored

ERROR, WARNING = "Error", "Warning"
TEXT = {
    "json_no_image": "JSON without image", "image_only_in_json": "Image only inside the JSON",
    "image_no_json": "Image without JSON", "shared_image": "Several JSONs, one image",
    "several_images": "Several images, one name", "path_other_name": "imagePath names another file",
    "label_no_image": "Label without image", "image_no_label": "Image without label",
    "case_mismatch": "Name differs in upper / lower case", "label_in_images": "Label in the images folder",
    "not_labelme": "Not a LabelMe JSON", "not_yolo": "Not a YOLO label file",
}
SEVERITY = {"json_no_image": ERROR, "image_only_in_json": WARNING, "image_no_json": WARNING,
            "shared_image": ERROR, "several_images": WARNING, "path_other_name": WARNING,
            "label_no_image": ERROR, "image_no_label": WARNING, "case_mismatch": ERROR,
            "label_in_images": ERROR, "not_labelme": ERROR, "not_yolo": ERROR}
FIX_TEXT = {"set_image_path": "point imagePath at it", "move_label": "move the label next to its image",
            "rename_label": "rename the label to match", "empty_label": "create an empty label (background)",
            "empty_json": "create an empty LabelMe JSON", "trash": "move to trash"}


@dataclass
class Finding:
    kind: str
    path: str                      # the file the finding is about
    partner: str = ""              # its (likely) partner, when one was found
    detail: str = ""
    fix: str = ""                  # see FIX_TEXT ("" = nothing safe to do automatically)
    fix_arg: object = None
    key: str | None = None         # the sample key when the file is a sample of the dataset

    @property
    def text(self) -> str:
        return TEXT.get(self.kind, self.kind)

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.kind, WARNING)


@dataclass
class PairReport:
    kind: str                      # "labelme" | "yolo"
    matched: int = 0
    images: int = 0
    labels: int = 0
    findings: list = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(f.kind for f in self.findings)


def _rel(p, root) -> str:
    try:
        return Path(os.path.relpath(p, root)).as_posix()
    except ValueError:
        return str(p)


def _by_stem(paths):
    out = defaultdict(list)
    for p in paths:
        out[Path(p).stem.lower()].append(Path(p))
    return out


# ======================================================================================================== LabelMe
def labelme_pairs(root, read_json=None, cancel=None) -> PairReport:
    """Every .json and image under root (hidden folders skipped)."""
    root = Path(root)
    read_json = read_json or _read_json_light
    jsons = list(walk_files(root, {".json"}))
    images = list(walk_files(root, IMAGE_EXTENSIONS))
    rep = PairReport("labelme", images=len(images), labels=len(jsons))
    img_set = {os.path.normcase(os.path.abspath(p)) for p in images}
    img_stem = _by_stem(images)
    json_stem = _by_stem(jsons)
    users = defaultdict(list)                       # image -> JSONs using it
    proposed = set()                                # images a JSON's finding already points at
    for n, jp in enumerate(jsons):
        if cancel is not None and n % 256 == 0 and cancel():
            return rep
        try:
            data = read_json(jp)
        except Exception as exc:                    # noqa: BLE001 — unreadable: say why
            _ignored(exc)
            rep.findings.append(Finding("not_labelme", str(jp), detail=f"unreadable JSON ({exc})", fix="trash",
                                        key=str(jp)))
            continue
        fr = labelme_report(data)
        if not fr.ok:
            rep.findings.append(Finding("not_labelme", str(jp), detail=f"looks like {fr.looks_like}", fix="trash",
                                        key=str(jp)))
            continue
        folder = jp.parent
        ip = str(data.get("imagePath") or "").replace("\\", "/")
        found = None
        if ip:
            for c in (folder / ip, folder / Path(ip).name):
                if os.path.normcase(os.path.abspath(c)) in img_set:
                    found = Path(os.path.abspath(c))
                    break
        same = [p for p in img_stem.get(jp.stem.lower(), []) if p.parent == folder]
        if found is None:
            if len(same) > 1:
                rep.findings.append(Finding("several_images", str(jp), detail=(
                    f"imagePath '{ip}' not found; next to it: {', '.join(p.name for p in same)}" if ip else
                    f"no imagePath; next to it: {', '.join(p.name for p in same)}"), key=str(jp)))
                found = same[0]
            elif same:
                found = same[0]
                rel = Path(os.path.relpath(found, folder)).as_posix()
                if ip:                               # the app finds it; LabelMe itself would not
                    rep.findings.append(Finding("path_other_name", str(jp), str(found),
                                                f"imagePath '{ip}' does not exist; '{rel}' sits next to the JSON",
                                                fix="set_image_path", fix_arg=rel, key=str(jp)))
        if found is None:
            cands = img_stem.get(Path(ip).stem.lower() if ip else jp.stem.lower(), []) or \
                img_stem.get(jp.stem.lower(), [])
            embedded = bool(data.get("imageData"))
            if cands:
                c = cands[0]
                proposed.add(os.path.normcase(os.path.abspath(c)))
                rel = Path(os.path.relpath(c, folder)).as_posix()
                more = f" (and {len(cands) - 1} more with that name)" if len(cands) > 1 else ""
                rep.findings.append(Finding("json_no_image", str(jp), str(c),
                                            f"imagePath '{ip or '—'}' not found; an image with that name is at "
                                            f"'{_rel(c, root)}'{more}", fix="set_image_path", fix_arg=rel,
                                            key=str(jp)))
            elif embedded:
                rep.findings.append(Finding("image_only_in_json", str(jp), detail=(
                    f"imagePath '{ip or '—'}' not found; the picture is only inside the JSON (imageData)"),
                    key=str(jp)))
            else:
                rep.findings.append(Finding("json_no_image", str(jp), detail=(
                    f"imagePath '{ip or '—'}' not found, and no image with that name in the dataset"),
                    fix="trash", key=str(jp)))
            continue
        users[os.path.normcase(os.path.abspath(found))].append(jp)
        if ip and Path(ip).stem.lower() != jp.stem.lower() and found.stem.lower() != jp.stem.lower():
            mine = [p for p in same if os.path.normcase(os.path.abspath(p)) != os.path.normcase(str(found))]
            rep.findings.append(Finding(
                "path_other_name", str(jp), str(found),
                f"imagePath '{ip}' — the JSON is named '{jp.stem}'" +
                (f"; '{mine[0].name}' sits next to it" if mine else ""),
                fix="set_image_path" if mine else "", fix_arg=mine[0].name if mine else None, key=str(jp)))
        rep.matched += 1
    shared = {str(j) for js in users.values() if len(js) > 1 for j in js}
    rep.findings = [f for f in rep.findings if not (f.kind == "path_other_name" and f.path in shared)]
    for img, js in users.items():
        if len(js) > 1:
            for jp in js:
                rep.findings.append(Finding("shared_image", str(jp), img,
                                            f"'{Path(img).name}' is also used by "
                                            f"{', '.join(_rel(j, root) for j in js if j != jp)}", key=str(jp)))
    for img in images:
        if os.path.normcase(os.path.abspath(img)) in users or os.path.normcase(os.path.abspath(img)) in proposed:
            continue
        cands = json_stem.get(img.stem.lower(), [])
        if cands:
            rep.findings.append(Finding("image_no_json", str(img), str(cands[0]),
                                        f"no JSON uses it; a JSON with its name is at '{_rel(cands[0], root)}' "
                                        "(pointing at another image)", fix="empty_json"))
        else:
            rep.findings.append(Finding("image_no_json", str(img), detail="no JSON uses it (not annotated yet?)",
                                        fix="empty_json"))
    return rep


_IMGDATA = re.compile(rb'("imageData"\s*:\s*)"([^"]*)"')


def _read_json_light(p):
    """The JSON without its (often several MB) imageData — one read; an embedded picture is noted, not parsed."""
    raw = Path(p).read_bytes()
    m = _IMGDATA.search(raw)
    embedded = bool(m and m.group(2))
    if m:
        raw = raw[:m.start()] + m.group(1) + b"null" + raw[m.end():]
    data = json.loads(raw)
    if embedded and isinstance(data, dict):
        data["imageData"] = "<embedded>"
    return data


# =========================================================================================================== YOLO
def yolo_pairs(bases, root, n_classes: int = 0, kpt: bool = False, cancel=None) -> PairReport:
    """bases: [(name, base folder)] — each with images/ and labels/ (the adapter's bases)."""
    root = Path(root)
    rep = PairReport("yolo")
    all_imgs, all_labels, stray = [], [], []          # stray: .txt in images/
    per_base = []
    for _name, base in bases:
        ir, lr = Path(base) / "images", Path(base) / "labels"
        imgs = list(walk_files(ir, IMAGE_EXTENSIONS))
        labels = [p for p in walk_files(lr, {".txt"}) if p.name.lower() not in CLASS_LIST_NAMES]
        stray += [p for p in walk_files(ir, {".txt"}) if p.name.lower() not in CLASS_LIST_NAMES]
        per_base.append((ir, lr, imgs, labels))
        all_imgs += imgs
        all_labels += labels
    rep.images, rep.labels = len(all_imgs), len(all_labels)
    label_exact = {str(p) for p in all_labels}
    label_ci = {os.path.normcase(str(p)).lower(): p for p in all_labels}
    label_stem = _by_stem(all_labels)
    img_stem = _by_stem(all_imgs)
    stray_by = {(p.parent, p.stem.lower()): p for p in stray}
    claimed, proposed = set(), set()
    for ir, lr, imgs, _labels in per_base:
        by_target = defaultdict(list)
        for img in imgs:
            by_target[str(lr / img.relative_to(ir).with_suffix(".txt"))].append(img)
        for n, (target, group) in enumerate(by_target.items()):
            if cancel is not None and n % 512 == 0 and cancel():
                return rep
            img = group[0]
            if len(group) > 1:
                rep.findings.append(Finding("several_images", target, ", ".join(str(g) for g in group),
                                            f"{', '.join(g.name for g in group)} all use '{Path(target).name}'",
                                            key=str(img)))
            if target in label_exact:
                claimed.add(target)
                rep.matched += len(group)
                continue
            ci = label_ci.get(os.path.normcase(target).lower())
            if ci is not None:
                claimed.add(str(ci))
                rep.findings.append(Finding("case_mismatch", str(ci), str(img),
                                            f"'{ci.name}' vs image '{img.name}' — fine on Windows, unpaired on Linux "
                                            "(where most training runs)", fix="rename_label", fix_arg=target,
                                            key=str(img)))
                continue
            s = stray_by.get((img.parent, img.stem.lower()))
            if s is not None:
                rep.findings.append(Finding("label_in_images", str(s), str(img),
                                            f"'{s.name}' is in images/ — YOLO reads labels from '{_rel(Path(target).parent, root)}'",
                                            fix="move_label", fix_arg=target, key=str(img)))
                continue
            elsewhere = [p for p in label_stem.get(img.stem.lower(), []) if not _has_image_for(p, per_base)]
            if elsewhere:
                proposed.add(str(elsewhere[0]))        # reported once — from the image's side
            for g in group:
                if elsewhere:
                    rep.findings.append(Finding("image_no_label", str(g), str(elsewhere[0]),
                                                f"no '{Path(target).name}' here; an unpaired label with its name is at "
                                                f"'{_rel(elsewhere[0], root)}'", fix="move_label",
                                                fix_arg=(str(elsewhere[0]), target), key=str(g)))
                else:
                    rep.findings.append(Finding("image_no_label", str(g), detail=(
                        f"no '{_rel(target, root)}' — a background image (no objects), or a missing label"),
                        fix="empty_label", fix_arg=target, key=str(g)))
    for lab in all_labels:
        if str(lab) in claimed or str(lab) in proposed:
            continue
        if cancel is not None and cancel():
            return rep
        cands = img_stem.get(lab.stem.lower(), [])
        dest = None
        for c in cands:
            for ir, lr, _i, _l in per_base:
                try:
                    t = lr / c.relative_to(ir).with_suffix(".txt")
                except ValueError:
                    continue
                if str(t) not in label_exact and os.path.normcase(str(t)).lower() not in label_ci:
                    dest = (c, t)
                break
            if dest:
                break
        if dest:
            rep.findings.append(Finding("label_no_image", str(lab), str(dest[0]),
                                        f"no image at the mirrored place; '{dest[0].name}' is at "
                                        f"'{_rel(dest[0], root)}' (another folder / split)", fix="move_label",
                                        fix_arg=(str(lab), str(dest[1]))))
        else:
            rep.findings.append(Finding("label_no_image", str(lab), detail="no image with its name in the dataset",
                                        fix="trash"))
    # file format: every label file (paired or not)
    for n, lab in enumerate(all_labels + stray):
        if cancel is not None and n % 512 == 0 and cancel():
            return rep
        try:
            raw = lab.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = "\x00"
        except OSError as exc:
            rep.findings.append(Finding("not_yolo", str(lab), detail=f"unreadable ({exc})"))
            continue
        fr = yolo_report(text, n_classes, kpt, lab.name)
        if not fr.ok:
            rep.findings.append(Finding("not_yolo", str(lab), detail=f"looks like {fr.looks_like}", fix="trash"))
    return rep


def _has_image_for(label: Path, per_base) -> bool:
    for ir, lr, imgs, _l in per_base:
        try:
            rel = label.relative_to(lr)
        except ValueError:
            continue
        stem = ir / rel.with_suffix("")
        return any(Path(str(stem) + e).exists() for e in IMAGE_EXTENSIONS)
    return False
