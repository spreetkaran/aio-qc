# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Split planner: protection, soft kinds on diverse data, repair, fill, groups, reports."""
import random
from collections import Counter

from AIO_QC.core import split as S


def _items(n=2000, seed=1, rare_every=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        labs = Counter({rng.choice("abcd"): 1 + rng.randrange(3)})
        if rare_every and i % rare_every == 0:
            labs["rare"] += 1
        boxes = [(rng.random() * .5, rng.random() * .5, .5 + rng.random() * .5, .5 + rng.random() * .5)]
        out.append(S.SplitItem(f"k{i}", S.detection_features(labs, boxes, rng.choice(["well_lit", "lowlight"])),
                               group=f"g{i // 2}", labels=labs))
    return out


def test_balanced_no_leaks_rare_in_train():
    items = _items(rare_every=700)                          # "rare": 3 images
    p = S.plan_split(items, 0.1, 0.05, 42, 3)
    sp = p.report["splits"]
    assert abs(sp["val"] - 200) <= 4 and abs(sp["test"] - 100) <= 4
    assert p.report["groups"]["split_across"] == 0
    rare = next(r for r in p.report["features"] if r["value"] == "rare" and r["kind"] == "class")
    assert rare["train"] == 3 and rare["val"] == rare["test"] == 0
    for c, v in p.report["classes"].items():
        if c != "rare":
            assert 0.08 < v["val"] / sum(v.values()) < 0.12
    again = S.plan_split(items, 0.1, 0.05, 42, 3)
    assert again.assign == p.assign                         # same seed, same split


def test_diverse_dataset_still_gets_a_val_set():
    """Every image its own class combination (COCO-like): combinations are too fine-grained to protect."""
    rng = random.Random(3)
    names = [f"c{i}" for i in range(40)]
    items = []
    for i in range(300):
        labs = Counter({n: 1 for n in rng.sample(names, 4)})
        items.append(S.SplitItem(f"k{i}", S.detection_features(labs, []), labels=labs))
    p = S.plan_split(items, 0.15, 0.0, 42, 3)
    assert "combination" in p.report["soft"]
    assert p.report["splits"]["val"] >= 0.7 * 45


def test_repair_brings_a_missing_feature_in():
    items = [S.SplitItem(f"a{i}", {"class": ["a"]}) for i in range(40)]
    items += [S.SplitItem(f"b{i}", {"class": ["a", "b"]}) for i in range(10)]
    units = [[i] for i in range(len(items))]
    feats = [Counter(it.keys()) for it in items]
    where = ["val" if i < 5 else "train" for i in range(len(items))]          # val: 5 plain "a" images
    counts = {"val": Counter(k for i in range(5) for k in feats[i])}
    avail = Counter(k for f in feats for k in f)
    caps = {"class:a": 6, "class:b": 2}
    n = S._repair("val", units, feats, where, counts, caps, avail, set(), 3)
    assert n == 1 and sum(1 for i, w in enumerate(where) if w == "val" and "class:b" in feats[i]) == 1
    assert where.count("val") == 5


def test_build_items_groups_and_reports(tmp_path):
    from tests import fixtures as FX
    from AIO_QC.core.dataset import DatasetType, open_dataset
    root = FX.labelme(tmp_path / "ds", n=12)
    ad = open_dataset(root, DatasetType.LABELME)
    samples = ad.discover()
    items = S.build_items(samples, ad, None, measure=False)
    assert len(items) == 12 and all("class" in it.features for it in items)
    assert len({it.group for it in items}) == 1           # cam_frame_0000 … one video: one group
    p = S.plan_split(items, 0.2, 0, 1, 3)
    assert p.report["groups"]["split_across"] == 0
    S.write_reports(tmp_path / "rep", items, p.assign, p.report, {"val": 20})
    names = sorted(x.name for x in (tmp_path / "rep").iterdir())
    assert names == ["feature_map.csv", "feature_summary.csv", "split_manifest.json", "split_report.txt"]
    assert (tmp_path / "rep" / "feature_map.csv").read_text().count("\n") == 13


def test_focus_classes_drive_the_balance():
    f = S.detection_features(Counter({"plate": 1, "line": 2}), [(0, 0, .5, .5, "plate"), (.1, .1, .2, .2, "line")],
                             focus={"plate"})
    assert f["combination"] == ["plate"] and f["objects"] == ["1"] and f["class"] == ["line", "plate"]
