# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The command line of add_data (no Qt): the folders are asked for when not given (paste or drag them in), the plan is
shown, nothing changes until you say yes — or --dry-run (a report only) / --yes (no question).

    aio-qc add-data     --dataset D --new N [--new N2 …] [--splits train|balanced|source] [--val 0.1] …
    aio-qc add-classes  --dataset D --source S [--classes "plate, person"] [--conflict keep|replace|union] …
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

from . import add_data as A
from . import common as C

PREVIEW = 6


def _parser():
    ap = argparse.ArgumentParser(prog="aio-qc", description="Add data to an existing LabelMe / YOLO dataset, "
                                                            "matched by image content (not file names).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dataset", help="the existing dataset (LabelMe folder or YOLO root, with or without splits)")
        p.add_argument("--rename-class", action="append", default=[], metavar="NEW=DATASET",
                       help="a label of the new data that is called differently in the dataset (repeatable)")
        p.add_argument("--new-classes", choices=("add", "drop", "skip"), default="add",
                       help="YOLO dataset — classes it does not have: add to its class list (default), drop those "
                            "objects, or leave the image out")
        p.add_argument("--dry-run", action="store_true", help="show the plan and write the report, change nothing")
        p.add_argument("-y", "--yes", action="store_true", help="go ahead without asking")
        p.add_argument("--workers", type=int, default=A.WORKERS, help="reading threads (default: %(default)s)")

    a = sub.add_parser("add-data", help="add the images a dataset does not have yet")
    common(a)
    a.add_argument("--new", action="append", default=[], help="a folder of new data (repeatable)")
    a.add_argument("--splits", choices=("train", "balanced", "source"), default="train",
                   help="a dataset with train / val / test: all new images to train (default), a balanced split, or "
                        "the new data's own splits")
    a.add_argument("--val", type=float, default=None, help="balanced: validation share (default: the dataset's)")
    a.add_argument("--test", type=float, default=None, help="balanced: test share (default: the dataset's)")
    a.add_argument("--on-match", choices=("skip", "replace", "merge"), default="skip",
                   help="an image the dataset already has: skip (default; differing annotations are listed), "
                        "replace its annotation, or merge in the classes it lacks")
    a.add_argument("--unlabelled", choices=("skip", "add"), default="skip",
                   help="new images without an annotation: leave out (default) or add as background")
    a.add_argument("--placement", choices=("mirror", "flat"), default="mirror",
                   help="LabelMe dataset: keep the new data's sub-folders (default) or one folder")
    a.add_argument("--subfolder", default="", help="LabelMe dataset: put the new data under this folder")
    a.add_argument("--link", choices=("copy", "hardlink"), default="copy", help="copies (default) or hard links")
    a.add_argument("--no-check-images", action="store_true", help="don't decode every new image first")
    a.add_argument("--naming", choices=("suffix", "folder", "path", "sequence", "hash", "custom"), default="suffix",
                   help="names taken by a different image: _1 suffix (default), folder prefix, full path prefix, "
                        "renumber, content hash, or --pattern")
    a.add_argument("--pattern", default="", help="custom naming pattern, e.g. {folder}_{stem} — tokens: "
                                                 "{stem} {folder} {folder2} {path} {n:04d} {hash8} {date} {split}")
    a.add_argument("--rename-all", action="store_true", help="rename every new file, not only those that clash")
    a.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("add-classes", help="bring annotations of classes over to the images a dataset already has")
    common(b)
    b.add_argument("--source", action="append", default=[], help="a folder with the same images, more classes "
                                                                 "(repeatable)")
    b.add_argument("--classes", default="", help="comma-separated (default: the classes the dataset does not have)")
    b.add_argument("--conflict", choices=("keep", "replace", "union"), default="keep",
                   help="a class both have in one image: keep the dataset's (default), replace them, or add the "
                        "objects that do not overlap")
    b.add_argument("--iou", type=float, default=0.6, help="union: overlap from which an object counts as there")
    b.add_argument("--also-add-new", action="store_true", help="also add the images the dataset does not have")
    return ap


def _ask(label, check, default=""):
    while True:
        try:
            raw = input(f"  {label}" + (f" [{default}]" if default else "") + ": ").strip().strip('"').strip("'")
        except EOFError:
            raise SystemExit(2) from None
        raw = raw or default
        if not raw:
            continue
        p = Path(raw).expanduser()
        why = check(p)
        if why is None:
            return str(p.resolve())
        print(f"  ✘ {why}")


def _is_dir(p):
    return None if p.is_dir() else f"not a folder: {p}"


def _width() -> int:
    return max(40, shutil.get_terminal_size((100, 20)).columns - 1)


def _clear():
    if sys.stdout.isatty():
        print("\r" + " " * _width() + "\r", end="", flush=True)


def _progress():
    tty = sys.stdout.isatty()
    last = [0.0, ""]

    def say(done, total, label):
        now = time.monotonic()
        if label != last[1] or now - last[0] > 0.3 or done == total:
            last[0], last[1] = now, label
            msg = f"  … {label}" + (f"  {done:,}/{total:,}" if total else "")
            if tty:
                w = _width()
                print("\r" + msg[:w].ljust(w), end="", flush=True)
            elif label != getattr(say, "shown", None):
                print(msg, flush=True)
                say.shown = label
    return say


def _short(P, text: str) -> str:
    """Paths shown below their folder's name (the report has them in full)."""
    for root, name in [(P.target.root, "dataset")] + [(D.root, D.root.name) for D in P.sources]:
        text = text.replace(str(root) + os.sep, name + "/")
    return text


def _show_plan(P):
    T = P.target
    print()
    print(f"  Dataset   {T.root}  ({T.fmt}{' · ' + T.task if T.fmt == 'yolo' else ''}"
          + (f" · splits {', '.join(T.splits)}" if T.splits else "") + f" · {len(T.items):,} images"
          + (f" · {len(T.classes)} class{'es' if len(T.classes) != 1 else ''}" if T.fmt == "yolo" else "") + ")")
    for D in P.sources:
        print(f"  New data  {D.root}  ({D.fmt} · {len(D.items):,} images)")
    print()
    w = max(len(str(n)) for _l, n in P.rows) if P.rows else 1
    for label, n in P.rows:
        print(f"  {n:>{max(w, 6)},}  {label}" if isinstance(n, int) else f"  {n}  {label}")
    for k, v in P.details.items():
        if not v:
            continue
        print(f"\n  {k}:")
        for line in v[:PREVIEW]:
            print(f"      {_short(P, str(line))}")
        if len(v) > PREVIEW:
            print(f"      … and {len(v) - PREVIEW:,} more (see the report)")
    if P.opts.on_match == "skip" and P.stats["match_diff"]:
        print("\n  Tip: --on-match replace takes the new annotations for those images; --on-match merge (or "
              "aio-qc add-classes) adds just the classes they lack.")
    print()


def _plan_report(P) -> Path:
    path = Path.cwd() / f"add_data_plan_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["status", "new data file", "dataset file", "added as", "note"])
        w.writerows(P.report)
    return path


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    interactive = sys.stdin.isatty() and not args.yes
    srcs = args.new if args.cmd == "add-data" else args.source
    if not args.dataset or not srcs:
        if not sys.stdin.isatty():
            print("  ✘ --dataset and " + ("--new" if args.cmd == "add-data" else "--source") + " are needed "
                  "(or run it in a terminal to be asked)")
            return 2
        print("\n  Folders  (type, paste or drag a folder in · Ctrl+C to quit)")
        if not args.dataset:
            args.dataset = _ask("Dataset (existing LabelMe / YOLO)", _is_dir)
        if not srcs:
            what = "New data" if args.cmd == "add-data" else "Source with the same images, more classes"
            srcs = [_ask(what, _is_dir)]
            while True:
                more = input("  Another folder (Enter: no more): ").strip().strip('"').strip("'")
                if not more:
                    break
                if Path(more).expanduser().is_dir():
                    srcs.append(str(Path(more).expanduser().resolve()))
                else:
                    print(f"  ✘ not a folder: {more}")
    rename = {}
    for r in args.rename_class:
        if "=" in r:
            k, v = r.split("=", 1)
            rename[k.strip()] = v.strip()
    if args.cmd == "add-data":
        naming = {"preset": args.naming, "pattern": args.pattern, "only_clashes": not args.rename_all}
        o = A.Options(args.dataset, srcs, add_new=True, on_match=args.on_match, split_mode=args.splits,
                      val=args.val, test=args.test, new_classes=args.new_classes, rename=rename,
                      unlabelled=args.unlabelled, placement=args.placement, subfolder=args.subfolder, link=args.link,
                      check_images=not args.no_check_images, naming=naming, seed=args.seed, workers=args.workers)
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        o = A.Options(args.dataset, srcs, add_new=args.also_add_new, on_match="merge", merge_classes=classes,
                      conflict=args.conflict, iou=args.iou, new_classes=args.new_classes, rename=rename,
                      workers=args.workers)
    say = _progress()
    t0 = time.monotonic()
    try:
        P = A.plan(o, say)
    except (ValueError, OSError) as exc:
        print(f"\n  ✘ {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n  Stopped — nothing was changed.")
        return 130
    _clear()
    _show_plan(P)
    if not P.actions and not P.new_classes:
        print(f"  Nothing to do ({time.monotonic() - t0:.1f} s).")
        return 0
    if args.dry_run:
        print(f"  Dry run — nothing changed. Report: {_plan_report(P)}")
        return 0
    if interactive:
        ans = input(f"  Go ahead ({len(P.adds):,} images to add, {len(P.updates):,} annotations to change)? "
                    "[Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("  Cancelled — nothing was changed.")
            return 1
    try:
        st = A.run(P, say)
    except KeyboardInterrupt:
        print("\n  Stopped — what was done is logged (Preparation ▸ Undo removes it).")
        return 130
    _clear()
    d = st["done"]
    print(f"  Done in {time.monotonic() - t0:.1f} s: {d.get('added', 0):,} images added, "
          f"{d.get('updated', 0):,} annotations changed" + (f", {d['classes']} classes added" if d.get("classes")
                                                            else ""))
    for e in st["errors"][:PREVIEW]:
        print(f"  ✘ {e}")
    if len(st["errors"]) > PREVIEW:
        print(f"  … and {len(st['errors']) - PREVIEW:,} more errors (see the log)")
    print(f"  Report: {st['report']}")
    print(f"  Undo:   AIO QC ▸ Preparation ▸ Undo (log {Path(st['undo']).name} in {C.LOG_DIR})")
    return 1 if st["errors"] else 0
