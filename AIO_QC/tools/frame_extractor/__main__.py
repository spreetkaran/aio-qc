# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""`python -m AIO_QC.tools.frame_extractor [...]` = `aio-qc --mode video [...]`; the subcommands (extract, probe,
testset, eval, clip) work the same. Kept so old shortcuts / scripts of the standalone FrameExtractor keep working."""
import sys


def main():
    from AIO_QC.__main__ import main as aio_main
    args = sys.argv[1:]
    if not args or args[0] not in ("extract", "probe", "testset", "eval", "clip"):
        if "--mode" not in args:
            args = ["--mode", "video"] + args
    sys.argv = [sys.argv[0]] + args
    aio_main()


if __name__ == "__main__":
    main()
