# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The Frame Extractor — AIO QC's video mode (formerly the standalone FrameExtractor 1.0). Imported lazily: only when
the mode, a CLI subcommand (aio-qc extract|probe|clip|eval|testset) or Testing needs it."""
from AIO_QC import __version__  # noqa: F401  (one version for the whole app)

__appname__ = "Frame Extractor"
