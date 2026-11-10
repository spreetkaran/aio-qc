# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Authorship guard: the author shown in About, the package metadata and the copyright notices must all name
Preetkaran Singh. If any of them is edited, this test fails (it does not stop someone who edits the tests too —
the licence's copyright notice is what protects authorship legally)."""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTHOR = "Preetkaran Singh"


def test_author_everywhere():
    from AIO_QC.widgets import about
    assert about.AUTHOR == AUTHOR
    assert about.author() == AUTHOR                      # installed metadata or the fallback
    py = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert re.search(r'authors\s*=\s*\[\s*\{\s*name\s*=\s*"Preetkaran Singh"', py)
    for name in ("README.md", "THIRD_PARTY_NOTICES.md"):     # LICENSE is the verbatim GPL text
        assert f"Copyright (c) 2026 {AUTHOR}" in open(os.path.join(ROOT, name), encoding="utf-8").read(), name


def test_about_reads_package_metadata(monkeypatch):
    """With the package installed, About takes the author from its metadata, not from the source file."""
    import importlib.metadata as im
    from AIO_QC.widgets import about

    from AIO_QC import __version__
    monkeypatch.setattr(im, "metadata", lambda name: {"Author": "From Metadata", "Version": __version__})
    assert about.author() == "From Metadata"
    monkeypatch.setattr(im, "metadata", lambda name: {"Author-email": '"Preetkaran Singh" <x@y>',
                                                      "Version": __version__})
    assert about.author() == AUTHOR
    monkeypatch.setattr(im, "metadata", lambda name: {"Author": "Old Build", "Version": "0.0.1"})
    assert about.author() == AUTHOR                      # metadata of another version: not trusted

    def missing(name):
        raise im.PackageNotFoundError(name)
    monkeypatch.setattr(im, "metadata", missing)
    assert about.author() == AUTHOR
