# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Data Studio's own user guide (F1 in Data Studio)."""
from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from AIO_QC.core.tabular import steps as S

GUIDE = """
<h2>Data Studio</h2>
<p>The table-data mode of AIO QC — its own window, menus and settings. <b>Ctrl+Alt+M</b> or the switch in the
top-right corner goes back to Image QC; the app starts in the mode you used last (<code>aio-qc --mode data</code>,
or <code>aio-qc file.csv</code>, starts here).</p>

<h3>1 · Open data</h3>
<p>CSV / TSV / text (separator and encoding detected), Excel (pick the sheet), Parquet, Feather, JSON, JSON lines and
SQLite tables — <i>File ▸ Open Data</i>, drag and drop, or the recent list. Files above 400 MB open as a
<b>sample</b> of their first million rows (a banner says so); <i>Export Data</i> and <i>Run on Spark</i> use the
whole file. From Image QC, <i>Tools ▸ Annotations → Data Studio</i> opens the dataset as a table: one row per shape
with its file, split, label, box, size relative to the image, review state, description and flags.</p>

<h3>2 · The recipe</h3>
<p>Everything you do is a <b>step</b> in the table's recipe (left panel): nothing changes your files. Add steps from
<i>Add step</i> (Ctrl+T), the <i>Data</i> menu, a column header (right-click or double-click), or the fix buttons in
<i>Profile</i>. Click a step to see the data right after it; untick it to switch it off; double-click to edit;
drag (or Alt+↑ / ↓) to reorder; Del removes it. Ctrl+Z / Ctrl+Shift+Z undo and redo anything. Results are cached per
step, so editing a late step does not re-read the file.</p>
<table cellpadding='3'>STEPS</table>

<h3>3 · Profile</h3>
<p>The table's health: rows, columns, missing cells, duplicate rows and a 0–100 quality score; one card per column
with its type, missing / distinct counts, statistics, a small chart and warnings — constant columns, mostly-missing
columns, IDs, numbers or dates stored as text, spaces at the ends, outliers, skew — each with a one-click fix that
adds the matching step. Strong correlations are listed at the top.</p>

<h3>4 · Charts</h3>
<p>Histogram, bar (count / sum / mean / median / min / max of a column by category), line (over a number or date;
many x values are binned), scatter (up to 20,000 random points) and box plots; colour by a category. Hover for
values, <i>Save PNG</i> to keep one.</p>

<h3>5 · Code and Spark</h3>
<p><i>Code</i> shows the recipe as a runnable <b>pandas</b> or <b>PySpark</b> script — the app runs exactly the
pandas code it shows. <i>Run on Spark</i> runs the PySpark script in a Python environment that has pyspark (and
Java): on this computer (<code>local[*]</code>), on a cluster (<code>--master yarn</code>, <code>spark://…</code>)
or through Spark Connect (<code>sc://host:15002</code>); the output is a folder of part files. The app itself never
loads Spark. Spark has no row order, so “previous / next row” fills and duplicate removal follow the file's read
order.</p>

<h3>6 · Save and share</h3>
<p><i>Save Workspace</i> (Ctrl+S) keeps the tables and their recipes in a small <code>.aioqc-data.json</code> file —
not the data. <i>Export Data</i> writes CSV, Parquet, Excel or JSON; <i>Profile Report</i> one HTML page with every
column's health and the recipe.</p>

<h3>Formulas (New column)</h3>
<pre>FORMULAS</pre>
"""


def guide_html() -> str:
    rows = []
    for g in ("Columns", "Rows", "Clean", "Reshape", "Features"):
        ops = [o.title for o in S.OPS.values() if o.group == g]
        rows.append(f"<tr><td><b>{g}</b></td><td>{' · '.join(ops)}</td></tr>")
    form = (S.OPS["derive"].help or "").split("\n\n", 1)[-1].strip()
    import html
    return GUIDE.replace("STEPS", "".join(rows)).replace("FORMULAS", html.escape(form))


def show_guide(parent):
    d = QDialog(parent)
    d.setWindowTitle("Data Studio — Guide")
    d.resize(860, 760)
    lay = QVBoxLayout(d)
    t = QTextBrowser()
    t.setOpenExternalLinks(True)
    t.setHtml(guide_html())
    lay.addWidget(t)
    d.exec()
