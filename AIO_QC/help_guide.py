# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
In-app user guide (Help ▸ User Guide, F1; shown once on first start).
HTML for QTextBrowser; <img src='aio-icon:NAME'> images are provided by the caller as document resources.
"""
from AIO_QC.core.diag import ignored as _ignored


def _i(name, size=18):
    return f"<img src='aio-icon:{name}' width={size} height={size}>"


GUIDE_ICONS = ["open", "list", "prev", "next", "hand", "cursor", "fit-window", "zoom", "brightness-contrast",
               "group-dataset", "group-labels", "group-shapes", "group-models", "group-review", "review-ok",
               "review-flag", "tool-stats", "tool-validation", "tool-pairs", "tool-clashes", "tool-empty", "tool-image-dups", "tool-ambiguity",
               "tool-label-map", "tool-remove-class", "tool-crop-gallery", "tool-overlaps", "tool-outliers",
               "tool-dim-area", "tool-model-setup", "tool-model-check", "tool-sampling", "tool-smart-filter", "tool-coverage", "tool-editor", "review-fixed", "tool-ocr", "restore-window", "tool-quality", "tool-rules", "tool-inbox", "tool-export",
               "logs", "delete", "undo", "edit", "tool-testing", "mode-video", "mode-data", "tool-sequence", "tool-similar", "tool-review-grid", "tool-saved-filter",
               "history", "tool-prep", "tool-sketch", "tool-roi", "tool-handoff", "import", "tool-report", "tool-guidelines", "tool-splits", "keyboard",
               "get-started", "screenshot", "next", "tasks", "reset-default", "review-pending", "crop-view",
               "tool-autoannotate", "search", "ai-box", "tool-ai-review", "terminal-spark"]


def _tool_map() -> str:
    """Which tool for what? — every QC tool by the question its tab answers, with what it leaves to another tool.
    Made from the tool list itself, so it can never drift from the app."""
    try:
        from AIO_QC.app import GROUPS, QC_TOOLS, TOOL_NOT_HERE
    except Exception as _exc:
        _ignored(_exc)
        return ""
    rows = []
    for key, title, question in GROUPS:
        rows.append(f"<tr><td colspan=3 style='padding:10px 4px 2px 4px'>{_i('group-' + key, 20)} <b>{title}</b>"
                    f" &nbsp;<i>{question}</i></td></tr>")
        for t in QC_TOOLS:
            if t.group == key:
                rows.append(f"<tr><td style='padding:2px 10px 2px 16px'>{_i(t.icon)} <b>{t.title}</b></td>"
                            f"<td style='padding:2px 10px'>{t.tip}</td>"
                            f"<td style='padding:2px 10px' class='muted'>{TOOL_NOT_HERE.get(t.key, '—')}</td></tr>")
    return ("<h2><a name='toolmap'></a>6b · Which tool for what?</h2>\n"
            "<p>One job per tool. The last column says what a tool deliberately leaves alone — and which tool "
            "handles it — so you know where to go next.</p>\n<table><tr><th align=left>Tool</th><th align=left>"
            "Answers</th><th align=left>Not here → goes to</th></tr>" + "".join(rows) + "</table>\n")


def guide_html(version: str) -> str:
    return f"""
<style>
  h1 {{ font-size: 20pt; margin-bottom: 2px; }}
  h2 {{ font-size: 14pt; margin-top: 22px; border-bottom: 1px solid palette(mid); padding-bottom: 3px; }}
  h3 {{ font-size: 11.5pt; margin-top: 14px; }}
  td {{ padding: 3px 10px 3px 0; vertical-align: top; }}
  .muted {{ color: gray; }}
  .key {{ font-weight: bold; }}
  .box {{ border: 1px solid palette(mid); padding: 8px; }}
</style>

<h1>AIO QC — User Guide</h1>
<p class='muted'>Version {version} · annotation quality control for LabelMe and YOLO datasets</p>

<p><b>Contents</b><br>
<a href='#what'>1 · What the app is for</a><br>
<a href='#start'>2 · Quick start (5 minutes)</a><br>
<a href='#data'>3 · Supported datasets</a><br>
<a href='#window'>4 · The window</a><br>
<a href='#view'>5 · Looking at images and annotations</a><br>
<a href='#workflow'>6 · The QC workflow, tab by tab</a><br>
<a href='#models'>7 · Checking labels with your models</a><br>
<a href='#review'>8 · Reviewing and signing off</a><br>
<a href='#safety'>9 · What changes your files (and how to undo)</a><br>
<a href='#large'>10 · Working with large datasets</a><br>
<a href='#team'>10a · Team work: hand-off batches, guidelines, history, reports</a><br>
<a href='#keys'>11 · Essential shortcuts</a><br>
<a href='#trouble'>12 · Troubleshooting</a></p>

<h2><a name='what'></a>1 · What the app is for</h2>
<p>AIO QC finds and fixes problems in annotated image datasets <b>before</b> you train on them:
broken or missing files, wrong or inconsistent label names, duplicate or sloppy shapes, class imbalance,
near-duplicate images leaking between train and val — and, with your own models, labels that are probably wrong.
It is a <b>quality-control</b> tool, not an annotation editor: it shows problems in ranked lists, lets you jump to
each one, and applies fixes only when you explicitly ask.</p>

<h2><a name='start'></a>2 · Quick start (5 minutes)</h2>
<p><b>New here?</b> With nothing open, the window shows what to do first: open a dataset, or <i>Try a sample
dataset</i> — a small street dataset made on your computer with a few problems planted on purpose. The first time a
dataset opens, a short tour lights up the parts of the window one at a time (<i>Help ▸ Quick Tour</i> repeats it).
<b>Health check</b> (toolbar, <span class='key'>Ctrl+Alt+H</span>) then runs the common checks in one go and says what
it found in plain words. <i>View ▸ Simple Menus</i> keeps the menus to the everyday commands (on for a new install):
<span class='key'>Ctrl+K</span> still finds every command, and every shortcut still works. A greyed-out button says
why when you point at it, and <span class='key'>?</span> shows
every key. Changes to many files at once, and moving files to the trash, end with a notification whose <b>Undo</b>
puts them back.</p>
<table>
<tr><td>{_i('open')}</td><td><b>1. Open a dataset</b> — toolbar <i>Open</i> (Ctrl+O), or drag a folder onto the window.
The layout (LabelMe / YOLO / YOLO split) is detected automatically.</td></tr>
<tr><td>{_i('list')}</td><td><b>2. Class names</b> — YOLO uses <code>dataset.yaml</code> (or <code>classes.txt</code>) automatically.
For LabelMe, optionally <i>Load Classes</i> (Ctrl+L) so unknown labels can be detected.</td></tr>
<tr><td>{_i('tool-stats')}</td><td><b>3. Get the overview</b> — <i>Dataset ▸ Dataset Stats</i>: labels, balance, image profile.</td></tr>
<tr><td>{_i('tool-validation')}</td><td><b>4. Run the checks left to right</b> — the tabs of the QC panel are in the recommended order
(Dataset → Labels → Shapes → Models → Review). Double-click any result to jump to it.</td></tr>
<tr><td>{_i('review-ok')}</td><td><b>5. Review</b> — step through files with <span class='key'>D</span> / <span class='key'>A</span>,
mark each <span class='key'>S</span> (OK) or <span class='key'>X</span> (flag). Progress is saved automatically.</td></tr>
</table>

<h2><a name='data'></a>3 · Supported datasets</h2>
<table>
<tr><td><b>LabelMe</b></td><td>A folder of <code>*.json</code> files (any depth) with the images next to them or referenced by
<code>imagePath</code>.</td></tr>
<tr><td><b>YOLO</b></td><td><code>root/images/…</code> + <code>root/labels/…</code> (also <code>images/train</code>-style sub-folders).
Detection, segmentation and OBB label lines are supported.</td></tr>
<tr><td><b>YOLO split</b></td><td><code>root/train/images</code>, <code>root/train/labels</code>, <code>root/val/…</code>, <code>root/test/…</code>.</td></tr>
<tr><td><b>COCO JSON</b></td><td>Opened in place: <code>annotations/instances_&lt;split&gt;.json</code> (images in
<code>&lt;split&gt;/</code>, <code>images/&lt;split&gt;/</code> or <code>&lt;split&gt;/images/</code>) or
<code>&lt;split&gt;/_annotations.coco.json</code> next to the images. Boxes, polygons (an object in pieces shares a
group ID) and RLE masks (shown as outlines). Edits are written back without losing anything: untouched objects keep
every field, a relabel changes only the category, a new label becomes a category; many edits are one write; Undo works
per image; taking an image out removes it (and its objects) from the file, and putting it back restores them.
Moving images between splits is not offered (the split is the file).</td></tr>
</table>
<p><b>Splits vs folders.</b> Folders are splits only when <code>train</code> and <code>val</code> (or
<code>valid</code> / <code>validation</code>) sit <b>side by side</b> in the same folder; <code>test</code> next to
them is optional. A lone <code>train/</code>, <code>val/</code> or <code>test/</code>, or train and val in different
places, are ordinary folders. Every other nesting (e.g. <code>cam1/day</code>) is a <i>source folder</i> with its own
<b>Folder</b> filter in the file list and its own rows in Stats and Model Check slices. When a dataset opens, the
status bar says which folders were taken as splits (also in the tooltip of the dataset name in the toolbar).</p>
<p><b>Class check (YOLO).</b> When the classes come from a classes.txt (no dataset.yaml), AIO QC checks the class ids
actually used in the label files against it: if a label uses an id the list has no name for, you are told at once
and asked to load the correct classes.txt; if everything fits, you are asked to confirm it is the final list (a
list in the wrong order would also fit) — remembered until the file changes. Load Classes runs the same check before
a file is used.</p>
<p>YOLO class names come from <code>dataset.yaml / data.yaml</code>; if there is none, from <code>classes.txt</code>
(in the dataset or loaded with <i>Load Classes</i>). The <i>Load Classes</i> button turns green when a class list is
active and amber when something needs attention — hover it to see why.</p>

<h3>Classification datasets</h3>
<p>Two layouts are recognised: <b>class folders</b> (<code>&lt;class&gt;/&lt;image&gt;</code>, optionally under
<code>train/ val/ valid/ test/</code> — any of them) and <b>LabelMe flags</b> (JSONs whose true flags are the image's
classes; values such as "yes" / 1 / "true" are read as true and reported). When a flags dataset opens you are asked
once whether an image has <b>one class</b> or <b>several</b> (multi-label) — remembered with the dataset.</p>
<p><b>Change class</b> — <i>Review ▸ Change Class…</i> (C) opens a picker: type to search, 1 … 9 pick, Enter takes the
first match or makes a new class; for multi-label datasets each class is ticked / unticked (mixed for a selection).
<span class='key'>1</span> … <span class='key'>9</span> on the image set that class directly (multi-label: toggle it).
For class folders the image moves to the other folder (review mark and findings follow). Every change is one
<span class='key'>Ctrl+Z</span> / <span class='key'>Ctrl+Y</span> step.</p>
<p><b>Class panel</b> — in a classification dataset the Annotations panel becomes <b>Class</b>: every class with its
colour, its number key and how many images it has, the image's class ticked. Click a class to set it (class folders:
the file moves); with several images selected in the file list a dash marks a class only some of them have and a click
sets it for all (multi-label: adds / removes it). Type to filter, Enter takes the first match; <b>New class…</b> adds
one, <b>Classes…</b> renames, merges and removes.</p>
<p><b>Class Gallery</b> (QC ▸ Labels; Ctrl+Shift+G — the Crop Gallery key — opens it in a classification dataset) — the fast
way to review thousands of images: one class at a time as a wall of tiles, where the image that does not belong stands
out. Left: every class with its colour, number key, image count, a bar of how much is reviewed and ✓ when signed off;
drop tiles on a class to move them. <b>Order</b>: <i>Most suspicious first</i> — your classifier's findings (sure of
another class, another class, its own class but unsure; from Model Check's cached results, no new model run), then the
least typical by look (vectors of a Visual Similarity run) — <i>Model: least sure first</i>, <i>Least typical
first</i>, <i>Grouped by look</i>, file order, random. Tiles carry the review mark, the model's other class ("→ dog
93 %") or its doubt ("? 41 %"), and "≈ cat" when a near-identical image sits in another class. Keys:
<span class='key'>1</span> … <span class='key'>9</span> that class, <span class='key'>C</span> pick a class,
<span class='key'>M</span> the model's class, <span class='key'>X</span> flag (Shift+X with a note),
<span class='key'>S</span> OK, <span class='key'>U</span> clear, <span class='key'>Space</span> every tile on screen you
did not mark becomes OK and the next screen opens (at the end it goes back to any image you scrolled past without
reviewing), <span class='key'>Del</span> trash, <span class='key'>Enter</span>
open in the viewer, hold <span class='key'>P</span> to peek at full size (with its look-alike), Ctrl+Z / Ctrl+Y.
<b>Compare with</b> shows a second class beside the first — drag tiles across. <b>Confusions</b> lists the classifier's
mix-ups (cat → dog, with counts) as image lists, with the model's class beside them; a double-click on a cell of Model
Check's confusion matrix opens the same. <b>Look-alikes</b>: images of the class with a near-identical image in another
class. <b>Sample check…</b> draws a random sample of the class big enough for an estimate (± 5 / 10 / 15 %); flag or
move the wrong ones, Space for the rest, then <b>Sign off</b> records the estimated number of wrong images in the whole
class (with its range). Built for big datasets: only tiles on screen are decoded, at tile size, in the background.
For classification datasets the filmstrip and the file list show each image's class.</p>
<p><b>Issues folder</b> (class folders) — <span class='key'>I</span> sets the selected image(s) aside: they move to
<code>issues/&lt;class&gt;/</code> (with splits <code>issues/&lt;split&gt;/&lt;class&gt;/</code>), out of the dataset,
and the next image opens — one key per image to clear thousands of crops; <span class='key'>Shift+I</span> adds a
note. Ctrl+Z puts them back (Ctrl+Y sets them aside again). It works on the image, in the Class Gallery (one tile
after another) and from the Class panel. Settings ▸ Classification chooses where the folder is: inside the dataset
(default — never read as a class), next to it, or a folder of your choice (another drive is fine). Every move is in
<code>issues_log.csv</code>. Review ▸ Issues Folder: <i>Review Issues…</i> opens the folder as a class-folder dataset
(fix classes in the Class Gallery there), <i>Put Issues Back…</i> returns each image to the class folder it is in
then.</p>
<p><b>Classifier versions of the box checks</b> — on a classification dataset the Shapes tab is called <b>Images</b>
and every check there (and on the Labels tab) opens its classifier version, asking the same question of whole images
and their classes: <b>Label Ambiguities</b> (one class written two ways — case, spaces, a typo, a plural; names
outside the class list; classes only one or two images use — Merge into… / Rename…), <b>Text Labels</b> (the class
names as text: spaces at the ends, invisible characters, characters a folder name cannot hold, separators or letter
case unlike the other names — Rename to suggestion), <b>Duplicates</b> (exact and near copies, above all a copy
filed under another class — Give copies the kept class), <b>Outliers</b> (images unusual for their class: size,
shape, brightness, contrast, colour, file size), <b>Sequence Check</b> (frames of one video in order: a class that
flips for a frame or two and back — Use the neighbours' class), <b>Dim / Area</b> (image width / height / area /
shape ranges), <b>Training Size</b> (what the classification transform does at imgsz: enlarged, centre crop cutting
much of the picture, squashed — and per class the imgsz that keeps 95 % sharp), <b>OCR Check</b> (the text in the
whole image read by OCR and compared with its class — digits, letters, signs), <b>Validation</b> and <b>Pairs</b> for
class folders (images outside any class folder, files that are not images, empty / unreadable files, wrong
extensions, empty class folders; annotation files left next to the images; one file name in several class folders),
and <b>Fix in Editor</b> (class folders: each image opens with one flag per class; ticking another class moves it to
that folder). Every fix is one step of Undo.</p>
<p><b>Classification Check</b> (QC ▸ Labels, Ctrl+Alt+D) — images without a class (and why: no flags, all false, an
unknown value), with several classes in a one-class dataset, flags that are not true / false, class names that differ
only in case / spaces / plural, classes with too few images, classes missing from a split, flags not in the class
list. <i>Fix values</i>, <i>Set class…</i>, <i>Merge into…</i>.</p>
<p><b>Classes</b> (Remove Class for classification) — rename, merge, remove (images to trash, or only the class), keep,
and the class order (up / down, A→Z, most images first) — one plan, one Apply, Undo.</p>
<p><b>Models</b> — Model Check scores a classifier (top-1 / top-5, per-class precision and recall, a confusion matrix)
and lists likely wrong labels, wrong classes and unsure images with <i>Use the model's class</i>. Model Assist shows
the classifier's three best classes as buttons. Train Model and Extend Model train classifiers straight from a
class-folder dataset — YOLO <code>-cls</code> models, and every other popular architecture (ResNet, EfficientNet,
ConvNeXt, MobileNet, ViT, Swin, EVA-02, DINOv2 … — see Train Model).</p>
<p><b>Testing</b> scores classifiers too: a class-folder or LabelMe-flag test set (or the dataset open in Image QC)
and exported classifiers (ultralytics <code>-cls</code>, ONNX with a class-probability output) give top-1 / top-5
accuracy, macro precision / recall / F1, per-class tables, a confusion matrix by class name, accuracy vs how many
images it answers (the threshold becomes "unsure below"), the surest mistakes as pictures with the answer under each,
A vs B (per-class recall difference, the images only one of them gets right), and on videos the answer's confidence
over time with brief flips and drop-outs. An image's class can be changed from the viewer (1 … 9, Enter); reports and
CSV / JSON exports are classifier-shaped. In the player a classifier's answer is a pill in the corner, not a box
around the frame.</p>
<p><b>Frame Extractor</b> pre-labels with a classifier as classes: LabelMe files whose flag is the class, or (YOLO
layout) each frame in its class folder — Open in Image QC opens either as a classification dataset. <b>Preparation ▸
Pick</b> asks a classifier where it hesitates between two classes and can sort the picks into folders by its guess
(Improve Model ▸ Pick new images uses it).</p>
<p>Stats, Duplicate Images (images in two classes = a conflict), Image Quality, Rules, Sampling, the Inbox, Data Studio
(one row per image class, with the image's size) and Convert &amp; Split (class folders, LabelMe flags or a CSV table)
work as usual.</p>

<h2><a name='window'></a>4 · The window</h2>
<table>
<tr><td>{_i('crop-view')}</td><td><b>Crop Viewer</b> — right-click an annotation in the Annotations panel ▸ <i>View
crop…</i> (or <span class='key'>Shift+C</span>): the shape's crop in its own window, cut from the image on screen.
Padding, outline, <i>Dim outside</i>, <i>As shown</i> (with brightness / contrast), fit / 1:1, wheel zoom, drag to pan;
<span class='key'>←</span> / <span class='key'>→</span> step through the image's annotations, <i>Follow selection</i>
keeps it on the shape selected on the canvas. Copy (Ctrl+C) or save the crop.</td></tr>
<tr><td><b>Appearance</b></td><td>Settings ▸ General ▸ Appearance: follow the system, light or dark — every mode
switches at once. Settings ▸ Viewer ▸ Canvas background: dark grey (default), light grey or the window colour.</td></tr>
<tr><td><b>Settings</b></td><td>The gear at the right of the menu bar, File ▸ Settings… (Ctrl+,) or the command
palette: every preference of Image QC in one searchable window (Ctrl+F) - interface, dataset watching and cache, review,
storage &amp; backups, models, keyboard. Changes apply at once; {_i('reset-default')} (next to a changed setting) puts it back to its default, and Export… /
Import… share a set-up. The menus keep the commands.</td></tr>
<tr><td><b>Toolbar</b></td><td>Open / Load Classes, previous / next image, Drag / Select mode, zoom, reset, logs, {_i('tasks')}<b>Background Tasks</b> (every job of this session, running and finished: progress, details, start time, how long it took, Cancel — also when you click the loading ring in the status bar).
Icons only by default (names in the tooltips); Settings ▸ Interface ▸ Compact toolbar switches the text back on.</td></tr>
<tr><td><b>QC panel</b> (top)</td><td>A slim tab row: click a tab — or rest the mouse on the tabs for a moment — and
the tools slide down <b>over</b> the image (the image keeps its size). The drawer is the same size for every tab, so
switching tabs (click, or rest on another tab) just swaps the tools in place. It slides back up when you launch a
tool, press Esc, click elsewhere, click the open tab again, or keep the mouse away from it for a moment. (or double-clicking a
tab, or View ▸ Pin QC Panel) keeps them open under the tabs. One tab per question: {_i('group-dataset')} Dataset, {_i('group-labels')} Labels,
{_i('group-shapes')} Shapes, {_i('group-models')} Models, {_i('group-review')} Review, and {_i('brightness-contrast')} Display
(view-only brightness / contrast / gamma / auto-contrast, fill opacity, outline width, label size). Class labels keep
one readable size at every zoom and are placed so they never cover each other.</td></tr>
<tr><td><b>File List</b></td><td>Search (Ctrl+P), split / folder / review filters (All / Unreviewed / OK / Flagged);
long names scroll sideways. A yellow banner
appears when a QC tool has narrowed the list — <i>Clear</i> shows everything again. Red names = unreadable files,
grey = no shapes.</td></tr>
<tr><td><b>Annotations</b></td><td>The shapes of the current image: shape type drawn in its label colour. Click to select,
click again to deselect, double-click to zoom to it, right-click to rename its label.</td></tr>
<tr><td><b>Labels</b></td><td>Every label in the dataset with its count; untick to hide a label, right-click for
“show files with this label”, “show only this label”, rename / mapping.</td></tr>
<tr><td><b>Thumbnails</b></td><td>Thumbnails around the current image (vertical when docked left / right).</td></tr>
<tr><td><b>Status bar</b></td><td>Left, like Paint: pointer position on the image, size of the selected shape / box
being drawn / ROI, image size; then messages. Right: review state of the current file, indexing progress, position,
mode, active label mapping.</td></tr>
<tr><td><b>Zoom</b></td><td>Toolbar: zoom box, Fit Window, 1:1, zoom out, slider, zoom in (Ctrl + wheel over the image
too).</td></tr>
</table>
<p class='muted'>Panels can be moved, floated or hidden (View ▸ Panels); the layout is remembered. The side panels run
the full height of the window. <b>Focus mode</b> (F11 or Ctrl+Alt+F) hides every panel and the toolbar — only the
image — and the same key brings your exact layout back. <b>Ctrl+K</b> opens the command palette: type a few letters
(“crop”, “dim”, “flag”, “storage”) to run any command, even with every panel hidden.</p>

<h2><a name='view'></a>5 · Looking at images and annotations</h2>
<table>
<tr><td>{_i('hand')}</td><td><b>Drag mode</b> (H) — for moving around: drag to pan; a click only clears a selection.</td></tr>
<tr><td>{_i('cursor')}</td><td><b>Select mode</b> (V) — for inspecting shapes: hover highlights shapes (thin lines and points
too), click selects, hold <span class='key'>Space</span> to pan, <span class='key'>Tab</span> / <span class='key'>Shift+Tab</span>
cycles shapes. Selecting a shape any other way (Annotations dock, Tab, a QC tool) switches to Select mode
automatically.</td></tr>
<tr><td>{_i('fit-window')}</td><td><b>Zoom</b>: Ctrl+wheel at the cursor, Ctrl+F fit, Ctrl+1 100 %.
<b>Double-click</b> toggles between fit and a close-up at the click; double-clicking a shape (Select mode) zooms to it.</td></tr>
<tr><td>{_i('brightness-contrast')}</td><td><b>Display tab</b>: make dark or hazy images readable without touching the files
(the tab shows <i>Display •</i> while an adjustment is active).</td></tr>
<tr><td>{_i('edit')}</td><td><b>Fix it right here</b> (Select mode, select a shape): drag its <b>handles</b> to move a
corner / edge / vertex, drag the shape to move it, <span class='key'>arrows</span> nudge by one image pixel
(<span class='key'>Shift</span> ×10). <b>Tighten a loose box from the keyboard</b>: click one of its edge or corner
handles (it fills in) — now the <span class='key'>arrows</span> move only that side or corner; <span class='key'>Esc</span>
(or a click on the shape) goes back to moving the whole shape. <span class='key'>Ctrl</span>+click on an outline adds a vertex,
<span class='key'>Alt</span>+click on a vertex removes it, <span class='key'>Del</span> deletes the shape,
<span class='key'>1</span>–<span class='key'>9</span> relabel to class 1–9 of the class list (shown in the status bar),
<span class='key'>L</span> / <span class='key'>F2</span> relabel from the list, <span class='key'>B</span> then drag draws a new
box. <span class='key'>Ctrl+Z</span> / <span class='key'>Ctrl+Shift+Z</span> undo / redo (Edit menu).<br>
<b>Several shapes at once</b>: <span class='key'>Ctrl</span>+click (⌘ on macOS) puts a shape in or takes it out of the
selection, <span class='key'>Shift</span>+click adds it, <span class='key'>Ctrl+A</span> selects all of them once one is
selected — in Select and Drag mode, and in the <i>Annotations</i> panel too (Ctrl+click, Shift+click for a range;
the rows and the image show the same selection). Drag, the arrows, <span class='key'>Del</span> and relabel then act
on the whole selection (one Undo step).
<b>Copy / cut / paste</b> (<span class='key'>Ctrl+C</span> / <span class='key'>Ctrl+X</span> / <span class='key'>Ctrl+V</span>;
Edit menu; anywhere in the window except text fields — the image, the Annotations panel, the file list): paste into
the same image (a little offset) or into another one (same place);
each pasted shape is one Undo step, so <span class='key'>Ctrl+Z</span> takes them back one by one. With one polygon
selected, Ctrl+click on its outline still adds a vertex.<br>
Every shape type (rectangle, polygon, circle, line, linestrip, point) in LabelMe and YOLO (detect, segment, OBB): a
rectangle stays a rectangle, a circle keeps its centre, polygons never drop below 3 points, YOLO 4-point polygons
keep 4 vertices (they may be oriented boxes), pose boxes are not reshaped here (their keypoints would not follow),
YOLO labels stay within the class map, untouched YOLO rows stay byte-identical. Each edit re-reads the file and is
refused if the file changed on disk meanwhile or the shape is no longer the one shown; the original is backed up
once, and undo restores the exact bytes. A flagged file you edit becomes {_i('review-fixed')} Fixed.</td></tr>
</table>
<p><span class='key'>Esc</span> lets go of an active handle first, then clears the selection.</p>

<h2><a name='workflow'></a>6 · The QC workflow, tab by tab</h2>
<p>Work left to right: fix files first, then names, then shapes, then use models, then sign off.
Every tool opens as a window you can keep open while browsing; <b>double-click a result</b> to open that file and shape
on the image — the keyboard goes to the image, so the arrows, <span class='key'>Del</span> and the relabel keys fix it
right away — then <span class='key'>F8</span> / <span class='key'>Shift+F8</span> open the next / previous result of that
check (from the image or the check's window), each with its shape selected. <b>Tick</b> rows to act on them,
<i>Show in file list</i> to review the results one by one, <i>CSV</i> to export.</p>
<p>{_i('review-flag')} <b>Flag from anywhere</b>: every tool has <i>⚑ Flag files</i>, and right-clicking a result row, card
or tile offers Flag / Flag with note / Mark OK / Clear. The note records which tool flagged it and why; the mark shows
up in the file list, thumbnails, Coverage Map and as a badge in every tool.</p>

<h3>{_i('group-dataset')} Dataset — are the files sound?</h3>
<table>
<tr><td>{_i('tool-stats')}</td><td><b>Dataset Stats</b> — counts per label and split, colour-coded balance, and an image profile
(resolution, aspect, lighting …). Double-click any row to list its files.</td></tr>
<tr><td>{_i('tool-validation')}</td><td><b>Validation</b> — is every file really a LabelMe JSON / YOLO label? If not,
it says what the file looks like (a COCO file, a Label Studio export, a VIA project …; a .txt with class names instead
of ids, pixel coordinates instead of 0…1, CSV, KITTI, DOTA …). Then the structure of each shape / line (label text,
[x, y] points, the point count for its type, class ids), the image size, wrong class ids and invalid geometry. Rows
with a <i>Proposed fix</i> can be fixed after you review each change (<i>Review fixes…</i>).</td></tr>
<tr><td>{_i('tool-pairs')}</td><td><b>Image–Label Pairs</b> (Ctrl+Alt+K) — every image with its label file, and
everything that is not. <b>LabelMe:</b> JSONs whose image is missing (neither its imagePath nor an image with its name
next to it), pictures only embedded in the JSON, images no JSON uses, two JSONs on one image, imagePath naming another
file, several images with the JSON's name. <b>YOLO</b> (images/ and labels/ are separate, mirrored folders): labels
without their image, images without a label (a background image — or a missing label), labels whose name differs only
in upper / lower case (they pair on Windows but not on Linux, where training usually runs), labels left in images/,
several images on one label. Files that are not LabelMe / YOLO at all are listed too. For each finding the likely
partner is searched across the dataset (another folder or split) and a safe fix proposed — point imagePath at the
image, move the label next to its image, rename it to the image's case, create an empty label (background) or an
empty LabelMe JSON. <i>Apply proposed fixes</i> never overwrites a file and <i>Undo fixes</i> puts everything back;
<i>Move to trash</i> is recoverable.</td></tr>
<tr><td>{_i('tool-clashes')}</td><td><b>Name Clashes</b> (Ctrl+Alt+C) — file names that would collide when the dataset
is flattened (Convert &amp; Split flat, Combine &amp; Flatten, most training tools) or copied to Windows: the same name in
different sub-folders (camA/img001.jpg and camB/img001.jpg), names that differ only in upper / lower case (Windows and
macOS see ONE file), two images sharing one label file (a.jpg and a.png both use a.json / a.txt), names Windows refuses
(back-slash, / : * ? " &lt; &gt; |, CON, PRN, AUX, NUL, COM1 …, LPT1 …, a trailing dot or space) and paths over 260 characters.
Each row says whether the images are identical (a duplicate — Duplicate Images removes one) or different, and what
would break. <i>Rename…</i> opens the naming panel: a preset (folder prefix, full path prefix, number suffix, renumber,
content hash) or your own pattern with tokens such as {{folder}}, {{path}}, {{n:04d}}, {{hash8}}, {{date}}; separator, letter
case, spaces and length; save your own presets. Every name is checked as you type — against every other file, letter
case ignored, by name without extension — and Rename stays off while a name still clashes (<i>Number the rest</i>
adds _1, _2 …). Image and label are renamed together, imagePath is fixed, review marks, notes, issues, cached results
and backups follow, renamed.csv lists every change, and Undo puts the names back.</td></tr>
<tr><td>{_i('tool-quality')}</td><td><b>Image Quality</b> (Ctrl+Shift+Q) — <b>errors</b>: corrupt or truncated files
(“Corrupt JPEG data: … extraneous bytes before marker”, “Premature end of JPEG file”, broken PNG chunks), images that
cannot be decoded, EXIF rotation that disagrees with the annotation (the boxes would appear rotated in tools that apply
EXIF), annotation size ≠ image size. <b>Information only</b>: blurry, under- / over-exposed and low-contrast images
(adjustable thresholds). Every image is also checked automatically in the background after a dataset opens: broken
files turn red in the file list, a notification offers <i>Show</i>, and a red banner appears over a broken image when
you open it. <i>⟳ Fix EXIF rotation</i> bakes the rotation into the pixels of the ticked images (tag reset; other EXIF
data and the colour profile kept) and brings their labels into the same upright frame — every file is kept first and
<i>↶ Undo rotation</i> restores it exactly.</td></tr>
<tr><td>{_i('tool-empty')}</td><td><b>Empty Annotations</b> — label files without shapes. For YOLO these are often intentional background images. (Images with no label file at all: <i>Image–Label Pairs</i>.)</td></tr>
<tr><td>{_i('tool-image-dups')}</td><td><b>Duplicate Images</b> — identical and near-identical images, and the same scene in
several splits (leakage). <i>Keep one per group</i> ticks the extra copies.</td></tr>
</table>

<h3>{_i('group-labels')} Labels — are the label names right?</h3>
<table>
<tr><td>{_i('tool-ambiguity')}</td><td><b>Label Ambiguities</b> — spelling variants, stray spaces, unknown labels.
<i>Fix in Label Mapping</i> hands the suggestions over.</td></tr>
<tr><td>{_i('tool-label-map')}</td><td><b>Label Mapping</b> (Ctrl+M) — the one place to rename or merge labels: either
<b>view-only</b> (files untouched; everything in the app uses the new names) or <b>written to files</b>.</td></tr>
<tr><td>{_i('tool-class-gallery')}</td><td><b>Class Gallery</b> (classification datasets; Ctrl+Shift+G opens it there) — one class at a
time as a wall of images, so the odd ones stand out; see <i>Classification datasets</i> below.</td></tr>
<tr><td>{_i('tool-text-labels')}</td><td><b>Text Labels</b> (Ctrl+Alt+T) — checks the text <b>stored</b> in your
annotations (LabelMe <code>description</code> or another shape field) without any OCR model, and without assuming
any particular kind of text. The classes that carry text are found in the data. Each text is normalised (upper case,
spaces / dashes removed — your choice) and judged by: the <b>validators you choose</b> (optional — the same library as
OCR Check, plus your own), and what the app <b>learns from the open dataset</b>, per class — the formats most texts
share (“4 letters · 7 digits”; only when ≥ 80 % of at least 10 different texts fit a few structures, so free text
gets no format), the texts that occur again and again, the characters the class uses (any script) and the characters
its frames confuse. In video / burst frames the same object is followed and compared. Findings: <b>no text</b>,
<b>rule failed</b>, <b>unusual format</b>, <b>look-alike of a frequent text</b>, <b>differs from other frames</b>,
<b>not normalised</b> (optional). Corrections, cheapest first: what the other frames agree on, a frequent known text
that is a look-alike, one look-alike character (letter↔digit, plus the learned confusions), a known text one or two
characters away, two look-alikes, any one character. <i>Sure</i> corrections are applied with <i>✓ Apply
suggestion</i>; otherwise you pick. Every write is re-checked against the file as it is now, backed up and
undoable.</td></tr>
<tr><td>{_i('tool-remove-class')}</td><td><b>Remove / Re-order Classes</b> — delete a class everywhere; for YOLO also renumber
ids and update the class file. Always preview first; backups are made by default.</td></tr>
</table>

<h3>{_i('group-shapes')} Shapes — are the annotations right?</h3>
<table>
<tr><td>{_i('tool-rules')}</td><td><b>Relationship Rules</b> (Ctrl+Shift+K) — how annotations must relate, for
<b>any dataset</b>: every rule takes any class (or <code>*</code> = any class) and any shape type. One <b>Parent /
part</b> rule covers a whole relationship — wheels on a car, heads on persons, nuclei in cells, text on a sign, any
object inside a region of interest: the part lies inside a parent (share of its area; points and lines: their
points), is <b>not larger than it</b> (long / short side of each, so rotated objects compare right — a part drawn
bigger than its parent is reported with how much bigger), optionally not smaller than a share of it and turned like it
(angle), each parent has min–max parts, and the parts of one parent do not overlap. Shapes are compared by their real
outline where it is convex (boxes, 4-point quads, circles), otherwise by their boxes. Also: <b>Must touch</b> (every A
overlaps a B), <b>Not more than</b> (no more A than B per image), <b>Tilt limit</b>, counts per image, A requires B,
classes never together, allowed classes, no overlap, size and aspect limits, shape type, vertex count, not at the
border, a required attribute, LabelMe groups, one class per image (classification). Same-class duplicates are
<i>Duplicate Shapes</i>' job. <i>Suggest from data…</i> finds the rules that already hold for ≥ 97 % of the dataset
(one Parent / part rule per pair, with the counts and size limit it learned) — tick the ones that describe how the
data <i>should</i> be. <b>Choices from your data:</b> while you add or edit a rule, the form reads the dataset (in
the background, once) and shows under each field what the data does — the likely parent for a part, the class a class
usually touches or appears with, and limits that ~99.5 % of the data already meets (inside share, size relative to
the parent, parts per parent, angle, overlaps, counts, sizes, aspect, tilt, vertices, shape types, attributes) — with
one click to <i>use</i> each value, or <i>Use all suggestions</i>; a pair that does not go together in the data is
called out. <b>Spotting and fixing:</b> each card shows the shape at fault in yellow with a tag of what was measured (IoU,
% inside, size, angle, counts …), what it relates to dashed and tagged, and the proposed fix in green; double-click opens the file framed on all of them with the fix drawn.
<i>Apply proposed fix</i> fits a part inside its parent (keeping its corners and rotation), deletes a stray part or the
smaller of two duplicates, or turns a polygon into a rectangle (and back) — one backup, <i>Undo</i> puts the files
back, fixed files are marked . <i>⚑ Flag files</i> notes the rule and the exact problem
(nothing ticked: every file found). Results: filter by rule (give a rule a short <i>Name</i>); a rule's guideline shows
above its results. Only files that contain a rule's classes are opened. Rules are saved per dataset (older rule files
are converted automatically), can be exported for similar datasets, and are never added twice.</td></tr>
<tr><td>{_i('tool-crop-gallery')}</td><td><b>Crop Gallery</b> — every instance of a label as a grid. The fastest way to spot
wrong labels and junk boxes; sort by size, select many, relabel (R) or delete (Del). Built for millions of instances:
tiles appear while it collects (big datasets use all CPU cores), sorting and Ctrl+A are instant, only visible crops
are decoded, and edits made anywhere refresh just the files they touched.</td></tr>
<tr><td>{_i('tool-overlaps')}</td><td><b>Duplicate &amp; Overlapping Shapes</b> (Ctrl+D) — every pair of shapes worth a
look, any classes, each listed once: <i>Exact copy</i> (identical points, also with two different labels), <i>Marked
twice</i> (IoU ≥ 0.85), <i>Overlap</i> (IoU above your threshold, 0.3 by default) and <i>Inside</i> (a shape almost
entirely inside a bigger one of its class). Copies are grouped — each paired with the group's first shape — so a chain
of copies never lists the same pair twice. A smaller shape of another class inside a bigger one — a plate on a car — is
a part of it and not listed. Each pair is one picture with the rest dimmed: A teal (kept / bigger), B coral (the copy,
dashed, or the smaller shape), the shared area hatched; double-click one and fix it on the image with B selected,
<span class='key'>F8</span> for the next. Pairs that overlap by nature in
your data (a person and their bag): <i>Expected for these classes</i> skips that class pair from then on
(<i>Manage expected…</i> brings it back).</td></tr>
<tr><td>{_i('tool-outliers')}</td><td><b>Outliers</b> — a strip per class shows where its shapes are, the normal range and the
unusual ones as red / amber dots; click a dot to open that shape.</td></tr>
<tr><td>{_i('tool-sequence')}</td><td><b>Sequence Check</b> (Ctrl+Shift+J) — frames from one video / burst checked
against their neighbours. Sequences come from the file names: same folder + the name without its last number
(<code>cam1_frame_000123</code> → <code>cam1_frame_#</code>, frame 123), ordered by that number; where the number jumps
far beyond its usual step (e.g. every 5th frame) a new sequence starts. Names without numbers: “each folder is one
sequence”. Optionally also cut where the picture changes completely. Every object is followed from frame to frame
(overlap, or a close centre and similar size for small / fast objects) and reported with the reason:
<b>missing in this frame</b> (in the frames before and after — drawn as a dashed box; <i>Add missing boxes</i> adds
it, interpolated, backed up), <b>label differs from the track</b> (a short run of another label), <b>size spike</b> /
<b>position spike</b> (only this frame disagrees while both neighbours agree), <b>attribute differs</b> (description /
text / flags), <b>only in this frame</b> (junk or a duplicate?). Works for any dataset with ordered frames.</td></tr>
<tr><td>{_i('tool-dim-area')}</td><td><b>Dim / Area</b> — scan once, then drag a rectangle on the width × height scatter to
select shapes by size (e.g. the cloud of tiny boxes); matching crops appear next to it. <i>Selection…</i> acts on <b>every</b> shape the rectangle (plus Area / Label limits) selects — delete them,
relabel them, or only flag their files. Shapes are re-identified in each file as it is now, so anything edited
since the scan is left alone; the originals are backed up and <i>↶ Undo</i> puts them back.<br>
<b>Region</b> tab — <i>where</i> in the picture: pick an exact image size (e.g. 1920 × 1080; listed with how many
images and shapes have it), then drag a rectangle on a sample image of that size (◀ ▶ for other samples). Only shapes
of images with exactly that size are kept — on a fixed camera the same pixels are the same place. Choose how a shape
must relate to the region (fully inside · centre inside · mostly inside · touching) or tick <i>Outside instead</i> for
everything outside a region of interest. Drag inside to move, drag an edge to resize, type exact x / y / w / h, Esc
or right-click to clear. A heat-map shows where the shapes of the other filters lie; matching boxes are outlined.
The region combines with the size / area / label limits, the size scatter then shows only the region's shapes, and
<i>Selection…</i>, the table and the crops follow it.</td></tr>
<tr><td>{_i('tool-train-size')}</td><td><b>Training Size</b> (Ctrl+Alt+Z) — every object measured the way the detector will
see it: the image shrunk so its longer side equals the training <i>imgsz</i> (like Ultralytics), and then shrunk again
by scale augmentation (default 0.5). <b>Too small</b>: the shorter side at training size is below the limit (8 px =
YOLO's finest stride; text you want read needs about 12–16 px). <b>At risk</b>: fine at training size but too small
once augmentation shrinks the image. Per class: how many objects, the median size, and the <b>imgsz that keeps 95 %
of the class above the limit</b> — with advice (train larger, crop / tile, or SAHI inference). The imgsz defaults to
your model's.</td></tr>
</table>

<h3>{_i('group-models')} Models — see <a href='#models'>section 7</a>.</h3>
<h3>{_i('group-review')} Review — see <a href='#review'>section 8</a>.</h3>

{_tool_map()}
<h2><a name='models'></a>7 · Checking labels with your models</h2>
<p><b>{_i('tool-testing')} Testing</b> (Ctrl+Alt+L, Models tab) measures and compares detection models on this dataset, a
test set built from videos, or videos only: leaderboard, charts, confusion matrix, per class, failures, plots, A / B,
video behaviour and reports. It uses this dataset's models and one prediction cache with Model Check; label fixes made
there go through AIO QC (backed up, in the file's history); <i>Send to review</i> flags failures here. Its chapter is
in the Help Center (Testing).</p>
<p><b>{_i('tool-train')} Train Model</b> (Ctrl+Alt+B, Models tab; no open dataset needed) trains an Ultralytics YOLO
model — detection or segmentation — on a YOLO dataset file (<code>dataset.yaml</code>), in your own Python environment.
The dataset is checked first: counts per split, the label format, val identical to train, classes with no objects,
class ids beyond the names, images Ultralytics skips, and a <code>path:</code> made on another computer (the file's own
folder is used). The environment check lists PyTorch, Ultralytics and the GPUs. Pick a preset (<i>Quick look</i>,
<i>Balanced</i>, <i>Best accuracy</i>) or the model (YOLO26 / YOLO11 / YOLOv8, n – x, for the dataset's task; an
earlier run's best.pt to fine-tune; any .pt or model .yaml); the image size is suggested from the object sizes; batch
<i>Auto</i> fills the GPU's memory; training stops by itself when val stops improving. <i>Augmentation and learning
rate</i>: presets (augmentation <i>Default · Light · Strong · Off</i>; schedule <i>Default · Fine-tuning · Long
run</i>) and every setting by hand — optimizer, start / final learning rate, linear or cosine decay, warm-up, momentum,
weight decay, frozen layers; mirror, upside-down, mosaic (and when it switches off), MixUp, CutMix, copy-paste, rotation,
shift, zoom, shear, perspective, colour. Only changes from Ultralytics' defaults are passed on. <i>Text in the
images</i> (ticked for plate / character classes) never mirrors. <i>Fine-tune from it</i> sets the fine-tuning schedule
(small learning rate, backbone frozen). <i>Extra effects</i>: a recipe built in Image Augmentation (weather, blur, noise …), applied to the training images in memory — no files. <i>More settings</i>: image cache (with the RAM it needs), seed, a time limit,
a share of the images, any other Ultralytics setting as <code>key=value</code> (a misspelt name is caught). Runs go to <code>&lt;dataset&gt;/runs</code>; each keeps going
when the window — or AIO QC — closes, and shows up again (a notification says when it ends). The run on the right:
epoch and batch progress with time left, scores, loss and mAP charts (hover for values; the best epoch marked),
per-class results, the log and its settings. <b>Stop</b> finishes now and keeps the results (best.pt checked, plots
made), <b>Pause</b> stops after the next saved epoch and <b>Resume</b> continues there, <b>Stop now</b> ends at once.
Then <b>Add to Models</b> (Model Check, Testing, Auto-Annotate), <b>Validate…</b> on val / test, <b>Fine-tune from
it</b>. A failed run says what to do (GPU memory, a download, a data-loader worker, the environment …).</p>
<p><b>Classifiers of any architecture</b> — on a class-folder dataset the model list has, besides the YOLO
<code>-cls</code> models, every popular classifier, grouped: <i>small &amp; fast</i> (MobileNetV3 / V4, ShuffleNetV2,
GhostNetV2, EfficientNet-B0, ConvNeXt V2 Atto, FastViT, RepViT, EfficientViT), <i>CNNs</i> (ResNet-18 … 101, ResNeXt,
EfficientNet-B2 / B3, EfficientNetV2-S / M, RegNetY, DenseNet, Inception-v3, ConvNeXt / ConvNeXt V2), <i>transformers</i>
(ViT, DeiT III, Swin, TinyViT, EVA-02) and <i>foundation features</i> for few labelled images — DINOv3 (ViT-S, S+, B, L,
H+, ViT-L for satellite &amp; aerial images, and ConvNeXt Tiny … Large distilled from it), DINOv2 (S, B, L, with
registers), SigLIP 2, CLIP, Perception Encoder, AIMv2 (L, H), NVIDIA's C-RADIOv4 (SO400M, H) and C-RADIOv3 (B, L,
H), and Google's TIPSv2 (B, L, So400m) — each with its parameters and image size — plus <i>any timm
model</i> or <i>any torchvision model</i> by name. <b>Few labelled images</b>: choose a foundation model and the
schedule <i>Few labelled images (linear probe)</i> — the backbone stays as it is, every image goes through it once (and
a mirrored copy), and only the classifier on top learns, on those features: seconds per epoch for any model size, even
ViT-L on a CPU; rare classes drawn as often as common ones. With more images later, fine-tune from that model. The
newest families need a recent timm (DINOv3: 1.0.20, TIPSv2: 1.0.28, C-RADIO: 1.0.31) — <i>Update timm</i> offers it; a few weights come under their own
licence (DINOv3: Meta's DINOv3 License), shown when you choose them; gated weights need your Hugging Face sign-in
(<code>hf auth login</code>, or HF_TOKEN in the environment — AIO QC never stores it). <i>GPU memory saver</i>
(gradient checkpointing) fits large models in less memory when fine-tuning them. They train in the same
environment with AIO QC's own PyTorch trainer: timm when it is installed (<i>Install timm</i> offers it), else
torchvision for the architectures it has; pretrained weights download once. The same run view, Stop / Pause / Resume,
Validate and Add to Models. A flat folder of class folders (no train / val) works too: 15 % of each class validates,
the same images every run, and they stay put as images are added (YOLO classifiers need the split made first).
<b>What a run leaves</b>: <code>weights/best.pt</code> — an Ultralytics classification model (the image normalisation
inside), so Testing, Model Check, the Class Gallery, Model Assist and the Frame Extractor use it like any
<code>-cls</code> model; <code>weights/best.onnx</code> to use it anywhere else; <code>val_predictions.csv</code>
(every validation image, its class, the answer, the confidence — the mistakes first) and
<code>confusion_matrix.csv</code>. <b>Fine-tuning and improving</b> (<i>Augmentation and learning rate</i> ▸
<i>Fine-tuning &amp; improving</i>): the head alone for the first epochs (a linear probe — the pretrained backbone is
not disturbed by the new head's first updates), layer-wise learning-rate decay (earlier layers learn slower), weight
averaging (EMA), label smoothing, stochastic depth, dropout, and <i>class balance</i> — rare classes drawn more often
(<i>Sampler</i>) or weighted in the loss, or <i>Hard examples</i>: the images your trained model gets wrong or is
unsure of are drawn more often. Start from an earlier run's best.pt and its classes are matched to the dataset's
<b>by name</b>: they keep what they learned, new ones start fresh, dropped ones go. Schedule presets
<i>Fine-tuning</i> and <i>Improve my model (hard examples)</i> set all of it. <b>Classifier augmentation</b> (YOLO and
the rest): a random crop (<i>Zoom</i>: how much of the image it may cut), mirror / upside-down, an auto-augment
policy — <i>RandAugment</i>, <i>AutoAugment</i>, <i>AugMix</i>, <i>TrivialAugment</i> or none (then the colour
settings) — and random erasing; the other architectures add rotation, MixUp and CutMix; an Image Augmentation recipe
works on classifiers too, in memory.</p>
<p><b>Extend a model</b> — <i>Goal</i>: <i>my model + more images</i> or <i>my model + new classes</i> (also Improve
Model ▸ <i>Extend it</i>). Start from your model (.pt) and press <i>Plan…</i>: its classes keep their ids; the new
data's classes are matched by name and the rest added at the end — per class <i>add</i>, <i>same as … (merge)</i>,
<i>rename</i> or <i>leave out</i>, with the object counts and which of the model's classes the new labels have. Give
the data it was trained on (found by itself when it was trained here): without it the model sees only the new images
and may forget. The training copy (in the run's <code>data</code> folder: hard links, label files rewritten only where
an id changes) keeps the old validation images apart, and a new image that is a copy of one is left out. The model
labels its own classes on the new images (<i>pre-labels</i>) — with <i>review first</i> the run waits: <i>Open the
training copy</i> (a YOLO dataset of its own; your dataset is never touched), fix or delete what is wrong,
<i>Continue training</i>. Its class-scoring head is enlarged, not reset (it starts out detecting exactly what it did);
stage 1 trains the new head with the backbone frozen, stage 2 everything at a low learning rate. When it ends, every
class is scored before and after on the old and the new validation images, and an old class that dropped more than the
limit is flagged. <code>aioqc/extend.json</code> is the plan; <code>aioqc/README.txt</code> says how to run it on a
training server.</p>
<p><b>{_i('tool-augment')} Image Augmentation</b> (Ctrl+Alt+A, Models tab; no open dataset needed) builds an
augmentation <i>recipe</i> on your own images. <b>Left</b>: a preset or saved recipe, how many effects an image gets
(<i>a few at random</i>, <i>every effect by its chance</i>, or <i>one per image</i>), overall <i>Strength</i>, and every
effect by family — tick it, set its chance and its range; the <b>eye</b> shows that effect alone across its range.
Geometry (mirror, rotate, shear, zoom, shift, perspective, crop, elastic / piecewise / lens warps), colour, weather and
lighting (fog, rain, snow, glare, flare, shadow, low light), blur, noise and camera artifacts, occlusion are built in;
<i>albumentations</i> and <i>imgaug</i> add their own when installed (<i>Install</i>). Labels follow every geometric effect
and are clipped to the image; <i>Labels after geometry</i> sets when an object pushed out loses its label.
<b>Middle</b>: the original and five variations of the image shown in Image QC, with labels, redrawn as you change
things (<span class='key'>R</span> / Shuffle: other draws); each says what was done. <b>Then</b>:</p>
<ul>
<li><b>Train with it (no files)</b> — <i>Use in Train Model</i>: the recipe is applied to every training image in
memory, differently each epoch, so it takes no disk space at any size (Train Model ▸ Augmentation and learning rate ▸
<i>Extra effects</i>; all or a share of the images). Boxes and masks follow; a classifier's whole image is
augmented (YOLO classifiers and every other architecture).</li>
<li><b>Generate files</b> — copies per image (train split only by default), more for rare classes (class folders
too: each copy stays in its class folder), only chosen classes, <i>Paste objects
from other images</i> (cut by their polygon, placed clear of labelled objects), originals <i>linked</i>,
<i>referred to in dataset.yaml</i> (nothing copied) or <i>left out</i>. Runs on every core, can be stopped and
continued (what exists is skipped), and the same seed gives the same files.</li>
</ul>
<p>⋯ next to Save: save as, export, import — also an <code>aug_parameters.yaml</code> of the older imgaug
pipeline.</p>
<p><b>{_i('tool-improve')} Improve Model</b> (Ctrl+Alt+U, Models tab) is the loop around Train Model, one round at a
time (a classifier: the check is Model Check for classification — precision / recall per class over images, top-1
accuracy, the class each one is confused with; a double-click on a class opens it in the Class Gallery, most
suspicious first): <b>1</b> the model — a finished training run (its best.pt) or any .pt; <b>2</b> <i>Run Model Check</i> with just
that model on this dataset — a table of the classes, weakest first, with precision, recall and mAP50-95 (on the val
split when there is one) and <i>what to do</i> in plain words (missed objects → label more of them; confused with
another class → compare them in Crop Gallery; loose boxes → tighter labels; false alarms → unlabelled objects or
negatives); <b>3</b> label more — <i>Flag the hardest images</i> (where model and labels disagree most: wrong or
missing labels are the quickest win) and <i>Review them</i>, or <i>Pick new images from a pool</i> (Preparation ▸ Pick,
set up for this model: varied images it is unsure about, with empty label files so they can be labelled right here,
joining a LabelMe dataset flagged “to label”); <b>4</b> <i>Fine-tune from this model</i> (Train Model with fine-tuning
settings; a LabelMe dataset is converted to YOLO again first). Every check adds a line to <i>Rounds</i> —
precision, recall, mAP50-95 and the weak classes — and says whether the model got better since the round before.
A finished training run offers it too: <i>Improve…</i> in Train Model, and <i>Check it</i> on the toast.</p>
<p><b>{_i('tool-assist')} Model Assist</b> (Edit ▸ Model Assist) — the dataset's models suggest labels on the shown
image: <i>Suggest Labels</i> (Ctrl+Alt+J), or <i>Suggest on Every Image</i>. Suggestions are drawn dashed with the
model's confidence; objects already labelled are not suggested again. <b>Click</b> one to accept it (a real shape,
undo with Ctrl+Z), <b>right-click</b> to drop it, <i>Accept All Suggestions</i> (Ctrl+Alt+Enter) takes every one
above the confidence (<i>Model Assist Settings…</i>). <i>Outline Boxes with SAM</i> (Ctrl+Alt+X): a box you draw —
or a box suggestion you accept — becomes the object's outline (SAM / SAM2 through ultralytics in the models'
environment; the weights download the first time; if SAM is not available the box stays a box and the reason is
shown). Predictions are cached and shared with Model Check, so a page you have checked suggests instantly.</p>
<p><b>{_i('tool-model-setup')} Models &amp; Environment</b> (Ctrl+Alt+N)</p>
<ol>
<li><b>Environment</b>: pick a Python environment that has <code>ultralytics</code> (conda / venv environments are listed
automatically) and press <i>Check</i>. If something is missing, <i>Install into a temporary environment</i> — it can reuse
an existing PyTorch, the install is shown live in the console, and the environment is deleted when the app closes.</li>
<li><b>Models</b>: add one or more model files (e.g. separate plate and line models). Classes are matched to your labels by
name; double-click a model to adjust the mapping. Set <i>Trained on</i> for each model (this dataset, train split only,
held-out, or an image list for k-fold models).</li>
<li><b>Test on current image</b> overlays every model's boxes — a quick sanity check before a long run.</li>
<li>Optional: <i>Export ▸ ONNX / TensorRT</i> for faster inference.</li>
</ol>
<p><b>{_i('tool-model-check')}Model Check</b> (Ctrl+Shift+M) runs the models once; predictions are cached, so every
threshold change afterwards is instant.</p>
<table>
<tr><td><b> Explorer</b></td><td>The whole dataset with your labels (solid) and every model's predictions (dashed)
drawn on each image. Coloured by the match result — green correct, purple wrong class, amber poor fit, red not
labelled, orange missed — or by class / model. <b>Sidebar:</b> each model with its own confidence slider, match IoU,
match results, classes, split / folder / review / training filters, all with live counts. <b>Sort</b> by most errors,
missed labels, unlabelled detections, wrong classes, worst fit, uncertainty or where the models disagree.
<b>Objects</b> shows every object as a crop (labels, predictions or only the problems); <b>Map</b> places the objects
Visual Similarity embedded on a 2-D map coloured by result — drag a rectangle to open them. <b> Thresholds</b>: the
confidence histogram (drag the line) and precision / recall per model and class, with the best-F1 threshold.
Double-click an image: full resolution, hover details, <span class='key'>Enter</span> accepts the model's suggestion,
<span class='key'>A</span> adds a prediction as a label, <span class='key'>R</span> relabels,
<span class='key'>Delete</span> deletes, <i>Compare</i> shows two models side by side. <b>✓ Accept…</b> applies the
suggestions of the selected (or all shown) images in one go; ↶ undoes. Save views (☰) per dataset. The last run comes
back when you reopen Model Check, and <i>View ▸ Model Predictions</i> (Shift+P) draws them in the main viewer.
Segmentation outlines, oriented boxes and keypoints are drawn when the model makes them.</td></tr>
<tr><td><b>Cards</b></td><td>Every finding as a crop — your box solid, the model's box dashed — ranked by how likely it
is a real label error: class mismatch, missing annotation, poor box fit, missed by model. Tick cards, then relabel,
add the predicted box, delete, or flag — only when you choose. (The same list as a table: <i>Findings table</i>.)
<b>Loose or cut-off boxes</b> are <i>poor box fit</i>: your box is compared side by side with the model's, and a side
that sticks out past it (loose) or cuts into it (cut off) by more than <i>fit ±</i> (10 % of the object's size by
default) is a finding — also when the two boxes overlap well. The card names the sides; pick <i>poor box fit</i> in
<i>Issue</i> and <i>Show in file list</i> keeps only those images. <i>Accept</i> takes the model's box, or double-click
and tighten it yourself — click the loose edge, then the <span class='key'>arrows</span> (Shift ×10).
(A model-free estimate was tried and rejected: on real, cluttered images it flagged most correct boxes too.)</td></tr>
<tr><td><b>Analysis</b></td><td>Report card per class, confusion matrix (double-click a cell to see those objects),
error types, box quality (are your boxes systematically looser or tighter?), slices (where recall drops: small objects,
low light …) and the <b>label-error estimate</b>.</td></tr>
</table>
<p class='box'><b>Models trained on this same data?</b> That is the normal case. A model usually reproduces its own training
labels, so where it disagrees the label is often inconsistent. With held-out or k-fold models, use the
<i>Out-of-fold</i> view for the most reliable signal.</p>
<p><b>Label-error estimate</b> (Analysis ▸ Label-error estimate): instead of judging every finding, create an audit sample
(about 20 per class and finding type), mark each <span class='key'>Y</span> (real error) or <span class='key'>N</span>
(model wrong); the app estimates how many labels are wrong per class, with a confidence range.</p>

<p><b>Accept / Reject, one finding after another</b>: <span class='key'>Enter</span> accepts the selected finding —
class mismatch → relabel to the model's class, missing annotation → add the model's box (as a 4-point polygon in
polygon-only YOLO files), poor box fit → replace the box, missed by model → delete the labelled shape — and the next
finding opens. <span class='key'>X</span> rejects it (“the label is right”): it is hidden now and on every later
run, with any thresholds (<i>Show rejected</i> / <i>Forget</i> bring it back). Each accepted change re-finds its shape
by the box it had in the scan (indices may have shifted), is skipped if the shape changed since, is never added twice,
and is backed up — <i>↶ Undo</i> puts the files back.
<i>✓≥ Accept by confidence…</i> accepts every missing-box / class-mismatch / poor-fit finding the model is sure about
in one go (choose the kinds and the minimum confidence, the counts update as you go); rejected findings and deletions
are never included.</p>

<h3>{_i('tool-similar')} Visual Similarity (Ctrl+Shift+Y)</h3>
<p>Finds mislabels by how objects <b>look</b>: each object (each image for classification datasets) becomes a vector
from a vision model, and look-alikes end up close together. Reported: <b>likely mislabel</b> (its closest look-alikes
carry one other label — with the suggested label), <b>atypical for its label</b>, and <b>look-alike, other label</b>
(near-identical objects labelled differently). <i>Use suggested label</i> relabels (backed up, undoable);
<i>Crop Gallery</i> opens that label sorted <i>Least typical first</i> or <i>Grouped by look</i>.</p>
<p><b>Models</b>: add a model <b>folder</b> (Hugging Face format: <code>config.json</code> + weights — DINOv3, DINOv2,
SigLIP / SigLIP2, CLIP; timm and open_clip folders; or the DINOv3 repository with its <code>.pth</code>), a model
<b>zip</b> (extracted once into a models folder you choose — space is checked first — and reused afterwards), or
<b>download</b> a preset (DINOv3 is licence-gated: accept it on huggingface.co and paste a token; it is not stored).
The model runs in your Python environment (the one from Models &amp; Environment; missing packages can be installed
with one click), on the GPU in fp16 when there is one. Small models (DINOv3 ViT-S/16, DINOv2-S, CLIP B/32) are fast;
large ones (DINOv3 ViT-L/16, SigLIP2-L) are better but slower — <i>max / label</i> checks a random sample first.
Vectors are cached per model in <code>.aio_qc/embeddings</code> (float16) and reused as long as the image and the
shape are unchanged, so later runs only embed what changed (relabelling never forces a re-embed).</p>

<h3>{_i('search')} Search in Words (Ctrl+Alt+W)</h3>
<p>The same models, used with words — local, nothing leaves the computer. Needs a <b>text-aware</b> model: SigLIP2
(best), SigLIP or CLIP (<i>Download…</i> offers them; DINO models understand images only).</p>
<p><b>Search in words</b>: type what you are looking for — “truck at night in rain”, “blurred plate”, “trailer roof
with a hole” — and get the images (or the labelled objects) that match best, best first. The summary says how many
matched per split next to the split's size, so a condition that only appears in test stands out. <i>Only images
WITHOUT a '…' label</i> finds what is probably unlabelled: search “a person”, keep the images without a 'person'
label. The first run embeds the images (shared with Visual Similarity); every search after that is instant — only the
words are embedded. <i>Review in grid</i>, <i>Flag files</i> and <i>Show in file list</i> work on the results.</p>
<p><b>Check labels against class descriptions</b>: write one line per class (<i>Class descriptions…</i> — e.g.
<i>tear</i>: “a long rip in a trailer roof”) and every label is scored against all of them. Reported: <b>looks like
another class</b> (with the class, <i>Use suggested label</i> relabels — backed up, Undo) and <b>unlike its class
description</b>. <i>Class agreement…</i> shows per class how often the model agrees with the labels and which class it
mistakes it for, and descriptions the model reads as nearly the same — a low share usually means the description
should say what sets the class apart. Works on the first day of a dataset, before any model is trained. Descriptions
are saved with the dataset (<code>.aio_qc/class_descriptions.json</code>) and also sent to Claude Review.
Scores are a ranking, not a verdict: look at the results.</p>

<h3>{_i('tool-ai-review')} Claude Review (Ctrl+Alt+V)</h3>
<p>Claude looks at each image with its labels drawn and numbered, the label list, your class descriptions and
written guidelines, and says what is wrong, in words: <b>missing object</b>, <b>wrong class</b>, <b>box too loose</b>,
<b>box cuts the object off</b>, <b>labelled twice</b>, <b>text does not match</b> (reads plates and container
numbers), <b>guideline not followed</b>, <b>nothing there</b>. Each finding has the shape, a suggestion, a
confidence and one sentence why.</p>
<p><b>Images</b>: this image (opening the window on a shown image starts it — a few seconds), selected files, flagged
files, the files in the list, or all. <b>How</b>: now (several requests at a time) or as a <b>batch</b> — half the
price, results in the background (usually within the hour; this window checks every minute, batches survive closing
the app). A cost estimate is shown before more than one image is sent. Results are kept per image in
<code>.aio_qc/claude_review.sqlite</code>: an unchanged image is never sent twice, and after a fix only that image is
checked again.</p>
<p><b>Apply</b> makes the edit for wrong classes (relabel), missing objects (adds the suggested box) and labels on
nothing / labelled twice (removes the label) — backed up, <b>Undo</b>, the files flagged for a look. Box-fit and
guideline findings are fixed on the image (Claude's boxes are approximate). <b>Dismiss</b> hides a finding for good.
Your accept / dismiss choices are counted per finding type, so you see which types to trust.</p>
<p><b>Setting it up</b>: (1) an Anthropic API key — console.anthropic.com ▸ API keys (billing set up on the
account); (2) put it in an environment variable — Windows: <code>setx ANTHROPIC_API_KEY "sk-ant-…"</code>, macOS /
Linux: <code>export ANTHROPIC_API_KEY="sk-ant-…"</code> in ~/.zshrc or ~/.bashrc — and restart AIO QC; (3)
<i>Connection…</i> ▸ <i>Test connection</i>. A company gateway that speaks the same API can be set as the server. AIO
QC stores only the variable's <b>name</b>, never the key; it is not written to any file, log or report. On the first
run you are asked whether this dataset's images may be sent (kept per dataset; <i>Connection…</i> changes it).
Findings are suggestions — nothing is changed without Apply.</p>
<p>{_i('terminal-spark')} <b>Or use your Claude plan</b> — <i>Use: Claude Code on this computer</i>. Install Claude
Code once (Windows PowerShell: <code>irm https://claude.ai/install.ps1 | iex</code> or <code>winget install
Anthropic.ClaudeCode</code>; macOS / Linux: <code>curl -fsSL https://claude.ai/install.sh | bash</code>), run
<code>claude</code> in a terminal and log in with your Claude account (Pro, Max, Team or Enterprise). AIO QC then runs
that Claude Code for each image — it never sees or stores your sign-in, and leaves API-key variables out so your plan
is what is used. <i>Connection…</i> ▸ <i>Test Claude Code</i> checks it. Plans are meant for ordinary, individual use,
so this route is live only, a few requests at a time and at most 25 images per run (you are offered the first 25);
for whole datasets and half-price batches use an API key. Each review also counts Claude Code's own instructions
against your plan's limits. When Claude Code is signed out or the plan's limit is reached, the run stops with its
message.</p>

<h3>{_i('tool-ocr')} OCR Check (Ctrl+Shift+T)</h3>
<p>Reads the text inside your <b>text classes</b> (e.g. <code>lp_num_line</code>, <code>trailer_text</code>) with an OCR
engine from your own Python environment — PaddleOCR, RapidOCR, EasyOCR or Tesseract (<i>Check</i> lists what is
installed and tests it). It finds boxes with <b>no text</b> or <b>low confidence</b>, text that breaks your
<b>rules</b> (validators), <b>truncated</b> or <b>loose</b> boxes and <b>several lines</b> in a single-line box (mode
<i>Read + locate</i>), differences from a <b>stored transcription</b>, and <b>identity leakage</b> — the same plate or
container ID in train and val. OCR runs once per box and is cached; thresholds and validators re-check instantly.</p>
<p><b>Languages</b>: tick one or several (English, Hindi, Tamil, Arabic, Chinese, Japanese …; the first is the primary).
EasyOCR and Tesseract read several languages in one pass; with PaddleOCR / RapidOCR choose <i>Best of all languages</i>
(most confident read wins) or <i>Primary first, others only when unsure</i> (faster on large datasets). Languages an
engine cannot read are greyed out. The Summary shows which language each read came from.</p>
<p><b>Validators</b> (optional, any one must pass): pick a built-in one (container ID with check digit, Indian plate,
VIN, barcode, ISBN, IBAN, date …), type a pattern like <code>AA 00 A{{1,2}} 0000</code>, paste correct examples and let
AIO QC learn the pattern, or write <code>validate(text)</code> in Python — test it live and on the dataset's reads first.</p>

<h2><a name='review'></a>8 · Reviewing and signing off</h2>
<table>
<tr><td>{_i('review-ok')} {_i('review-flag')} {_i('review-fixed')}</td><td><b>Review marks</b>: <span class='key'>S</span> OK, <span class='key'>X</span>
flag, <span class='key'>Shift+X</span> flag with a note, <span class='key'>U</span> clear, <span class='key'>N</span> next
unreviewed. The next file opens automatically (toggle in the Review tab). Marks are saved in the dataset.
The status bar shows the open file's mark — {_i('review-pending')} <b>Pending</b> (not reviewed yet), OK, Flagged or
Fixed, with a note icon when the flag has a note (the note is in its tooltip).
Files corrected in an editor become <b>Fixed</b> — check them and press S to mark them OK.
At the end of the list marking stays on the last file. With a review filter (e.g. Unreviewed) the marked file leaves
the list but stays open, and Next continues with the file that came after it.
<i>Review ▸ Save Flagged Files…</i> copies the flagged files (annotation + image, folder structure kept, with a
notes CSV) to a folder, <b>or</b> saves a list of them as CSV (file, image, status, note, date — the default) or TXT,
named <i>&lt;dataset&gt;_flagged</i>.</td></tr>
<tr><td>{_i('next')}</td><td><b>Recommended checks</b> — when a dataset opens, the app works out which checks matter
most for it (files that cannot be read, broken images, label names that disagree or are not in the class list, empty
files, splits, names that look like video frames, classes that lie inside one another, texts, models set up) and
shows <i>Start here</i> with the reason in a slim banner above the image. The status bar's <b>Next check</b> button runs the next one; after each check
finishes, the next is suggested (“and so on”). <i>Review ▸ Recommended Checks…</i> lists them all, best first, with
what the finished ones found (kept per dataset; <i>Start over</i> clears it). Settings ▸ Interface ▸ <i>Recommend
checks when a dataset opens</i> (on by default) turns the hints off.</td></tr>
<tr><td>{_i('tool-inbox')}</td><td><b>Issues Inbox</b> (Ctrl+I) — every tool's findings in one place. Each file gets a
risk score (errors weigh most; one noisy tool cannot dominate); filter by severity, tool and review state, and press
<i>Review worst first</i> to walk the file list in risk order. A rescan replaces that tool's findings.</td></tr>
<tr><td>{_i('tool-convert')}</td><td><b>Convert &amp; Split</b> (Ctrl+Shift+X; Ctrl+Alt+Y opens it on YOLO) —
one window that writes this dataset into another folder, in any format, as one set, with its own splits or with a new
balanced split; the dataset itself is not changed. A <b>side pane</b> on the left walks through the steps — Output,
Classes, Split, Files, Options (Alt+1 … Alt+5) — and shows what is chosen in each; a dot marks a step that needs a look
(a folder that can't be used, no class ticked, names that still clash). Under it, <i>What will be written</i> sums it
up live. Ctrl+Enter converts.<br>
<b>Output</b> — the format: YOLO detection / segmentation (suggested from the shapes; from LabelMe, YOLO or COCO),
LabelMe, COCO, segmentation masks (palette PNG with the class index + 16-bit instance PNG + labelmap.txt), Pascal VOC
XML, CVAT for images 1.1 XML, YOLO OBB, DOTA, YOLO pose (each box with the points of its group; kpt_shape and, from left
/ right names, flip_idx in dataset.yaml), crops of every shape as a classifier dataset, image flags as class folders;
for classification datasets class folders, LabelMe flags or a CSV table. The folder is checked as you type (new,
empty, an earlier run to replace, not empty, inside the dataset); its default name follows the format
(<code>&lt;dataset&gt;_yolo</code>, <code>_coco</code> …, <code>_split</code> for the dataset's own format). Images are
hard-linked (no extra space) or copied.<br>
<b>Classes</b> — every format writes this list: tick the classes, drag them into order (the order is the class id;
<i>Load order from file…</i> takes a model's <code>data.yaml</code> or <code>classes.txt</code>; a YOLO or COCO
dataset starts in its own id order). Right-click a class (or F2) to write it under another name — an existing class
name merges them. A classifier's classes work the same way.<br>
<b>Split</b> — one set, the dataset's own splits, or a <b>new balanced split</b>: val / test %, seed and the rare limit;
duplicates and look-alikes, frames of one video and one OCR identity (plates, container IDs — after an OCR Check scan)
stay in one split; features seen in ≤ 3 images stay in train; every class is spread like the target. On a dataset of up
to 30,000 images the split is planned as soon as you open this step; <i>Preview split</i> shows the whole split as one
bar, one bar per class (dashed = the target) and a per-feature table. <code>reports/</code> gets the split report,
per-feature summary and a feature map of every image.<br>
<b>Files</b> — sub-folders kept as in the dataset (the default) or <b>flattened</b>: every image straight in
images/, files with the same name renamed with the naming scheme (checked before it starts, listed in renamed.csv);
YOLO: images trainers can't read re-encoded as PNG.<br>
<b>Options</b> — leave out flagged and broken images, keep or leave out images without a written class (background),
join the pieces of one object (YOLO). <i>More options</i>: merge labels that differ only in case or spaces, leave out
images with a class that is not written; YOLO also: objects smaller than N px², images with a broken shape, <b>parts
inside a whole</b> (e.g. a text line must lie on a plate — orphans dropped or their image left out), read every label
back.<br>
Circles, oriented boxes and masks convert; lines and points are counted but not written as boxes. The label mapping is
applied. Every run writes a report (<code>conversion_report.txt</code> / <code>export_report.txt</code>) and a small
manifest: converting into the same folder again replaces only what the last run wrote, and <i>Undo</i> removes it —
anything else in the folder stays. <i>Open in AIO QC</i> opens the result; <i>Train a model…</i> opens Train Model on a
YOLO result or on class folders. <b>Large datasets</b>: annotations are read once; when the files wait on storage (a
network share, large copies) reading and copying run on several threads, otherwise in one, where Python is fastest; a
million images never sit in memory as a million pending jobs.</td></tr>
<tr><td>{_i('tool-sampling')}</td><td><b>QC Sampling</b> — is the dataset good enough without checking every file? It picks a
random sample of the right size (rare classes included), you review it with S / X, and it shows the estimated error rate
with a <b>PASS / FAIL</b> verdict against your limit.</td></tr>
<tr><td>{_i('tool-editor')}</td><td><b>Fix in Editor</b> (Ctrl+Alt+O) — open the flagged files (or a selection:
right-click in the file list) in <b>labelme</b> 5.x or 6.x (any dataset, all shapes) or <b>labelImg</b> (YOLO boxes).
<b>Save image data in the JSON (imageData)</b> — unticked by default: changed files are written with
<code>"imageData": null</code> (small files; the image is read from imagePath); ticked: the image is embedded the way the
selected labelme version does it. Same key in labelme 5 and 6, and a JSON whose only copy of the image is embedded
always keeps it. The editor itself always works without image data, so a session costs a few KB per file. The first time, the editor is installed into its own environment
(progress is shown). Every save in the editor is checked (labels must be in the class list; shapes must fit the
dataset type), written back to the real file with your imagePath and settings kept, and the file becomes
{_i('review-fixed')} <b>Fixed</b>. Originals go to the compact backup store — <i>Undo session</i> restores them
byte-for-byte. Files changed outside
the session are never overwritten (conflict → you choose). New editor versions: <i>Add source zip…</i>.
The session window's <b>minimise</b> button shrinks the session into a small panel inside AIO QC (top right,
picture-in-picture) with every file and its status: drag it by its header, resize it from the corner, double-click a
file to open it, and use its restore button {_i('restore-window')} (or double-click the panel) to bring the full window
back.</td></tr>
<tr><td>{_i('tool-review-grid')}</td><td><b>Review Grid</b> (Ctrl+Shift+W) — whole images with their annotations, a
page at a time (2 × 2 … 6 × 5). Look at the page, press <span class='key'>X</span> on the few bad ones
(<span class='key'>Shift+X</span> with a note), then <span class='key'>Space</span>: every tile you did not mark becomes
OK and the next page opens. Arrows move, <span class='key'>S</span> OK, <span class='key'>U</span> clear,
<span class='key'>Enter</span> next page, <span class='key'>Backspace</span> previous, double-click opens the image in the
viewer. Files: unreviewed in the current list, the current list, flagged / fixed, everything, or a Saved View. The
list is fixed when the grid opens (marking never reshuffles pages); images are decoded at tile size and the next
page is prepared while you look.</td></tr>
<tr><td>{_i('tool-saved-filter')}</td><td><b>Saved Views</b> (Ctrl+Shift+B) — a question about your files, saved:
conditions on labels and their counts, the number of objects or of different labels, review state and note, split,
folder, file name (text or regular expression), Issues-Inbox findings (a tool, a minimum risk), unreadable files, or a
fixed list of files ( <i>Current list</i> pins whatever the file list shows, e.g. a Dim / Area result). Join them with
ALL or ANY, negate any of them. The view explains itself in one sentence (“Files with at least 3 'plate' AND not
reviewed yet”) and warns about labels / splits this dataset does not have. Counts stay live: they use the label index,
review marks and check results already in memory, so a view over a million files is counted in about a second — in the
background — after every change. <i>Show in file list</i> or <i>Review in grid</i>; <i>Export</i> / <i>Import</i>
reuse views on similar datasets (labels matched by name).</td></tr>
<tr><td>{_i('tool-report')}</td><td><b>QC Report</b> (Ctrl+Alt+R) — one click, one self-contained file (HTML that
opens in any browser, or PDF): summary, class balance per split, review coverage, findings by tool, the latest QC
sample estimate, the riskiest files with thumbnails of their annotations, and <b>what changed since the last report</b>
(edits, review marks, merges, outside changes, and how the headline numbers moved). Built in the background.</td></tr>
<tr><td>{_i('tool-coverage')}</td><td><b>Coverage Map</b> — every file as a square (grey unreviewed, green OK, amber flagged),
grouped by split or folder: see at a glance what has been QC'ed; click a square to open the file.</td></tr>
<tr><td>{_i('tool-smart-filter')}</td><td><b>Smart Filter</b> — build a subset to review (e.g. files with 3+ plates but no line).
Classification datasets: by class — images with cat AND dog, with two or more classes, not bird.</td></tr>
</table>
<p><i>Review ▸ Export review report</i> writes a CSV of every file's state and notes.</p>
<p><b>{_i('tool-presets')} QC Presets</b> (Ctrl+Alt+Q, Review tab) — the same checks, the same way, on every
delivery. A preset is a list of checks (Validation, Text Labels, Outliers, Model Check …), each with its window's
settings (taken when it is added, or later with <i>Take settings from its window</i>) and a limit: <i>passes with at
most N files</i>, counting every finding or only errors. <i>Run preset</i> runs them one after the other in the
background (Stop works between and inside checks) and writes a report: <b>PASS / FAIL</b>, a line per check and,
per file, what to fix — HTML for people, CSV for spreadsheets (<code>.aio_qc/qc_runs/</code>). <i>Flag the failing
files</i> marks them with the checks that found them, ready for <i>Hand-off batch…</i>. <i>Export…</i> / <i>Import…</i>
share a preset with the team.</p>

<h2><a name='safety'></a>9 · What changes your files (and how to undo)</h2>
<ul>
<li><b>Nothing changes without a confirmation.</b> Views, statistics, mappings (view-only) and display adjustments never
touch files.</li>
<li>Edits (relabel, delete shapes, fixes, class removal) rewrite annotation files <b>atomically</b> — a crash cannot leave a
half-written file. Images are only rewritten by the EXIF rotation fix (Image Quality / Preparation), which keeps every
original first.</li>
<li>“Deleted” files are <b>moved</b> to <code>&lt;dataset&gt;/.aio_qc_trash/</code> (File ▸ Open Trash Folder) — move them back
to restore.</li>
<li>Every rewrite that can be undone (Fix in Editor, Dim / Area, class removal, …) first stores the original in
<code>.aio_qc/backups.sqlite</code>: compressed, and a LabelMe JSON's embedded image is kept by reference when the
image file can rebuild it (checked by SHA-256) — typically a few KB per file instead of a full copy. Restores are
byte-for-byte.</li>
<li>Every change is also written to the file's <b>History</b> (<code>.aio_qc/history.sqlite</code>, a few dozen bytes
per change): Ctrl+H shows it and reverts to any saved version.</li>
<li>All app data lives in <code>&lt;dataset&gt;/.aio_qc/</code>: review marks, label mapping, model setup, samples, audits,
exports, backups and the cache. Deleting that folder resets the app for this dataset (your annotations are not in
it).</li>
<li><b>Tools ▸ Storage…</b> shows what that folder holds — cache (per model / OCR setting), backups (per operation:
export the originals or delete), editor sessions, old-style backup folders, trash — and the retention: backups older
than 30 days, finished sessions after 14 days and model / OCR results unused for 90 days are removed automatically
after indexing (all adjustable; 0 = keep). Old backup folders from earlier versions are compacted automatically,
without losing anything.</li>
</ul>

<h3>Changes made outside AIO QC</h3>
<p>While a dataset is open, AIO QC notices files that other programs change, add or remove (another annotation
tool, a script, a file sync). It checks every few seconds and immediately when you switch back to AIO QC, waits until a
file has finished being written, and then <b>re-reads</b> it — it never writes anything in response. Changed files
are re-indexed (the current image refreshes in place), new files appear in the file list, deleted ones disappear, and
a notification says what happened (<i>Details</i> lists every file). Open QC tools show “results may be outdated ·
Rescan” instead of re-running on their own. If the class list or the split folders change, you are asked to reload.
Turn it off in Settings ▸ Dataset ▸ Watch the dataset for outside changes.</p>

<h3>Provenance — where every annotation, dataset, image and model came from</h3>
<p>Everything AIO QC writes says what made it: a manual edit, a model (its weights file name, a fingerprint of the
weights and the score), an import, a conversion, a rule (a batch fix, a resize, a renamed label) or an augmentation —
with the feature, the app version and the time. <b>No person is ever recorded</b>, no folder path (paths carry user
names), no setting that looks like a key, token or password. Settings ▸ General ▸ Record provenance turns it off.</p>
<ul>
<li><b>Each annotation</b> (LabelMe): every shape a feature adds or changes gets a <code>"provenance"</code> record —
an id, how it was made, and a short history of changes (what changed: label, geometry, type, group, description,
flags). Shapes nobody touched are left exactly as they are. LabelMe keeps these keys when it opens and saves the
files. Hover a shape in the Annotations panel to read it (“Made by a model · yolo11n.pt (0.91) · Model Assist ·
2026-11-20 — Changed 2×, last: geometry”). YOLO label files have no room for it: each save adds a line to
<code>.aio_qc_label_edits.jsonl</code> in the dataset folder. COCO annotations carry the record too.</li>
<li><b>Each folder AIO QC writes or changes</b> (Convert &amp; Split, Preparation runs, Image Augmentation, hand-off
batches, Frame Extractor outputs, training runs) gets a hidden <code>.aio_qc_provenance.json</code>: one record per
run — the action and feature, the settings, counts, a fingerprint of the input and of the output (file count, size
and a SHA-256 of the file list), and the id of the input's own last record. Following those ids leads from a trained
model back to the raw data.</li>
<li><b>Images AIO QC makes</b> (extracted frames, crops, masks, augmented and resized images, re-encoded images,
screenshots) carry the record inside the file — XMP (readable by exiftool, image viewers and asset managers) plus
EXIF / PNG Software — without re-encoding the pixels. A frame names its video, frame number and time. Images that are
linked or copied from your dataset are never touched.</li>
<li><b>Trained models</b>: the run folder's record names the training data (fingerprint and record id), the classes,
the settings and the starting model with its SHA-256; when training ends, best.pt (and best.onnx) get the record
inside and the weights' SHA-256 is added.</li>
</ul>

<h2><a name='team'></a>10a · Team work</h2>
<table>
<tr><td>{_i('tool-handoff')}</td><td><b>Export Hand-off Batch</b> (Ctrl+Alt+E) — send files to an annotator: flagged,
flagged + fixed, the files in the list, or the riskiest in the Issues Inbox. The batch is a folder in the dataset's own
layout (YOLO images / labels paired, LabelMe imagePath pointing at the copied image) with your notes and the findings
(<code>notes.csv</code>), <code>_hints/</code> images with the flagged regions drawn, <code>GUIDELINES.html</code> and a
README. <code>batch.json</code> remembers a fingerprint of every annotation as it was sent. <b>Class folders:</b> the
batch keeps the class folders (train/cat/…) — the annotator <i>moves</i> a wrongly classed image into the right folder
(a new folder for a new class); Import Batch turns every move into a class change (the files move, one Ctrl+Z).</td></tr>
<tr><td>{_i('import')}</td><td><b>Import Hand-off Batch</b> (Ctrl+Alt+I) — pick the returned folder or zip. Every file
is compared three ways (as sent · as returned · the dataset now): <b>ready</b> (only the annotator changed it),
<b>conflict</b> (changed here too — you choose), unchanged, already merged, not returned, or invalid (broken JSON, a YOLO
class id outside the class list, a different image size). Each row says what the annotator changed (“2 added (plate) ·
1 relabelled (car → truck)”) and what changed here. Double-click shows the returned shapes over the image. Merge backs
every file up first, keeps the dataset's imagePath / imageData, records it in the file's History and marks it Fixed.</td></tr>
<tr><td>{_i('tool-guidelines')}</td><td><b>Guidelines</b> (Ctrl+Alt+G) — the labelling rule for each class, each
relationship rule and each check that finds labelling mistakes, with ✓ good and ✗ bad example crops (the shape's own
outline drawn on them). Add examples from the image (select a shape, <span class='key'>G</span> /
<span class='key'>Shift+G</span>, or right-click it in Annotations) or from the Crop Gallery. The rule appears in a card
next to the selected shape and at the top of the matching QC tool; <i>Export for annotators</i> writes one HTML page.
</td></tr>
<tr><td>{_i('history')}</td><td><b>History</b> (Ctrl+H) — everything that happened to the open file: when, who, which
tool, what (edits, review marks, merges, moves, changes made outside AIO QC). <i>Saved versions</i> are the originals
kept before each rewrite: <i>Preview</i> draws that version's shapes over the image and lists the differences;
<i>Revert</i> restores it exactly — after storing the current file, so a revert can be reverted.</td></tr>
<tr><td>{_i('tool-splits')}</td><td><b>Split Health</b> (Dataset tab, Ctrl+Alt+S) — every class in every split with
enough examples, val / test drifting from train (Jensen–Shannon divergence), splits that are too small, and
<b>leakage</b>: near-identical images in different splits. Suggested moves fix it (leaking groups reunited, val / test
topped up from train without starving train); <i>Apply</i> moves the files between the split folders (layout kept,
review marks follow), <i>Undo last moves</i> puts them back.</td></tr>
</table>

<h3>Sketch and ROI capture (Tools menu)</h3>
<table>
<tr><td>{_i('screenshot')}</td><td><b>Screenshot</b> (<span class='key'>E</span>, the toolbar button, File ▸
Screenshot, or right-click the image) — saves the image exactly as you see it: the part on screen at the current
zoom, brightness / contrast, the labels you ticked, the predictions layer and a sketch drawing, at your screen's real
resolution, cropped to the image (no grey bands). The hover highlight, edit handles and floating panels are left
out. <span class='key'>E</span> saves straight into the screenshots folder (Settings ▸ General ▸ Saving; default
<i>Pictures/AIO QC Screenshots</i>) as <code>&lt;dataset&gt;/&lt;sub-folders&gt;/&lt;file name&gt;.png</code> and never
overwrites (<code>_2</code>, <code>_3</code> …); <span class='key'>Shift+E</span> asks for a name; the right-click menu
also copies it to the clipboard. Works in sketch mode too.<br><b>Many files at once:</b> <i>Review ▸ Flagged
Screenshots</i> (Ctrl+Shift+Alt+E), or right-click files in the file list ▸ <i>Save screenshots of N files</i> — each
file drawn as Image QC shows it, in the background (progress and Cancel in the status bar; your view is not touched):
the same spot as now on every file (default) or the whole image at full resolution, with a caption strip (file,
review mark, flag note) and each file's sketch. Defaults in Settings ▸ Screenshots.<br>
<b>Auto-sketch</b> — point at the shapes that matter, on every screenshot, without drawing: <i>Choose shapes…</i> shows the
files of the batch one by one; click the shapes on the picture (or tick them; <span class='key'>D</span> /
<span class='key'>A</span> next / previous file, <i>All</i> / <i>None</i>). The marks are drawn live as they will be saved:
an arrow from the emptiest side (it avoids every other shape, the other arrows and the image edges), with a ring, a
box, or neither — the <i>Mark</i> style, colour and <i>Label text at the arrow</i> apply to the whole batch. On the
screenshot only the chosen shapes are drawn (the others are hidden), as the whole image so every mark is in the
picture; files with no chosen shapes are saved as usual. Choices are kept per dataset
(<code>.aio_qc/autosketch.json</code>) and follow their shapes when a file is edited later.<br>
<b>Text at the arrows</b>: <i>Text</i> — a template (or anything you type) written at every arrow, with
<code>{{label}}</code> the shape's label, <code>{{n}}</code> its number, <code>{{file}}</code> the file, <code>{{date}}</code> today;
empty for no text. <i>S / M / L / XL</i> — its size (M follows the arrow, which follows the image). Double-click a shape in
the list to give just its arrow its own text (e.g. “Loose box” on one, “Wrong label” on another).
<i>Templates…</i> edits the list, shared with Sketch's Text tool. The <i>Look</i> next to it styles every arrow's text, its colours
worked out for each text on its own image.</td></tr>
<tr><td>{_i('tool-sketch')}</td><td><b>Sketch on Image</b> (<span class='key'>P</span>) — draw on the image to explain
something: pen, see-through highlighter, a shapes box as in MS Paint (line, arrow, rectangle, ellipse, circle —
two at a time, scroll it sideways; Shift: 45° / square / circle), text, an
eraser with its own size and a colour picker (dropper: click the image to take its colour, then back to your last
tool, as in MS Paint); ten colours plus any colour, size with <span class='key'>[</span> / <span class='key'>]</span>
(the cursor shows the real brush at the current zoom), filled shapes, Ctrl+Z / Ctrl+Y, clear. Entering it unticks every
label so the image is clean — tick the labels you want in the Labels panel; leaving it (Esc, P or Done) restores your
choice. <i>Copy</i> (Ctrl+C) and <i>Save</i> give the image at full resolution with exactly what you see. Drawings are
kept per image in <code>.aio_qc/sketches/</code> and return when you sketch on that image again. Zoom and Space-pan work
as usual.<br><b>Text tool</b>: the size slider sets the letter height in image pixels (<i>Aa 40 px</i>; <span class='key'>[</span> /
<span class='key'>]</span>); a click opens the text window — click a template (fields filled in for the shape under the
click: <code>{{label}}</code>, <code>{{n}}</code>, <code>{{file}}</code>, <code>{{date}}</code>) or type, and set the size. Click a
text you placed to change its words or size (Ctrl+Z undoes the change). <i>Edit templates…</i> adds, edits and
re-orders the templates (shared with auto-sketch). <i>Look</i> — the style and colours: outlined, on a chip, with a
shadow or plain; colours fixed (Alert white on red, Note black on yellow, Info white on blue), the sketch colour, or
<i>for the background</i>: chosen to stand out from the part of the image the text lands on. The preview shows the
text on that very part of the image as it will be drawn; text near an edge is moved so none of it is cut off.<br>
While Sketch (or ROI) is on, its tools take the place of the QC drawer's tab row — the same strip, so nothing covers
the image and the image does not move; Done / Esc brings the tabs back. (Mouse wheel over the tools scrolls them
sideways in a narrow window.)</td></tr>
<tr><td>{_i('tool-roi')}</td><td><b>ROI Capture</b> (<span class='key'>O</span>) — drag boxes like a snipping tool:
crosshair cursor, the rest of the image dims, live <i>x1, y1 → x2, y2 · w × h</i>. Coordinates are real image pixels
(sub-pixel when zoomed in), rounded to 2 decimals and kept inside the image. Each image keeps its boxes
(<code>.aio_qc/rois.json</code>) while you move through the dataset; right-click a box removes it, Esc removes the last.
<i>Copy</i> (Ctrl+C) puts the image name and its boxes on the clipboard; <i>Export</i> writes every image's boxes as CSV,
TXT (name x1 y1 x2 y2) or JSON; <i>Crops</i> saves each box as a PNG.</td></tr>
</table>

<h3>{_i('tool-autoannotate')} Auto-Annotate Videos (Tools ▸ Auto-Annotate Videos, Ctrl+Shift+U)</h3>
<p>A folder of videos (or images) → unique frames → one LabelMe file per frame drawn by your model, in five steps you
tick or leave out: <b>Frames</b> (every Nth or the sharpest of every N, repeats skipped within and across videos),
<b>Similar frames</b> (set aside to <code>similar/</code>), <b>Annotate</b> (.onnx or .pt; classes and image size read
from the model file; confidence, IoU, per-class minimum confidence, renaming; LabelMe 6.3.1 or 5.11.4; imageData off
by default), <b>Keep rule</b> (frames without the classes you need → <code>extra_images/</code>) and <b>Package</b>
(folders by video or by a name pattern, parts of N frames, zipped; hard links or copies). Frames only is a plain batch extraction; any later
step runs alone on what the run folder holds. <b>AI annotate</b> (Annotate ▸ Who annotates) needs no trained model: a
vision model — LiteLLM, OpenAI, Gemini, Claude, Bedrock, Ollama … or one in your Python environment — annotates from
plain-word class descriptions and a few examples, with checks, a budget and a readiness check first; YOLO txt by
default, LabelMe on request. <i>Try on 4 frames</i> shows the boxes before a run; Stop is always safe
and Run continues; a saved setup runs unchanged from the command line (<code>aio-qc autoannotate</code>). Some local
models (Florence-2, Moondream, several chat models) ship code that runs when they load — never allowed by default:
tick <i>Allow this model to run its own code</i> under the model, once, only for a model from a source you trust (it
is remembered for that model on this computer, never taken from a shared setup file). The Frame Extractor's guide
(F1 there) has the details.</p>

<h3>Preparation — raw data to dataset (QC drawer ▸ Dataset; Ctrl+Alt+P)</h3>
<p>The first button of the Dataset tab, since preparing the folders comes before checking them. Works on folders, with
or without a dataset open. Every tool: <i>Preview</i> (a plan — counts and the files
concerned; nothing changes) → <i>Run</i> (background, stoppable) → summary, CSV log, <i>Undo This Run</i>. Runs are
logged in <code>&lt;folder&gt;/_aioqc_prep/</code>; the <i>Undo a run</i> page reverses any of them (moved files go
back, created files are removed only if unchanged, rewritten JSONs get their original bytes back).</p>
<table>
<tr><td>{_i('tool-prep')}</td><td><b>Unzip archives</b> — every .zip under a folder (nested zips too) into a folder
named after it; already-extracted ones skipped; corrupt, password-protected and unsafe archives (paths leaving the
folder, zip bombs) reported.<br>
<b>Combine &amp; flatten</b> — one or more nested folders → one flat LabelMe folder. Each JSON finds its image by
imagePath, the same name in its folder, or a unique name anywhere; images that exist only as imageData are restored
(fully verified); imageData → null; byte-identical images are added once (also against what the flat folder already
holds) — the other copies go to <i>orphans/duplicates</i> with their JSON and the plan lists copies whose labels
<b>differ</b>; images without JSON and JSONs without image go to <i>orphans/</i>; different images with the same name
get a name from the naming scheme (<i>Same names</i>: folder prefix by default, any preset or your own pattern — see
Name Clashes). Copy (sources untouched) or move (empty source folders removed).<br>
<b>Combine YOLO datasets</b> — several YOLO datasets (images/ and labels/ mirrored, any nesting, with or without
train / val / test) → one: the class lists merged by name (the first dataset's order kept, new classes at the end),
every label file rewritten with the new ids, names made unique with the naming scheme (renamed.csv), exact duplicate
images once (the plan says whether their labels agree), hard links — the sources never change.<br>
<b>Add new data</b> — new image + annotation pairs into an EXISTING dataset (LabelMe, or YOLO detection / segmentation
with or without train / val / test), matched by the <i>image</i>, not the file name: an image the dataset already has
is recognised however it was renamed or moved — left alone (and listed when its annotation differs), its annotation
replaced, or merged (the classes it lacks). Only images whose byte size also occurs on the other side are ever read,
so a dataset of millions is listed, not read; an image repeated in the new data counts once. The new data — LabelMe or
YOLO, any nesting, several folders — is converted to the dataset's format, class ids by name (letter case and spaces
ignored, and a <i>same class, other name</i> map); classes the dataset lacks are added to its class list
(classes.txt and data.yaml kept in step, comments kept), dropped, or the image left out. New images go to train, a
balanced split (like the dataset's own shares or yours — frames of one video together) or the new data's own splits;
LabelMe datasets keep the new data's sub-folders or take one folder. A name the dataset already uses for a different
image is renamed by the naming scheme (number suffix by default; folder or path prefix, renumbering, content hash, your
pattern or your saved presets — every name checked against the dataset's). Every new image is decoded first
(optional); copies or hard links; a report of every decision (…_report.csv beside the run log). Run it again with
more data: what is already there is recognised. Also on the command line: <code>aio-qc add-data</code> (asks for the
folders when not given; <code>--dry-run</code>, <code>-y</code>, <code>-h</code> for every option).<br>
<b>Add classes by image</b> — bring the annotations of more classes to the images a dataset already has, matched by
image: e.g. a LabelMe master with every class and a YOLO dataset converted with a few — the YOLO labels get the other
classes (by default every class the dataset lacks, or the ones you list), its class list extended. Any direction:
LabelMe ↔ YOLO, detection ↔ segmentation (boxes and polygons converted). A class both have in an image (a partly
labelled class): keep the dataset's objects, replace them, or add only the source's objects that do not overlap
(IoU). Every copy of an image the dataset has twice gets them (and the plan warns when one image sits in two splits).
<b>The dataset images the source does not have are listed</b> — they would lack the added classes, which teaches a
model those objects are background there. Command line: <code>aio-qc add-classes</code>. Undo restores every
annotation and class list.<br>
<b>Find orphans</b> — in place: unmatched JSONs / images to sibling folders, structure kept.<br>
<b>Integrity check</b> — every image fully decoded (truncated JPEGs, empty files, wrong types), every JSON checked as
LabelMe (shapes, numeric points, labels, imageWidth / imageHeight against the real image) → quarantine, with partners.<br>
<b>Exact duplicates</b> — byte-identical images anywhere in a tree; kept: annotated &gt; original name &gt; oldest;
the rest set aside with their JSON.<br>
<b>Resize images</b> — downscale into a new folder (longest side, height rule or factor; JPEG / PNG / keep, quality);
LabelMe points and sizes scaled, YOLO labels copied.<br>
<b>Import annotations</b> — the format is detected by itself. Main formats: COCO JSON (boxes, polygons, RLE masks
traced to polygons, keypoints as points grouped with their box), segmentation masks (a folder of PNGs: class index,
colour per class with a labelmap.txt, or instance masks with their class masks; regions traced to polygons), Pascal VOC
XML (a folder) and CVAT XML (images and video tracks; rotated boxes, ellipses, masks; attributes become flags /
description; tags become image classes). Also: Label Studio JSON (boxes incl. rotated, polygons, points, ellipses,
brush masks, choices), KITTI label files, VGG Image Annotator JSON (choose the region attribute that holds the class),
a Supervisely project folder (bitmaps traced), CreateML JSON, Open Images CSV (with class-descriptions for names) and
Datumaro JSON; and classification tables (CSV / TSV / TXT: file + class, several in one cell, one 1 / 0 column per
class, "path label" lists, a split column, JSON Lines such as Hugging Face's <code>metadata.jsonl</code>). → a NEW folder as LabelMe, YOLO detection / segmentation or a
classification dataset (class folders or LabelMe flags). Images are found by path, then by name, then by the name
without an upload prefix or extension, and hard-linked (no extra space) or copied.<br>
<b>Fix EXIF rotation</b> — in place: the EXIF rotation baked into the pixels and LabelMe / YOLO labels brought into
the same upright frame; per file the label size tells whether they were drawn on the raw or the upright picture.
Every file is kept first (Undo restores it).<br>
<b>Pick images to label</b> — from a folder of unlabelled images, the set worth labelling first: each pick is the
image least like everything picked so far (so near-identical frames are not picked twice), optionally unlike the open
dataset's images and weighted by how unsure its model is. Compared by built-in image features (fast, no model) or the
Visual Similarity model. Copied or moved to a folder with <code>picked.csv</code> in rank order — label from the
top. Two picks with one name in one folder (class folders gather images from everywhere), or a name an earlier pick
left there, get their folder as a prefix.</td></tr>
</table>
<p>In a dataset, <b>Duplicate Images</b> now keeps the best copy first (train &gt; annotated &gt; original name &gt; most
pixels &gt; largest file &gt; oldest) and says for every copy whether its labels match the kept one. <b>Export &amp;
Split ▸ Class folders</b> writes an image into every class it is flagged with and decodes images that exist only as
imageData.</p>

<h3>{_i('mode-video')} {_i('mode-data')} Three modes: Image QC, the Frame Extractor, Data Studio (Ctrl+Alt+M)</h3>
<p>AIO QC is one app with three modes, each its own window. The switch in the top-right corner of the menu bar (or
<span class='key'>Ctrl+Alt+M</span>, next mode) changes to the <b>Frame Extractor</b> — play videos, mark and extract
frames for annotation (batch, markers, timeline analysis, search, clip and annotated-video export, its own command line
<code>aio-qc extract | probe | testset | eval | clip | autoannotate</code>); <i>File ▸ Open output in Image QC</i> brings the extracted
frames here as a dataset, and <i>File ▸ Show Source Video in Frame Extractor</i> goes back from an image to its video.
See <i>Help ▸ Modes &amp; working across tools</i>. Or to <b>Data Studio</b>: its own window, menus and settings for table data — CSV, Excel, Parquet, JSON, SQLite.
Clean, reshape, build features and chart a table step by step (every step undoable and shown as pandas / PySpark
code), see each column's health with one-click fixes, and run the recipe on the full data with Spark. <i>Tools ▸
Annotations → Data Studio</i> opens this dataset as a table — one row per shape. Data Studio has its own guide
(F1 there). The app starts in Image QC.</p>

<h3>Keyboard-only work</h3>
<table>
<tr><td>{_i('keyboard')}</td><td><span class='key'>?</span> (or <span class='key'>Ctrl+F1</span>, Help ▸ Keyboard Shortcuts) opens every shortcut and mouse
gesture over the window,
the keys for where your focus is first; type to filter, click (or Enter) to run. <span class='key'>F6</span> /
<span class='key'>Shift+F6</span> move the focus between panels (the panel flashes);
<span class='key'>Alt+1 … Alt+6</span> open a QC tab with the focus on its first tool — arrows move, Enter runs, Esc
closes.</td></tr>
<tr><td>{_i('get-started')}</td><td><b>Help ▸ Quick Tour</b> — the short tour shown when the first dataset
opens, one part of the real window at a time. Settings ▸ General ▸ Animations turns all motion off.</td></tr>
<tr><td>{_i('restore-window')}</td><td><b>Minimise</b> — a feature window (any check, Search in Words, Claude
Review, Preparation, Train, Testing, the Crop Viewer, Settings …) minimises into the <b>shelf</b>: a row of chips at
the bottom-left of the main window. Its job keeps running and the chip shows the progress, then <i>✓ done</i> with a
short pulse; click the chip — or open the feature again from its menu or shortcut — and the window comes back where it
was; × on a chip closes the window (asking first while a job runs). Whatever minimises a feature window (Win+Down, a
desktop's own button with the system title bar) ends in the shelf, so a window is never lost. The main window
minimises to the taskbar / Dock; on Windows a click on the taskbar button minimises and restores it (Settings ▸
General ▸ <i>Windows taskbar and window animations</i>). On macOS <span class='key'>Cmd+M</span> minimises the window
in front (Label Mapping is <span class='key'>Cmd+Shift+Option+M</span> there). When a job finishes while AIO QC is
minimised or in the background, the taskbar button flashes / the Dock icon bounces.</td></tr>
</table>

<h2><a name='large'></a>10 · Working with large datasets</h2>
<ul>
<li>The first open indexes every file; later opens reuse the cache and only re-read changed files.</li>
<li>Run expensive checks (Model Check, Duplicate Images) on a <b>QC sample</b> first: create the sample, then choose
<i>Scope: files in current list</i>.</li>
<li>Filter the file list (search, split, review state, QC results) and review with D / S / X — keyboard only.</li>
<li>Crop Gallery sorted <i>Smallest first</i> finds junk boxes in thousands of objects in seconds.</li>
</ul>

<h3>Logs</h3>
<p>Everything is logged to <code>aio-qc.log</code> (next to where AIO QC was started; rotated at 10 MB): what you
open, what tools find, Qt's own warnings, and how the app ended — closed by you, Ctrl+C / SIGTERM in the terminal (AIO QC
saves and closes gracefully; press Ctrl+C twice to force), or a crash with its full traceback. A native crash (e.g. in a
graphics driver) leaves <code>aio-qc-crash.log</code> with the stack, and the next start points to it.</p>

<h2><a name='keys'></a>11 · Essential shortcuts</h2>
<table>
<tr><td class='key'>D / A</td><td>next / previous image</td><td class='key'>S / X / U</td><td>mark OK / flag / clear</td></tr>
<tr><td class='key'>H / V</td><td>Drag / Select mode</td><td class='key'>N</td><td>next unreviewed</td></tr>
<tr><td class='key'>Ctrl+wheel</td><td>zoom at cursor</td><td class='key'>Ctrl+F / Ctrl+1</td><td>fit / 100 %</td></tr>
<tr><td class='key'>Tab</td><td>next shape</td><td class='key'>Esc</td><td>let go of the handle / clear selection</td></tr>
<tr><td class='key'>F8 / Shift+F8</td><td>next / previous finding of the check used last</td><td class='key'>arrows</td><td>move the shape — or, after a click on an edge / corner, just that side</td></tr>
<tr><td class='key'>Ctrl+P</td><td>search files</td><td class='key'>Ctrl+M</td><td>label mapping</td></tr>
<tr><td class='key'>Ctrl+K</td><td>command palette</td><td class='key'>F11</td><td>focus mode (image only)</td></tr>
<tr><td class='key'>Del · 1–9 · L</td><td>delete (asks first) / relabel the selected shape</td><td class='key'>Ctrl+R</td><td>create a new bbox (drag it)</td></tr>
<tr><td class='key'>Ctrl+N</td><td>create a new polygon: click its corners; click the first corner (it lights up), Enter or double-click closes it</td><td class='key'>Esc</td><td>stop drawing</td></tr>
<tr><td class='key'>Ctrl+Z / Ctrl+Shift+Z</td><td>while drawing a polygon: take the last corner back / put it back</td><td class='key'>Shift</td><td>while drawing a polygon: edges at 45° steps</td></tr>
<tr><td class='key'>Ctrl+Z</td><td>undo an edit on the image</td><td class='key'>Ctrl+Shift+Z</td><td>redo</td></tr>
<tr><td class='key'>?</td><td>key map</td><td class='key'>F6</td><td>next panel</td></tr>
<tr><td class='key'>P</td><td>sketch on the image</td><td class='key'>O</td><td>ROI capture</td></tr>
<tr><td class='key'>E</td><td>save a screenshot of the view</td><td class='key'>Shift+E</td><td>screenshot as …</td></tr>
<tr><td class='key'>Ctrl+H</td><td>file history</td><td class='key'>G / Shift+G</td><td>guideline example ✓ / ✗</td></tr>
<tr><td class='key'>F1</td><td>this guide</td><td class='key'>? · Ctrl+F1</td><td>all shortcuts</td></tr>
</table>

<h2><a name='trouble'></a>12 · Troubleshooting</h2>
<p><b>Errors say what happened, why, and what fixes it.</b> A window that cannot do something tells you in one
sentence, gives the likely cause below it (the original message is under <i>Show Details…</i>), and offers the fix
when there is one: <i>Open Models &amp; Environment</i>, <i>Install it</i>, <i>Try again</i>, <i>Show the log</i>.
<b>Long jobs</b> look the same everywhere: a progress bar with the time left, Stop (what is done so far is kept), an
entry in Background Tasks (the ring in the status bar), and at the end one line with Undo or Open folder where they
fit.</p>
<table>
<tr><td>{_i('logs')}</td><td><b>Something failed?</b> Open <i>Logs</i> (Ctrl+Shift+L) — every error of the session is listed there.</td></tr>
<tr><td>{_i('undo')}</td><td><b>View looks odd?</b> <i>Reset View</i> (R) shows all labels, clears filters and fits the image;
<i>Display ▸ Reset</i> removes image adjustments.</td></tr>
<tr><td>{_i('delete')}</td><td><b>Stale numbers?</b> Settings ▸ Dataset ▸ Dataset cache, then reopen.</td></tr>
<tr><td>{_i('tool-model-setup')}</td><td><b>Model problems?</b> Models &amp; Environment ▸ Check environment, and read the console.
Missing packages can be installed into a temporary environment.</td></tr>
</table>
<p class='muted'><a href='#toolmap'>Which tool for what?</a> (section 6b) — every tool on one page.</p>
"""
