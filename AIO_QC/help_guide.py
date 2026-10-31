# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
In-app user guide (Help ▸ User Guide, F1; shown once on first start).
HTML for QTextBrowser; <img src='aio-icon:NAME'> images are provided by the caller as document resources.
"""


def _i(name, size=18):
    return f"<img src='aio-icon:{name}' width={size} height={size}>"


GUIDE_ICONS = ["open", "list", "prev", "next", "hand", "cursor", "fit-window", "zoom", "brightness-contrast",
               "group-dataset", "group-labels", "group-shapes", "group-models", "group-review", "review-ok",
               "review-flag", "tool-stats", "tool-validation", "tool-empty", "tool-image-dups", "tool-ambiguity",
               "tool-label-map", "tool-remove-class", "tool-crop-gallery", "tool-dup-shapes", "tool-outliers",
               "tool-dim-area", "tool-model-setup", "tool-model-check", "tool-sampling", "tool-smart-filter", "tool-coverage", "tool-editor", "review-fixed", "tool-ocr", "restore-window", "tool-quality", "tool-rules", "tool-inbox", "tool-export",
               "logs", "delete", "undo", "edit", "tool-sequence", "tool-similar", "tool-review-grid", "tool-saved-filter",
               "history", "tool-dataprep", "tool-sketch", "tool-roi", "tool-handoff", "import", "tool-report", "tool-guidelines", "tool-splits", "keyboard",
               "get-started"]


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
<table>
<tr><td>{_i('open')}</td><td><b>1. Open a dataset</b> — toolbar <i>Open Dir</i> (Ctrl+O), or drag a folder onto the window.
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
<code>train/ val/ test/</code>) and <b>LabelMe flags</b> (JSONs whose true flags are the image's classes). The class is
shown as a chip over the image; <i>Review ▸ Change Class…</i> (C) gives the selected images another class — for class
folders the image is moved into the other folder (its review mark and findings follow it). Tools that need shapes are
disabled; Stats, Duplicate Images (images in two classes = a conflict), Image Quality, Rules (one class per image),
Sampling, the Inbox and Export &amp; Split work as usual. LabelMe-flag datasets can be corrected in labelme (Fix in
Editor offers the classes as flags).</p>

<h2><a name='window'></a>4 · The window</h2>
<table>
<tr><td><b>Toolbar</b></td><td>Open / Load Classes, previous / next image, Drag / Select mode, zoom, reset, logs.
Icons only by default (names in the tooltips); View ▸ Compact Toolbar switches the text back on.</td></tr>
<tr><td><b>QC panel</b> (top)</td><td>A slim tab row: click a tab — or rest the mouse on the tabs for a moment — and
the tools slide down <b>over</b> the image (the image keeps its size). The drawer is the same size for every tab, so
switching tabs (click, or rest on another tab) just swaps the tools in place. It slides back up when you launch a
tool, press Esc, click elsewhere, click the open tab again, or keep the mouse away from it for a moment. 📌 (or double-clicking a
tab, or View ▸ Pin QC Panel) keeps them open under the tabs. One tab per question: {_i('group-dataset')} Dataset, {_i('group-labels')} Labels,
{_i('group-shapes')} Shapes, {_i('group-models')} Models, {_i('group-review')} Review, and {_i('brightness-contrast')} Display
(view-only brightness / contrast / gamma / auto-contrast, fill opacity, outline width, label size). Class labels keep
one readable size at every zoom and are placed so they never cover each other.</td></tr>
<tr><td><b>File List</b></td><td>Search (Ctrl+F), split / folder / review filters (All / Unreviewed / OK / Flagged);
long names scroll sideways. A yellow banner
appears when a QC tool has narrowed the list — <i>Clear</i> shows everything again. Red names = unreadable files,
grey = no shapes.</td></tr>
<tr><td><b>Annotations</b></td><td>The shapes of the current image: shape type drawn in its label colour. Click to select,
click again to deselect, double-click to zoom to it, right-click to rename its label.</td></tr>
<tr><td><b>Labels</b></td><td>Every label in the dataset with its count; untick to hide a label, right-click for
“show files with this label”, “show only this label”, rename / mapping.</td></tr>
<tr><td><b>Thumbnails</b></td><td>Thumbnails around the current image (vertical when docked left / right).</td></tr>
<tr><td><b>Status bar</b></td><td>Review state of the current file, indexing progress, position, mode, active label mapping.</td></tr>
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
<tr><td>{_i('fit-window')}</td><td><b>Zoom</b>: Ctrl+wheel at the cursor, Ctrl+0 fit, Ctrl+1 100 %.
<b>Double-click</b> toggles between fit and a close-up at the click; double-clicking a shape (Select mode) zooms to it.</td></tr>
<tr><td>{_i('brightness-contrast')}</td><td><b>Display tab</b>: make dark or hazy images readable without touching the files
(the tab shows <i>Display •</i> while an adjustment is active).</td></tr>
<tr><td>{_i('edit')}</td><td><b>Fix it right here</b> (Select mode, select a shape): drag its <b>handles</b> to move a
corner / edge / vertex, drag the shape to move it, <span class='key'>arrows</span> nudge by one image pixel
(<span class='key'>Shift</span> ×10), <span class='key'>Ctrl</span>+click on an outline adds a vertex,
<span class='key'>Alt</span>+click on a vertex removes it, <span class='key'>Del</span> deletes the shape,
<span class='key'>1</span>–<span class='key'>9</span> relabel to class 1–9 of the class list (shown in the status bar),
<span class='key'>L</span> / <span class='key'>F2</span> relabel from the list, <span class='key'>B</span> then drag draws a new
box. <span class='key'>Ctrl+Z</span> / <span class='key'>Ctrl+Shift+Z</span> undo / redo (Edit menu).<br>
Every shape type (rectangle, polygon, circle, line, linestrip, point) in LabelMe and YOLO (detect, segment, OBB): a
rectangle stays a rectangle, a circle keeps its centre, polygons never drop below 3 points, YOLO 4-point polygons
keep 4 vertices (they may be oriented boxes), pose boxes are not reshaped here (their keypoints would not follow),
YOLO labels stay within the class map, untouched YOLO rows stay byte-identical. Each edit re-reads the file and is
refused if the file changed on disk meanwhile or the shape is no longer the one shown; the original is backed up
once, and undo restores the exact bytes. A flagged file you edit becomes {_i('review-fixed')} Fixed.</td></tr>
</table>
<p><span class='key'>Esc</span> always clears the selection.</p>

<h2><a name='workflow'></a>6 · The QC workflow, tab by tab</h2>
<p>Work left to right: fix files first, then names, then shapes, then use models, then sign off.
Every tool opens as a window you can keep open while browsing; <b>double-click a result</b> to open that file and shape,
<b>tick</b> rows to act on them, <i>Show in file list</i> to review the results one by one, <i>CSV</i> to export.</p>
<p>{_i('review-flag')} <b>Flag from anywhere</b>: every tool has <i>⚑ Flag files</i>, and right-clicking a result row, card
or tile offers Flag / Flag with note / Mark OK / Clear. The note records which tool flagged it and why; the mark shows
up in the file list, thumbnails, Coverage Map and as a badge in every tool.</p>

<h3>{_i('group-dataset')} Dataset — are the files sound?</h3>
<table>
<tr><td>{_i('tool-stats')}</td><td><b>Dataset Stats</b> — counts per label and split, colour-coded balance, and an image profile
(resolution, aspect, lighting …). Double-click any row to list its files.</td></tr>
<tr><td>{_i('tool-validation')}</td><td><b>Validation</b> — broken / missing files, wrong class ids, invalid geometry. Rows with a
<i>Proposed fix</i> can be fixed after you review each change (<i>Review fixes…</i>).</td></tr>
<tr><td>{_i('tool-quality')}</td><td><b>Image Quality</b> (Ctrl+Shift+Q) — <b>errors</b>: corrupt or truncated files
(“Corrupt JPEG data: … extraneous bytes before marker”, “Premature end of JPEG file”, broken PNG chunks), images that
cannot be decoded, EXIF rotation that disagrees with the annotation (the boxes would appear rotated in tools that apply
EXIF), annotation size ≠ image size. <b>Information only</b>: blurry, under- / over-exposed and low-contrast images
(adjustable thresholds). Every image is also checked automatically in the background after a dataset opens: broken
files turn red in the file list, a notification offers <i>Show</i>, and a red banner appears over a broken image when
you open it.</td></tr>
<tr><td>{_i('tool-empty')}</td><td><b>Empty Annotations</b> — files without shapes. For YOLO these are often intentional background images.</td></tr>
<tr><td>{_i('tool-image-dups')}</td><td><b>Duplicate Images</b> — identical and near-identical images, and the same scene in
several splits (leakage). <i>Keep one per group</i> ticks the extra copies.</td></tr>
</table>

<h3>{_i('group-labels')} Labels — are the label names right?</h3>
<table>
<tr><td>{_i('tool-ambiguity')}</td><td><b>Label Ambiguities</b> — spelling variants, stray spaces, unknown labels.
<i>Fix in Label Mapping</i> hands the suggestions over.</td></tr>
<tr><td>{_i('tool-label-map')}</td><td><b>Label Mapping</b> (Ctrl+M) — the one place to rename or merge labels: either
<b>view-only</b> (files untouched; everything in the app uses the new names) or <b>written to files</b>.</td></tr>
<tr><td>{_i('tool-remove-class')}</td><td><b>Remove / Re-order Classes</b> — delete a class everywhere; for YOLO also renumber
ids and update the class file. Always preview first; backups are made by default.</td></tr>
</table>

<h3>{_i('group-shapes')} Shapes — are the annotations right?</h3>
<table>
<tr><td>{_i('tool-rules')}</td><td><b>Relationship Rules</b> (Ctrl+Shift+K) — how annotations must relate: a child
inside its parent (a line inside its plate), a parent containing 1–2 children, counts per image, A requires B, A and B
never together, allowed classes, no overlap / children apart, size and aspect limits, shape type, vertex count, not at the
border, a required attribute, LabelMe groups, one class per image (classification). <i>Suggest from data…</i> finds the
rules that already hold for ≥ 97 % of the dataset — tick the ones that describe how the data <i>should</i> be, and the
files that break them are listed. Rules are saved per dataset and can be exported for similar datasets.</td></tr>
<tr><td>{_i('tool-crop-gallery')}</td><td><b>Crop Gallery</b> — every instance of a label as a grid. The fastest way to spot
wrong labels and junk boxes; sort by size, select many, relabel (R) or delete (Del). Built for millions of instances:
tiles appear while it collects (big datasets use all CPU cores), sorting and Ctrl+A are instant, only visible crops
are decoded, and edits made anywhere refresh just the files they touched.</td></tr>
<tr><td>{_i('tool-dup-shapes')}</td><td><b>Duplicate Shapes</b> — the same object annotated twice, shown as side-by-side
pairs (kept | duplicate); tick a pair and delete either side.</td></tr>
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
</table>

<h3>{_i('group-models')} Models — see <a href='#models'>section 7</a>.</h3>
<h3>{_i('group-review')} Review — see <a href='#review'>section 8</a>.</h3>

<h2><a name='models'></a>7 · Checking labels with your models</h2>
<p><b>{_i('tool-model-setup')} Models &amp; Environment</b> (Ctrl+Alt+M)</p>
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
<p><b>{_i('tool-model-check')} Model Check</b> (Ctrl+Shift+M) runs the models once; predictions are cached, so every
threshold change afterwards is instant.</p>
<table>
<tr><td><b>Cards</b></td><td>Every finding as a crop — your box solid, the model's box dashed — ranked by how likely it
is a real label error: class mismatch, missing annotation, poor box fit, missed by model. Tick cards, then relabel,
add the predicted box, delete, or flag — only when you choose. (The same list as a table: <i>Findings table</i>.)</td></tr>
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
and is backed up — <i>↶ Undo</i> puts the files back.</p>

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
Files corrected in an editor become <b>Fixed</b> — check them and press S to mark them OK.
At the end of the list marking stays on the last file. With a review filter (e.g. Unreviewed) the marked file leaves
the list but stays open, and Next continues with the file that came after it.
<i>Review ▸ Save Flagged Files…</i> copies the flagged files (annotation + image, folder structure kept, with a
notes CSV) to a folder, <b>or</b> writes their names to a text file.</td></tr>
<tr><td>{_i('tool-inbox')}</td><td><b>Issues Inbox</b> (Ctrl+I) — every tool's findings in one place. Each file gets a
risk score (errors weigh most; one noisy tool cannot dominate); filter by severity, tool and review state, and press
<i>Review worst first</i> to walk the file list in risk order. A rescan replaces that tool's findings.</td></tr>
<tr><td>{_i('tool-export')}</td><td><b>Export &amp; Split</b> (Ctrl+Shift+X) — the clean, trainable copy. Keep the existing
splits or build new ones: val is required (default 10 %), test optional. Duplicates, frames of the same video and the
same OCR identity stay in one split; features seen in ≤ 3 images stay in train; every class is spread like the target.
The preview shows the whole split as one bar and one bar per class (dashed lines = the target), plus a per-feature
table. Formats: YOLO detection / segmentation, LabelMe, COCO, class folders; images are hard-linked (no extra space);
reports as in the training pipeline. The label mapping is applied; flagged and broken files can be left out.</td></tr>
<tr><td>{_i('tool-sampling')}</td><td><b>QC Sampling</b> — is the dataset good enough without checking every file? It picks a
random sample of the right size (rare classes included), you review it with S / X, and it shows the estimated error rate
with a <b>PASS / FAIL</b> verdict against your limit.</td></tr>
<tr><td>{_i('tool-editor')}</td><td><b>Fix in Editor</b> (Ctrl+E) — open the flagged files (or a selection:
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
fixed list of files (📌 <i>Current list</i> pins whatever the file list shows, e.g. a Dim / Area result). Join them with
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
<tr><td>{_i('tool-smart-filter')}</td><td><b>Smart Filter</b> — build a subset to review (e.g. files with 3+ plates but no line).</td></tr>
</table>
<p><i>Review ▸ Export review report</i> writes a CSV of every file's state and notes.</p>

<h2><a name='safety'></a>9 · What changes your files (and how to undo)</h2>
<ul>
<li><b>Nothing changes without a confirmation.</b> Views, statistics, mappings (view-only) and display adjustments never
touch files.</li>
<li>Edits (relabel, delete shapes, fixes, class removal) rewrite annotation files <b>atomically</b> — a crash cannot leave a
half-written file. Images are never modified.</li>
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
Turn it off in View ▸ Watch Dataset for Outside Changes.</p>

<h2><a name='team'></a>10a · Team work</h2>
<table>
<tr><td>{_i('tool-handoff')}</td><td><b>Export Hand-off Batch</b> (Ctrl+Alt+E) — send files to an annotator: flagged,
flagged + fixed, the files in the list, or the riskiest in the Issues Inbox. The batch is a folder in the dataset's own
layout (YOLO images / labels paired, LabelMe imagePath pointing at the copied image) with your notes and the findings
(<code>notes.csv</code>), <code>_hints/</code> images with the flagged regions drawn, <code>GUIDELINES.html</code> and a
README. <code>batch.json</code> remembers a fingerprint of every annotation as it was sent.</td></tr>
<tr><td>{_i('import')}</td><td><b>Import Hand-off Batch</b> (Ctrl+Alt+I) — pick the returned folder or zip. Every file
is compared three ways (as sent · as returned · the dataset now): <b>ready</b> (only the annotator changed it),
<b>conflict</b> (changed here too — you choose), unchanged, already merged, not returned, or invalid (broken JSON, a YOLO
class id outside the class list, a different image size). Each row says what the annotator changed (“2 added (plate) ·
1 relabelled (car → truck)”) and what changed here. Double-click shows the returned shapes over the image. Merge backs
every file up first, keeps the dataset's imagePath / imageData, records it in the file's History and marks it Fixed.</td></tr>
<tr><td>{_i('tool-guidelines')}</td><td><b>Guidelines</b> (Ctrl+Alt+G) — the labelling rule for each class and each QC
rule, with ✓ good and ✗ bad example crops. Add examples from the image (select a shape, <span class='key'>G</span> /
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
<tr><td>{_i('tool-sketch')}</td><td><b>Sketch on Image</b> (<span class='key'>P</span>) — draw on the image to explain
something: pen, see-through highlighter, a shapes box as in MS Paint (line, arrow, rectangle, ellipse, circle —
two at a time, scroll it sideways; Shift: 45° / square / circle), text, an
eraser with its own size and a colour picker (dropper: click the image to take its colour, then back to your last
tool, as in MS Paint); ten colours plus any colour, size with <span class='key'>[</span> / <span class='key'>]</span>
(the cursor shows the real brush at the current zoom), filled shapes, Ctrl+Z / Ctrl+Y, clear. Entering it unticks every
label so the image is clean — tick the labels you want in the Labels panel; leaving it (Esc, P or Done) restores your
choice. <i>Copy</i> (Ctrl+C) and <i>Save</i> give the image at full resolution with exactly what you see. Drawings are
kept per image in <code>.aio_qc/sketches/</code> and return when you sketch on that image again. Zoom and Space-pan work
as usual.</td></tr>
<tr><td>{_i('tool-roi')}</td><td><b>ROI Capture</b> (<span class='key'>O</span>) — drag boxes like a snipping tool:
crosshair cursor, the rest of the image dims, live <i>x1, y1 → x2, y2 · w × h</i>. Coordinates are real image pixels
(sub-pixel when zoomed in), rounded to 2 decimals and kept inside the image. Each image keeps its boxes
(<code>.aio_qc/rois.json</code>) while you move through the dataset; right-click a box removes it, Esc removes the last.
<i>Copy</i> (Ctrl+C) puts the image name and its boxes on the clipboard; <i>Export</i> writes every image's boxes as CSV,
TXT (name x1 y1 x2 y2) or JSON; <i>Crops</i> saves each box as a PNG.</td></tr>
</table>

<h3>Data Prep — raw data to dataset (Tools ▸ Data Prep, Ctrl+Alt+P)</h3>
<p>Works on folders, with or without a dataset open. Every tool: <i>Preview</i> (a plan — counts and the files
concerned; nothing changes) → <i>Run</i> (background, stoppable) → summary, CSV log, <i>Undo This Run</i>. Runs are
logged in <code>&lt;folder&gt;/_aioqc_prep/</code>; the <i>Undo a run</i> page reverses any of them (moved files go
back, created files are removed only if unchanged, rewritten JSONs get their original bytes back).</p>
<table>
<tr><td>{_i('tool-dataprep')}</td><td><b>Unzip archives</b> — every .zip under a folder (nested zips too) into a folder
named after it; already-extracted ones skipped; corrupt, password-protected and unsafe archives (paths leaving the
folder, zip bombs) reported.<br>
<b>Combine &amp; flatten</b> — one or more nested folders → one flat LabelMe folder. Each JSON finds its image by
imagePath, the same name in its folder, or a unique name anywhere; images that exist only as imageData are restored
(fully verified); imageData → null; byte-identical images are added once (also against what the flat folder already
holds) — the other copies go to <i>orphans/duplicates</i> with their JSON and the plan lists copies whose labels
<b>differ</b>; images without JSON and JSONs without image go to <i>orphans/</i>; name clashes become _1, _2 ….
Copy (sources untouched) or move (empty source folders removed).<br>
<b>Find orphans</b> — in place: unmatched JSONs / images to sibling folders, structure kept.<br>
<b>Integrity check</b> — every image fully decoded (truncated JPEGs, empty files, wrong types), every JSON checked as
LabelMe (shapes, numeric points, labels, imageWidth / imageHeight against the real image) → quarantine, with partners.<br>
<b>Exact duplicates</b> — byte-identical images anywhere in a tree; kept: annotated &gt; original name &gt; oldest;
the rest set aside with their JSON.<br>
<b>Resize images</b> — downscale into a new folder (longest side, height rule or factor; JPEG / PNG / keep, quality);
LabelMe points and sizes scaled, YOLO labels copied.</td></tr>
</table>
<p>In a dataset, <b>Duplicate Images</b> now keeps the best copy first (train &gt; annotated &gt; original name &gt; most
pixels &gt; largest file &gt; oldest) and says for every copy whether its labels match the kept one. <b>Export &amp;
Split ▸ Class folders</b> writes an image into every class it is flagged with and decodes images that exist only as
imageData.</p>

<h3>Keyboard-only work</h3>
<table>
<tr><td>{_i('keyboard')}</td><td><span class='key'>?</span> opens the <b>key map</b> over the window — every shortcut,
the keys for where your focus is first; type to filter, click (or Enter) to run. <span class='key'>F6</span> /
<span class='key'>Shift+F6</span> move the focus between panels (the panel flashes);
<span class='key'>Alt+1 … Alt+6</span> open a QC tab with the focus on its first tool — arrows move, Enter runs, Esc
closes.</td></tr>
<tr><td>{_i('get-started')}</td><td><b>Help ▸ Get Started</b> — the guided overview shown on the first start, with
<i>Show me</i> spotlights on the real window. View ▸ Animations turns all motion off.</td></tr>
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
<p>Everything is logged to <code>aio_qc_tool.log</code> (next to where AIO QC was started; rotated at 10 MB): what you
open, what tools find, Qt's own warnings, and how the app ended — closed by you, Ctrl+C / SIGTERM in the terminal (AIO QC
saves and closes gracefully; press Ctrl+C twice to force), or a crash with its full traceback. A native crash (e.g. in a
graphics driver) leaves <code>aio_qc_crash.log</code> with the stack, and the next start points to it.</p>

<h2><a name='keys'></a>11 · Essential shortcuts</h2>
<table>
<tr><td class='key'>D / A</td><td>next / previous image</td><td class='key'>S / X / U</td><td>mark OK / flag / clear</td></tr>
<tr><td class='key'>H / V</td><td>Drag / Select mode</td><td class='key'>N</td><td>next unreviewed</td></tr>
<tr><td class='key'>Ctrl+wheel</td><td>zoom at cursor</td><td class='key'>Ctrl+0 / Ctrl+1</td><td>fit / 100 %</td></tr>
<tr><td class='key'>Tab</td><td>next shape</td><td class='key'>Esc</td><td>clear selection</td></tr>
<tr><td class='key'>Ctrl+F</td><td>search files</td><td class='key'>Ctrl+M</td><td>label mapping</td></tr>
<tr><td class='key'>Ctrl+K</td><td>command palette</td><td class='key'>F11</td><td>focus mode (image only)</td></tr>
<tr><td class='key'>Del · 1–9 · L</td><td>delete / relabel the selected shape</td><td class='key'>B</td><td>draw a new box</td></tr>
<tr><td class='key'>Ctrl+Z</td><td>undo an edit on the image</td><td class='key'>Ctrl+Shift+Z</td><td>redo</td></tr>
<tr><td class='key'>?</td><td>key map</td><td class='key'>F6</td><td>next panel</td></tr>
<tr><td class='key'>P</td><td>sketch on the image</td><td class='key'>O</td><td>ROI capture</td></tr>
<tr><td class='key'>Ctrl+H</td><td>file history</td><td class='key'>G / Shift+G</td><td>guideline example ✓ / ✗</td></tr>
<tr><td class='key'>F1</td><td>this guide</td><td class='key'>Ctrl+F1</td><td>all shortcuts</td></tr>
</table>

<h2><a name='trouble'></a>12 · Troubleshooting</h2>
<table>
<tr><td>{_i('logs')}</td><td><b>Something failed?</b> Open <i>Logs</i> (Ctrl+Shift+L) — every error of the session is listed there.</td></tr>
<tr><td>{_i('undo')}</td><td><b>View looks odd?</b> <i>Reset View</i> (R) shows all labels, clears filters and fits the image;
<i>Display ▸ Reset</i> removes image adjustments.</td></tr>
<tr><td>{_i('delete')}</td><td><b>Stale numbers?</b> File ▸ Clear Dataset Cache, then reopen.</td></tr>
<tr><td>{_i('tool-model-setup')}</td><td><b>Model problems?</b> Models &amp; Environment ▸ Check environment, and read the console.
Missing packages can be installed into a temporary environment.</td></tr>
</table>
<p class='muted'>Help ▸ Which tool for what? gives a one-page map of every tool.</p>
"""
