# Changelog

## 3.0.0 — A new look; one window for the three modes; overlap and loose-box QC; auto-sketched screenshots

Changes since 2.4.0.

### What's new

- Claude Review has its own icon — an AI agent with a reviewing magnifier, in one colour (ink / white) — so it stands apart from the other model tools.
- **Provenance:** everything the app writes says what made it — a manual edit, a model (weights file name and
  SHA-256, the score), an import, a conversion, a rule or an augmentation — with the feature, the version and the
  time; never a person, a folder path or a secret.
  - Each LabelMe shape a feature adds or changes gets a record with its history of changes; untouched shapes stay
    as they are. Hover a shape in the Annotations panel to read it. YOLO saves are logged per file; COCO annotations
    carry the record.
  - Each folder written or changed (Convert & Split, Preparation, augmentation, hand-off batches, Frame Extractor
    outputs, training runs) gets a hidden `.aio_qc_provenance.json`: settings, counts, input and output
    fingerprints and the input's record id — a chain from a model back to the raw data.
  - Frames, crops, masks, augmented / resized / re-encoded images and screenshots carry the record inside the file
    (XMP + EXIF / PNG Software, no re-encoding); linked or copied originals are never touched.
  - Trained models: best.pt / best.onnx carry what they were trained from; the run records the weights' SHA-256.
  - Settings ▸ General ▸ Record provenance turns it off.
- **Convert & Split is one window with a side pane** — no more two tabs. The steps (Output, Classes, Split, Files,
  Options; Alt+1 … Alt+5) sit in a pane on the left, each showing what is chosen in it, a dot on a step that needs a
  look, and under them a live summary of what will be written; one page at a time on the right, Ctrl+Enter converts.
  - Every format in one list, every option in one place: YOLO detection / segmentation, LabelMe, COCO, masks, VOC,
    CVAT, YOLO OBB, DOTA, YOLO pose, crops, flags → class folders; for classifiers class folders, LabelMe or CSV. The
    default folder name follows the format.
  - The **class list applies to every format** (ticks, order = ids, write as / merge, order from a model's file) — a
    classifier's classes too; a YOLO or COCO dataset starts in its own id order.
  - **YOLO from YOLO and COCO datasets** too (re-ordered, renamed, merged, filtered classes; a split of a YOLO
    dataset), not only from LabelMe.
  - One set, the dataset's splits or a new balanced split for every format, with the OCR-identity grouping and the
    per-feature table in the split preview; on a dataset of up to 30,000 images the split is planned as soon as the
    Split step opens.
  - Every format: images without a written class kept or left out, images with a class not written left out (as
    YOLO had), a report, a replace-in-place second run and **Undo** of exactly what the run wrote; Open in AIO QC, and
    Train a model… for class folders as for YOLO.
  - Large datasets: reading and copying adapt to the storage — several threads when files wait on it (a network
    share, large copies), one where Python is fastest; a run keeps a bounded number of files in flight (a million
    images no longer become a million pending jobs), and the window's checks are remembered instead of repeated.
- **Preparation** (formerly Data Prep) is a feature of the QC drawer: the first button of the **Dataset** tab, with
  its own icon, enabled with no dataset open (Ctrl+Alt+P). It is no longer in the Tools menu — one place per
  feature; the command palette finds it as QC ▸ Dataset ▸ Preparation (typing “data prep” still gets there).
- **One place per command in the menus:** Next unreviewed / Next Finding / Previous Finding live in the Go menu only
  (no longer repeated in Review); in the Frame Extractor, Batch Extract and Auto-Annotate live in Tools only (no
  longer repeated in File). Shortcuts are unchanged.
- **Fixed: a rare crash at a random moment** while background work runs (a scan, a conversion, a training monitor):
  Python's cycle collector could free a window part on a worker thread, which Qt does not allow. Collection now runs
  on the interface thread only, with the same thresholds.
- **Add new data** (Preparation, and `aio-qc add-data` on the command line): new image + annotation pairs into an
  existing dataset — LabelMe, or YOLO with or without train / val / test — matched by the **image content**, not the
  file name, so renamed or moved copies are recognised.
  - Images the dataset already has: left alone (listed when the annotation differs), the annotation replaced, or the
    classes it lacks merged in. Annotations count as the same within about a pixel, so a LabelMe copy of a YOLO
    label (coordinates rounded on the way) is not listed as different. Images repeated in the new data count once. Only images whose byte size occurs on the
    other side are read: a dataset of millions is listed, not read.
  - LabelMe or YOLO new data (any nesting, several folders) converted to the dataset's format, class ids by name
    (case and spaces ignored, plus a rename map); classes the dataset lacks added to its class list (classes.txt and
    data.yaml in step, comments kept), dropped, or the image left out; images without annotation left out or added
    as background.
  - New images to train, a balanced split (the dataset's own shares or yours; frames of one video together) or the
    new data's own splits; LabelMe datasets keep the new data's sub-folders or take one folder.
  - Names the dataset already uses for a different image renamed with the naming scheme — number suffix by default,
    folder / path prefix, renumbering, content hash, your pattern or your saved presets, every name verified.
  - Images decoded first (optional), copies or hard links, a CSV report of every decision, Undo.
- **Add classes by image** (Preparation, and `aio-qc add-classes`): annotations of more classes brought to the images a
  dataset already has, matched by image — e.g. a LabelMe master with every class into a YOLO dataset converted with a
  few. Any direction (LabelMe ↔ YOLO, boxes ↔ polygons for detection / segmentation); by default every class the
  dataset lacks, or the ones you list; a class both have in an image kept, replaced, or only the non-overlapping
  objects added. Every copy of a duplicated image is updated; dataset images the source does not have are listed, as
  they would lack the added classes. Undo restores every annotation and class list.
- **Pick images to label** no longer fails when two picks share a name in one class folder (or a name from an earlier
  pick is there): those get their folder as a prefix.
- Preparation remembers its page by name, and shorter pages no longer spread their fields out.

- **Train classifiers of any popular architecture** — on a class-folder dataset Train Model lists, besides the YOLO
  `-cls` models, 40 classifiers grouped by kind: small & fast (MobileNetV3 / V4, ShuffleNetV2, GhostNetV2,
  EfficientNet-B0, ConvNeXt V2 Atto, FastViT, RepViT, EfficientViT), CNNs (ResNet-18 … 101, ResNeXt-50, EfficientNet-B2
  / B3, EfficientNetV2-S / M, RegNetY, DenseNet-121, Inception-v3, ConvNeXt / ConvNeXt V2), transformers (ViT, DeiT
  III, Swin, TinyViT, EVA-02) and DINOv2 for few labelled images — or any timm / torchvision model by name. They train
  in your own environment with AIO QC's PyTorch trainer (timm when installed — *Install timm* offers it — else
  torchvision), with the same run view, Stop / Pause / Resume, Validate and Add to Models.
  - `best.pt` is an Ultralytics classification model with the image normalisation inside, so Testing, Model Check,
    the Class Gallery, Model Assist and the Frame Extractor use it like any `-cls` model; `best.onnx` (with
    Ultralytics' metadata) runs anywhere else. Each run also writes `val_predictions.csv` (every validation image, its
    answer and confidence, the mistakes first) and `confusion_matrix.csv`.
  - A flat folder of class folders (no train / val) trains as it is: 15 % of each class validates — the same images
    every run, and they stay put as images are added.
  - **Fine-tuning and improving**: the head alone for the first epochs (a linear probe), layer-wise learning-rate
    decay, weight averaging (EMA), label smoothing, stochastic depth, dropout; class balance by sampling or by loss;
    **hard examples** — the images your trained model gets wrong or is unsure of are drawn more often. Starting from
    an earlier run's model, its classes are matched to the dataset's by name (new classes start fresh, dropped ones
    go). Presets *Fine-tuning* and *Improve my model (hard examples)*. Extend Model works with these models too, with
    the before / after check per class.
- **Foundation models for few labelled images**: DINOv3 (ViT-S/16, S+/16, B/16, L/16, H+/16, the satellite & aerial
  ViT-L/16, and ConvNeXt Tiny / Small / Base / Large distilled from it), DINOv2 (S, B, L, with and without registers),
  SigLIP 2 (B/16, So400m), CLIP (B/16, L/14), Perception Encoder (S, B, L), AIMv2 (L, H), NVIDIA's C-RADIOv4
  (SO400M, H) and C-RADIOv3 (B, L, H), and Google's TIPSv2 (B, L, So400m) in Train Model's *Foundation features*
  group.
  - New schedule **Few labelled images (linear probe)**: the backbone runs once over every image (and a mirrored copy)
    and only the classifier layer learns on those features — seconds per epoch for any model size, even ViT-L on a
    CPU — with rare classes drawn as often as common ones. *With augmentation* runs the backbone every epoch instead.
  - A timm too old for a family (DINOv3 needs 1.0.20, Perception Encoder 1.0.16, SigLIP 2 1.0.15, AIMv2 1.0.13) is
    offered *Update timm* (TIPSv2 needs 1.0.28, C-RADIO 1.0.31); weights under their own licence (the DINOv3 License,
    Apple's ASCL for AIMv2, the NVIDIA Open Model License for C-RADIO) say so; gated
    weights and unknown models get a plain explanation (sign in with `hf auth login` or HF_TOKEN — never stored).
  - *GPU memory saver* (gradient checkpointing) for fine-tuning large models; image sizes follow 14-pixel patches too
    (multiples of 224).
- **Classifier augmentation** for YOLO classifiers and the rest: the auto-augment policy (RandAugment, AutoAugment,
  AugMix, TrivialAugment or none), random erasing, and the crop's zoom range in Train Model; the other architectures
  add rotation, MixUp and CutMix. Image Augmentation recipes now reach classifier training (whole images, in memory),
  and Generate files on class folders gives rare classes more copies and can take only chosen classes.
- Resume and Continue show a run as queued at once (it showed "paused" until the runner started).

- **Issues folder for class-folder datasets** — press **I** to set the image (or the selected images / gallery
  tiles) aside: it moves to `issues/<class>/` (`issues/<split>/<class>/` with train / val / test), out of the
  dataset, and the next image opens — one key per image when clearing thousands of crops. **Shift+I** adds a note;
  **Ctrl+Z** puts them back, Ctrl+Y sets them aside again. Works on the image, in the Class Gallery (tile after
  tile) and from the Class panel's *To issues* button. Settings ▸ Classification: inside the dataset (default —
  never read as a class), next to it (`<dataset>_issues`), or a folder of your choice (another drive is fine).
  Every move is logged in `issues_log.csv`. Review ▸ Issues Folder ▸ *Review Issues…* opens the folder as a
  class-folder dataset (fix classes there in the Class Gallery); *Put Issues Back…* returns each image to the class
  folder it is in then. Quick on big datasets: about 30 ms per image at 100,000 images, including opening the next
  one.
- Large datasets: removing or adding a few files (a delete, an undo, a file appearing) no longer recomputes the
  sort keys of every file — the first such step on a 100,000-image dataset took about 3.5 s.

- **Full classifier support, everywhere** — every tool that was written for boxes now has a classifier version, and
  the features outside the QC tools work with classifiers too:
  - **Testing (Model Lab)** scores classifiers: class-folder / LabelMe-flag test sets (or the dataset open in Image
    QC), exported classifiers (ONNX with a class-probability output, ultralytics `-cls`). Leaderboard with top-1 /
    top-5 accuracy and macro precision / recall / F1; per-class F1 and a confusion matrix by class name (click a cell
    for those images); accuracy vs coverage (the threshold becomes "unsure below": how many images it answers and how
    often those are right); the surest mistakes as pictures with the answer under each; A vs B (per-class recall
    difference, the images only one of them gets right); videos: confidence over time, brief class flips, drop-outs,
    agreement. Give an image another class right from the viewer (1 … 9, Enter — the image moves to that class folder
    / its flags are rewritten; the cached answers move with it and the run re-scores). HTML reports and
    metrics.json / per_class.csv / per_image.csv are classifier-shaped. Pre-labelling a test set with a classifier
    writes LabelMe flags.
  - In the video player a classifier's answer is a **pill in the corner** (the model's badge, "cat : 0.97", faint when
    unsure) instead of a rectangle around the whole frame.
  - The Shapes tab is **Images** for classification datasets and its checks open their classifier versions:
    **Duplicates** (copies filed under another class — Give copies the kept class), **Outliers** (images unusual for
    their class: size, shape, brightness, contrast, colour, file size), **Sequence Check** (a class that flips for a
    frame or two and back — Use the neighbours' class), **Dim / Area** (image size ranges), **Training Size** (what
    the classification transform does at imgsz: enlarged, centre crop cutting much, squashed; per class the imgsz
    that keeps 95 % sharp).
  - Labels tab: **Label Ambiguities** for class names (written two ways, plurals, outside the class list, rarely
    used — Merge into… / Rename…) and **Text Labels** for the names as text (spaces at the ends, invisible characters,
    characters a folder cannot hold, separators / case unlike the others — Rename to suggestion).
  - **OCR Check** reads the text in the whole image and compares it with the class (digits, letters, signs).
  - Class folders: **Validation** (images outside any class folder, files that are not images, empty / unreadable
    files, wrong extensions, empty class folders) and **Pairs** (annotation files left next to the images or without
    one, one file name in several class folders); **Fix in Editor** opens each image with one flag per class — ticking
    another class moves the image.
  - **Frame Extractor** pre-labels with a classifier as classes (LabelMe flags, or — YOLO layout — each frame in its
    class folder); Open in Image QC opens the result as a classification dataset.
  - **Data Prep ▸ Pick** (and Improve Model ▸ Pick new images) asks a classifier where it hesitates between two
    classes and can sort the picks into folders by its guess.
  - **Data Studio**'s annotation table gives each image-class row the image's size and shape.

- **Big datasets open without freezing** — opening a large LabelMe dataset read every JSON file (on a 12 GB dataset,
  about 12 GB) just to rule out COCO, with the window frozen — long enough for Windows to call it "Not responding". A
  LabelMe file is now ruled out from its first 4 KB, and a folder with many JSONs only checks COCO-looking names.
  Folder checks use one directory listing instead of a disk call per file. While indexing, the file list was told
  about every file of every batch: a 60,000-image YOLO dataset took over 5 minutes to index with multi-second
  freezes; now 15 s. The YOLO class-id scan stops when the dataset closes or the app exits (it could crash the app
  on exit).
- **Every feature checked on classification datasets** (class folders flat / with splits, LabelMe flags one class /
  several per image):
  - **Train Model** shows a classifier run's top-1 / top-5 accuracy (tiles, chart, run list, per-class precision /
    recall) instead of empty mAP fields.
  - **Improve Model** works for classifiers: the check is Model Check for classification (weak classes, the class
    each is confused with, misclassified images to flag); a class opens in the Class Gallery.
  - **Hand-off batches for class folders**: the annotator moves images between class folders; Import turns each move
    into a class change.
  - **Classification Check** finds the same picture filed under two classes.
  - **Smart Filter** works on classes (images with cat AND dog, not bird, two or more classes).
  - **Relationship Rules** (never together, requires … for image classes) are on the Labels tab for classifiers.
  - **Review Grid** tiles and **batch screenshots** show each image's class; screenshot captions name it.
  - Empty Annotations speaks of images without a class.
  - Data Prep ▸ Import reads JSON Lines classification tables (Hugging Face `metadata.jsonl`).
- **Class Gallery** (QC ▸ Labels; Ctrl+Shift+G in a classification dataset) — reviewing a classification dataset no longer means opening images
  one by one: each class is a wall of tiles where the image that does not belong stands out (Crop Gallery showed
  nothing for classifiers and now opens it). **Most suspicious first** puts the classifier's findings on top (sure of
  another class, another class, unsure — read from Model Check's cached results) and then the least typical images by
  look (Visual Similarity vectors); also least sure, least typical, grouped by look, random. Tiles show the review
  mark, "→ dog 93 %", "? 41 %" and "≈ cat" for a near-identical image in another class. **1 … 9**, **C**, **M** (the
  model's class) or dragging a tile onto a class moves it; **X** / **S** / **U** mark; **Space** marks every unmarked
  tile on screen OK and opens the next screen (at the end it goes back to any image scrolled past unreviewed); **P**
  peeks at full size (small images enlarged, crisp); Ctrl+Z undoes. **Compare with** puts two
  classes side by side; **Confusions** opens the classifier's mix-ups (also a double-click on Model Check's confusion
  matrix); **Look-alikes** finds images duplicated across classes; **Sample check** + **Sign off** estimate how many
  images of a class are wrong from a random sample and record it (✓ in the class list, with the range).
- Classification datasets: the filmstrip shows each image's class,
  and Visual Similarity's vectors follow images that change class folder.
- **Class panel for classification datasets** — the Annotations panel, empty for class folders and LabelMe flags
  before, now shows the image's **class**: every class with its colour, number key (1 … 9) and image count, the
  current one ticked. One click changes it (class folders: the file moves into that folder); with several images
  selected in the file list a dash marks classes only some of them have and a click sets it for all (multi-label: adds
  / removes it). Filter as you type (Enter takes the first match), **New class…**, **Classes…**; every change is one
  Ctrl+Z, and the panel follows undo / redo.
- The file list of a classification dataset shows each image's **class** ("cat", "cat +1" for multi-label, "no
  class") instead of "1 shape".
- Classes are marked with a coloured **dot** in the Class panel (the box icon is kept for box shapes only); the
  class chip on a classifier image sits at the top centre, below the hint card, instead of covering it.
- **Steadier, cleaner labels in video review** — labels are rounded tags with fixed-width scores, keep their spot from
  frame to frame (no more jumping between above / inside as boxes move), step aside with a thin leader line in crowds,
  shorten long class names with "…", and turn into a dot on tiny boxes; estimated (in-between) boxes are fainter.
  New options: **Clean while playing**, **Shade by confidence**, **Glide labels**.
- **Video review in the player** (Testing ▸ Review videos ▸ Review in player): boxes are coloured **by class** with
  Image QC's palette (each class its own, clearly different colour, kept across videos and sessions) instead of one
  colour per model; with several models each label starts with a dark badge carrying the model's number in its colour.
  New display options (remembered): colour by class / model, labels class + confidence ("car : 0.64") / class / off,
  light fill, label size. Outlines get a dark halo and label text is black or white by contrast; labels avoid each other. A **Classes**
  legend with colours and per-frame counts (untick to hide); hover a box for model, class, confidence and size;
  Alt+1 … 9 show / hide a model; **Save frame…** writes the frame at full resolution with its boxes. The paused-frame
  Detect overlay uses the same class colours.
- **Minimise, the same everywhere** — every feature window (checks, Search in Words, Claude Review, Data Prep, Train,
  Testing, the Crop Viewer, Settings …) now has a minimise button that tucks it into a **shelf** of chips at the
  bottom-left of the main window: its job keeps running, the chip shows the progress and then "done" with a short
  pulse; click the chip, or open the feature again, and it comes back where it was; × closes it (asking while a job
  runs). Before, most feature windows had no minimise at all (and closing one stopped its scan), while Testing and the
  Crop Viewer minimised to nowhere on Windows.
- Opening a feature again always brings its window back — never left minimised. Anything that minimises a feature
  window (Win+Down, a desktop's own button) ends in the shelf.
- When a job finishes while AIO QC is minimised or in the background, the taskbar button flashes (Windows), the Dock
  icon bounces (macOS) or the window is marked urgent (Linux).
- **Windows: taskbar minimise / restore is back** — the safety net that switched the taskbar integration off after any
  unclean exit (a kill, or an unrelated crash) now does so only for a crash inside the integration or three failed
  starts in a row, says so with a "Turn it back on" button, and never overrides your own choice; the automatic "off"
  left by the old guard is undone once.
- macOS: the main window's minimise works (a window without a title bar could not go to the Dock); Cmd+M minimises
  the window in front, so Label Mapping moved to Cmd+Shift+Option+M there.
- The Frame Extractor's "Review extracted frames" belongs to the app window (hides and returns with it).
- **Claude Review with your Claude plan** — *Use: Claude Code on this computer* runs the Claude Code you installed and
  signed in to yourself (Pro / Max / Team / Enterprise), so reviews count against your plan instead of an API key. AIO
  QC never sees the sign-in and keeps API-key variables out of it. Live only, at most 25 images per run (plans are for
  ordinary, individual use — whole datasets and batches stay on the API key); a signed-out Claude Code or a reached
  limit stops the run with its own message; *Connection… ▸ Test Claude Code*. Each call is trimmed to the Read tool, a
  short system prompt, no MCP servers and no saved session. Checked with a real Claude Code.
- Claude Review asks for the class of a missing object, so *Apply* can add it.
- **Search in Words** (QC ▸ Models, Ctrl+Alt+W) — search the dataset in words ("truck at night in rain", "blurred
  plate", "trailer roof with a hole") with a text-aware model (SigLIP2 / SigLIP / CLIP) running locally: images or
  labelled objects ranked best first, how many matched per split next to the split's size, and "only images without
  a '…' label" to find what is probably unlabelled. The first run embeds the images (shared with Visual Similarity);
  every search after that is instant. **Class descriptions**: one line per class, saved with the dataset; every label
  is scored against all of them — "looks like another class" (relabel with Undo), "unlike its class description", and
  per class how often the model agrees with the labels and which class it mistakes it for. Works before any model is
  trained.
- **Claude Review** (QC ▸ Models, Ctrl+Alt+V) — Claude looks at each image with its labels drawn and numbered, the
  label list, the class descriptions and the written guidelines, and says what is wrong in one sentence per finding:
  missing objects, wrong classes, loose or cut-off boxes, double labels, text that does not match, guideline breaches,
  labels on nothing. This image (a few seconds), selected / flagged / listed / all files live, or as a half-price
  batch that finishes in the background and survives closing the app; a cost estimate first. Results are kept per
  image — an unchanged image is never sent twice. *Apply* relabels, adds the suggested box or removes a label (backed
  up, Undo); *Dismiss* is remembered; accepted / dismissed counts per finding type show which to trust. The API key
  is read from an environment variable — only its name is stored; images are sent only for datasets you allow (asked
  once per dataset); a company gateway can be set as the server.
- **Visual Similarity** reads image sizes correctly with transformers 5 (it fell back to 224 px).
- **Classification datasets, done properly** — class folders under any of train / val / valid / test (a folder that
  is not a split next to them is reported, not guessed), LabelMe flags read as true / false whatever they say ("yes",
  1, "true" — reported so they can be fixed) and one class or several per image (asked once, remembered). *Change
  Class* (C) is a searchable picker (1 … 9, Enter, a new class; tick boxes for several classes) and 1 … 9 on the image
  set a class straight away; every change undoes with Ctrl+Z, for flags and for moved images alike. **Classification
  Check** (QC ▸ Labels, Ctrl+Alt+D): images without a class and why, several classes in a one-class dataset, flag
  values that are not true / false, class names that differ only in case / spaces / plural, classes with too few
  images, classes missing from a split, flags outside the class list — with *Fix values*, *Set class…* and *Merge
  into…*. **Classes** window: rename, merge, remove, keep and order the classes in one plan with one Undo. Stats speak
  of images and classes.
- **Classifier models** — Model Check scores a classifier on the dataset (top-1 / top-5, precision and recall per
  class, a confusion matrix you can click into) and lists the images it thinks are labelled wrong, with *Use the
  model's class*; Model Assist shows its three best classes as buttons; Train Model trains a -cls model straight from
  class folders; Extend Model adds images or classes to a classifier (new classes start silent, so the old answers
  stay the same until it has learned them). Tested with a real YOLO11n-cls.
- **COCO JSON opens in place** — annotations/instances_<split>.json or <split>/_annotations.coco.json; boxes,
  polygons (an object in pieces shares a group ID) and RLE masks as outlines; every QC tool works on it. Edits are
  written back losslessly (untouched objects keep every field, a relabel changes only the category, a new label becomes
  a category), many edits are one write, Undo is per image, and taking an image out (and putting it back) removes and
  restores its objects in the file.
- **Segmentation masks in and out** — import a folder of PNG masks (class index, a colour per class with a
  labelmap.txt, or one value per object with the class masks beside them; small regions left out) as polygons; export
  palette PNGs with the class index, 16-bit instance PNGs and labelmap.txt — and back again to the same objects.
- **More export formats** (Convert & Split) — Pascal VOC XML (with ImageSets and labels.txt), CVAT for images 1.1 XML
  (every split as a subset; image classes as tags, shape flags as attributes), YOLO oriented boxes (rectangles keep
  their corners, other areas get the smallest rotated box), DOTA, and YOLO pose (each box with the points of its
  group, one keypoint order for the dataset, kpt_shape and a left / right flip_idx in dataset.yaml). Also: crops of
  every shape as a classifier dataset, image flags as class folders, and classification datasets as class folders,
  LabelMe flags or a CSV table. OBB and pose output checked with a real Ultralytics loader.
- **More import formats** (Data Prep ▸ Import annotations, detected by themselves) — COCO keypoints (points grouped
  with their box), Label Studio brush masks, KITTI, VGG Image Annotator (choose the attribute that holds the class),
  Supervisely projects (bitmaps traced), CreateML, Open Images CSV (class ids named from class-descriptions) and
  Datumaro; whole-image classes from CVAT tags, Label Studio choices and file attributes; classification tables (one
  column, several classes in a cell, one-hot columns, "path label" lists, a split column) to class folders or LabelMe
  flags. Images are also matched by name without the extension.
- **Extend Model** (Train Model ▸ *Goal*: my model + more images / my model + new classes; Improve Model ▸ *Extend
  it*) — fine-tune a model you trained with more images or new classes without it forgetting what it knew. *Plan…*
  shows everything before anything runs: the model's classes keep their ids (locked), the new data's classes are
  matched by name (letter case and spaces ignored) and the rest added at the end — per class *add / same as … (merge)
  / rename / leave out*, with the object counts; which of the model's classes the new labels have (completeness); the
  old training data (found from the model's own run; without it a warning says the model may forget); the size of the
  training copy and the time per epoch. The training copy mixes the old and the new images with hard links (label
  files rewritten only where an id changes; the old validation images stay apart; a new image that is a copy of an
  old validation image is left out). The model pre-labels its own classes on the new images (they would otherwise be
  learned as background) — and with *review first* training waits: *Open the training copy* (a YOLO dataset of its
  own: your dataset is never touched) and *Continue training*. Its class-scoring head is enlarged, not reset: the old
  classes' part is copied, so it starts out detecting exactly what it did. Two stages: the new head first with the
  backbone frozen, then everything at a low learning rate. After training every class is scored before and after on
  the old and the new validation images: an old class that lost more than the limit (3 mAP points by default) is
  flagged with what helps. The plan file, the lineage of each model and a command line for a training server are kept
  with the run. Tested with a real YOLO11n learning a new class next to the 80 COCO ones.
- **Name Clashes** (QC ▸ Dataset, Ctrl+Alt+C) — file names that would collide when the dataset is flattened or copied
  to Windows: the same name in different sub-folders, names that differ only in upper / lower case (one file on
  Windows and macOS), two images sharing one label file (a.jpg and a.png both use a.json / a.txt), names Windows
  refuses (\ / : * ? " < > |, CON / PRN / AUX / NUL / COM1… / LPT1…, a trailing dot or space) and paths over 260
  characters. Each says whether the images are identical (a duplicate) or different, and what would break.
  *Rename…* renames in place — image and label together, imagePath fixed; review marks, notes, issues, cached results
  (labels, image facts, model predictions) and backups follow the new names; renamed.csv lists every change; one Undo.
- **One naming engine for every place names can clash** — presets (folder prefix, full path prefix, number suffix,
  renumber, content hash) or your own pattern with tokens ({stem} {ext} {folder} {folder2} {path} {n} / {n:04d}
  {hash8} {date} {split} {sep}); separator, letter case, spaces → _, a length limit; your own presets saved. Every
  name is checked as you type — against every other planned name and the files already there, letter case ignored,
  by name without extension, Windows' rules and path length — and nothing starts while a name still clashes
  (*Number the rest* adds _1, _2 … and checks again). Only the clashing files are renamed by default (or every file).
  A million names are planned and checked in seconds.
- **Convert to YOLO can flatten** (*Sub-folders: keep them* — the default — *or flatten*): every image straight in
  images/ (per split), names made unique with the naming engine, verified before it starts, listed in renamed.csv and
  the conversion report.
- **Combine YOLO datasets** (Data Prep) — several YOLO datasets (any nesting, with or without splits) into one: the
  class lists merged by name (the first dataset's order kept, new classes added at the end) and every label file
  rewritten with the new ids; names made unique with the naming engine; exact duplicate images once (the plan says
  whether their labels agree); hard links, sources untouched, renamed.csv, Undo.
- **Combine & Flatten, Clean Export and hand-off batches use the naming engine** — files that would share a name get
  a name from your scheme (default: the folder prefix) instead of only _1, _2; Clean Export can flatten; a hand-off
  batch can be one flat folder (for editors that do not open sub-folders) and still merges back to the right files.
- **Change Label works wherever the focus is** — L / F2 (Change Label…) and 1 … 9 (that class) act on the selected
  shape from the image, the Files list, the Annotations list or anywhere in the window (a text box keeps its typing);
  with nothing selected the status bar says what to do.
- **Paste puts the shapes exactly where they were copied from** — on top of the originals (in another image: the
  same position), ready to drag.

- **Improve Model** (Models tab, Ctrl+Alt+U) — the loop around training, round by round: check a trained model on
  the dataset (Model Check with just that model: the classes weakest first, with what to do about each in plain
  words), flag the images where model and labels disagree most, pick new images the model is unsure about (with empty
  label files, so they are labelled right here), fine-tune from it — and every check is a line in *Rounds*, so you see
  whether the model got better. Train Model offers it for a finished run (*Improve…*, and *Check it* on the toast).
- **Model Assist** (Edit ▸ Model Assist) — the dataset's models suggest labels on the shown image (Ctrl+Alt+J, or on
  every image): click one to accept it, right-click to drop it, *Accept All* (Ctrl+Alt+Enter); objects already
  labelled are not suggested again. *Outline Boxes with SAM* (Ctrl+Alt+X): a drawn or accepted box becomes the
  object's outline (SAM / SAM2 in the models' environment).
- **QC Presets** (Review tab, Ctrl+Alt+Q) — named sets of checks, each with its settings and a limit, run on a
  delivery with one click: PASS / FAIL with what to fix per file (HTML + CSV), flag the failing files, hand them off;
  export / import to share.
- **Errors say what happened, why, and what fixes it** — with a button for the fix (Open Models & Environment,
  Install it, Try again, Show the log) and the original message under Details; in scan windows, Model Assist,
  Improve Model, QC Presets, Data Prep, the QC Report, Image Augmentation and when a dataset cannot be opened.
- **Long jobs, one standard** (also Export / Import Hand-off Batch — Undo deletes a new batch — and the Frame
  Extractor's clip and annotated-video exports — time left, Stop, Undo deletes the new file, plain errors) —
  progress with the time left, Stop, a Background Tasks entry and Undo / Open at the end:
  every scan window now shows the time left; the QC Report can be stopped and shows its progress; Convert to YOLO and
  Image Augmentation appear in Background Tasks, and a conversion can be undone (only what it wrote is removed).
- **One design system, every mode.** Image QC, the Frame Extractor, Data Studio and every dialog share one set of
  colours, corner radii and spacing on Fusion (the same look on Windows, macOS and Linux): flat toolbars, underlined
  tabs, rounded inputs, slim scroll bars, quiet table headers, one teal accent; each window's main action is the accent
  button. Inter for the interface and *AIO Geist* for numbers, file names, timecodes and key hints — Geist with every
  digit the same width and a slashed zero, the look of a phone's recording timer (both bundled). Icons,
  never emoji.
- **Light and dark appearance** (*Settings ▸ General ▸ Appearance*: Follow the system · Light · Dark) — switches live,
  also when the system does; the canvas background follows it (warm light grey / near-black).
- **One window for the three modes, with its own title bar:** Image QC, the Frame Extractor and Data Studio are pages
  of one window — switching cross-fades in place while the mode switch's pill glides; the OS title bar is merged into
  the header (mark, mode switch, gear, minimise / maximise / close); drag the header to move, double-click to maximise,
  drag an edge to resize. *Settings ▸ General ▸ Use the system title bar* brings the OS frame back. On Windows the
  taskbar button minimises / restores it and the system animations play (*Settings ▸ General ▸ Windows taskbar and
  window animations*).
- **The approved layout:** toolbar *Open · Classes ● · location path* with navigation, Drag / Select, the zoom group,
  fit, 1:1, screenshot and Background Tasks on the right; QC tabs as text with their keys (*Dataset Alt 1* …) and
  *Search tools & actions… Ctrl K*; **Labels** with colour swatches (click to hide / show), count badges and "same
  as …?" beside look-alike names; **Annotations** with the shape outline in the label colour and #n; **Files** grouped
  by split with folding headers; rounded 16:9 **thumbnails**; read-outs over the image (pointer, selection, image size,
  file type and size).
- **Overlap and loose-box QC, made direct:**
  - **Duplicate & Overlapping Shapes** (QC drawer ▸ Shapes, Ctrl+D) — Duplicate Shapes and the overlap check merged
    into one: every pair worth a look, any classes, each listed once — *Exact copy* (identical points, also with two
    labels), *Marked twice*, *Overlap* (IoU above a threshold, 0.3 by default) and *Inside* (a shape inside a bigger
    one of its class). Copies are grouped (each paired with the group's first shape), so chains of copies never repeat
    a pair; parts of another class (a plate on a car) are not listed. Each pair is one picture with the rest dimmed — A
    teal, B coral (dashed when it is a copy), the shared area hatched, A / B tags that never cover each other; delete
    either side; *Expected for these classes* skips class pairs that overlap by nature (*Manage expected…* undoes it).
  - **Loose and cut-off boxes in Model Check** (*poor box fit*): your boxes compared side by side with your detection
    model's — a side sticking out past the object (*loose*) or cutting into it (*cut off*) by more than the new *fit ±*
    threshold (10 % by default) is a finding, also when the two boxes overlap well; the card names the sides, the
    model's box is drawn dashed, *Accept* takes it, *Show in file list* keeps only those images.
  - **Tighten a side from the keyboard:** click an edge or corner handle of the selected shape, then the arrows move
    only that side (Shift ×10); Esc goes back to moving the whole shape.
  - **F8 / Shift+F8 — next / previous finding** of the check used last (any check, the Issues Inbox, the Crop
    Gallery), opened with its shape selected and the keyboard on the image; a fix re-checks that file in every open
    check, so fixed findings leave the list.
- **Auto-sketch for Flagged Screenshots:** *Choose shapes…* walks through the batch — click the shapes on the picture
  (D / A between files) and the marks are drawn live: an arrow from the emptiest side (avoiding every other shape, the
  other arrows and the edges) with a ring, a box or nothing, in a chosen colour, optionally with text at the arrow; the
  screenshots show only the chosen shapes. Choices are kept per dataset and follow their shapes after edits.
- **Text for sketches:** templates with fields ({label}, {n}, {file}, {date}), a text size, eight **looks** (outlined,
  chip, shadow, plain; Alert / Note / Info chips) with colours chosen to stand out from the part of the image the text
  lands on, and a **live preview** on that very spot — in Sketch's Text tool and for auto-sketch's arrow texts.
- **Auto-Annotate** (Frame Extractor toolbar, *Image QC ▸ Tools ▸ Auto-Annotate Videos*, Ctrl+Shift+U): a folder of
  videos (or images) → unique frames → one LabelMe file per frame drawn by your model, as five steps you tick or leave
  out and can each run alone — *Frames* (every Nth or the sharpest of every N; repeats skipped within a video and,
  optionally, across videos and runs; blur filter), *Similar frames* (set aside to `similar/`, strict / balanced /
  loose, the sharper of two kept), *Annotate* (.onnx here or .pt in your environment; classes and the image size read
  from the model file — a .pt without loading PyTorch; confidence, IoU, max boxes, smallest box, per-class minimum
  confidence, renaming or leaving out classes; files exactly as **LabelMe 6.3.1 or 5.11.4** write them; **imageData
  off by default**; YOLO .txt optional), *Keep rule* (any / all of chosen classes, any box, always; other frames →
  `extra_images/` or kept) and *Package* (one folder, per video or by a name pattern; parts of N as `part_001` or
  `part_0, part_200`; zipped; hard links — no extra space — or copies). *Try on 4 frames* shows the boxes and the rule's verdict
  before a run. Stop is always safe and Run continues; running a step again first puts back what it set aside;
  annotation files the run did not write or that you edited are never overwritten; changed frame settings ask before
  replacing. Setups save every choice per project, and `aio-qc autoannotate` runs a setup unchanged on a server.
- **AI annotate** (Auto-Annotate ▸ Annotate ▸ *Who annotates*): annotation without a trained model. A vision model
  annotates from plain-word class descriptions and a few example images (per camera, with their boxes — drawn in the
  app). Every setting is in the app in seven tabs (Connect, Model, Classes, Examples, Checks, Output, Try) and saved
  as a project config.yaml (also editable as YAML, checked as you type with line numbers). *Is everything ready?*
  checks the app, the Python environment (PyTorch, transformers … with an Install button), the API key and the
  service first. Services: LiteLLM, OpenAI, Azure OpenAI, Gemini, Anthropic, AWS Bedrock, OpenRouter, DashScope,
  Together, Groq, Mistral, xAI, vLLM, Ollama, LM Studio, any OpenAI-compatible address, or a Hugging Face model in
  your environment; model profiles set each family's box format (Gemini, Qwen2.5/3-VL, InternVL, GLM, DeepSeek-VL,
  MiniCPM, GPT, Claude, Llama, Pixtral, Grok …), and *Detect…* finds it from an example. Modes: the model finds the
  boxes · a detector on your GPU (Grounding DINO, OWLv2, Florence-2, YOLO-World, your YOLO) proposes and the model
  checks · detector only · yes/no questions; optional SAM polygons; read text with a pattern and ISO 6346 / Luhn check
  digits; parts; "inside" rules; looking only inside a class's boxes; extra answers and accept rules; budget and
  per-class targets; cost estimate. The API key lives in an environment variable — asked once and saved as one when
  missing — never in the project, the logs or the reports. Answers are cached (Try again and re-runs are free),
  rejected images go to `rejected/` with reasons, every answer is audited. Output: **YOLO txt by default** (class
  order from the project, classes.txt, data.yaml; text / confidence / flags in a meta.jsonl), LabelMe 6.3.1 / 5.11.4,
  or both. Command line: `aio-qc autoannotate RUN --images DIR --ai-config project.yaml`.
- Auto-Annotate: the trained-model annotator writes LabelMe, YOLO txt or both; Package can lay YOLO parts out as
  `images/` + `labels/` with a data.yaml.
- **Convert & Split** (QC drawer ▸ Review, Ctrl+Shift+X) — Export & Split and the new Convert to YOLO are one
  window with two tabs: **Convert to YOLO** (LabelMe datasets; Ctrl+Alt+Y opens it on this tab) and **Split** (any
  dataset: train / val / test written as YOLO, LabelMe, COCO or class folders — now to `<dataset>_split` by default).
  A page's Close or Esc closes the window; closing it stops both pages' work and keeps their settings.
- **Convert to YOLO** (Convert & Split's first tab; LabelMe datasets): writes the dataset as YOLO **detection**
  or **segmentation** in another folder, with `dataset.yaml`, `classes.txt` and a conversion report; the dataset itself
  is not changed.
  - **Classes:** tick the ones to include and drag them into order (the order is the class id; Alt+↑ / ↓ too); each row
    shows its id, objects, images and shape types. Sort A–Z / by objects / ticked first, or *Load order from file…*
    (a model's `data.yaml` / `classes.txt`): classes the dataset lacks keep their ids. Right-click (or F2) ▸ *Write
    as…* / *Merge into*: a label written under another name, several labels as one class.
  - **Detection or segmentation**, suggested from the shapes. Rectangles, polygons, circles, oriented boxes and masks
    convert (masks traced to polygons); lines and points cannot and are counted, not written. Shapes are clipped to
    the image; pieces of one object (same group ID and label) are joined into one box or polygon.
  - **Save to** is checked as you type: a new or empty folder, an earlier conversion (*replaced*: only the files it
    wrote are removed), not empty (*Use "<name>_yolo" inside it*), or inside / around the dataset (refused).
  - **Layout:** one set (`images/` + `labels/`), the dataset's own splits, or a **new split**: val / test %, seed, the
    rare limit; exact duplicates, look-alike images (perceptual hash) and frames of one video kept together; the
    images looked at once (reduced decode, remembered) for lighting and look-alikes; *Preview split* shows the split
    and every class across train / val / test before anything is written; `reports/` gets `split_report.txt`,
    `feature_summary.csv`, `feature_map.csv` (every image with its features and split) and `split_manifest.json`.
    Images are hard-linked (copied when that is not possible); background images kept as empty label files or left
    out; flagged and broken images can be left out.
  - **More options:** merge labels that differ only in case or spaces; leave out objects smaller than N px² (in the
    image's real pixels); images with a class that is not written, or with a broken shape: keep them without those
    shapes, or leave them out; **parts inside a whole** — a part (a text line) is written only when it lies inside a
    whole (a plate) by a set share, else dropped or its image left out, and parts don't drive the split balance;
    re-encode images YOLO trainers can't read (GIF …) as PNG; read every label back after writing (*verified*).
    A JPEG without its end marker is always copied, never linked (trainers repair such files in place — through a
    hard link that would change the original).
  - A live summary of what will be written, progress with time left, *Stop*, *Open folder* and *Open in AIO QC*; the
    `dataset.yaml` is read back to check the class names; *Train a model…* opens Train Model on it. Settings are
    remembered per dataset.
- **Train Model** (QC drawer ▸ Models, Ctrl+Alt+B; no open dataset needed): trains an Ultralytics YOLO model
  (detection or segmentation) on a YOLO `dataset.yaml`, in your own Python environment.
  - **Checked before it starts:** splits and counts, the label format, val identical to train, classes without
    objects (in train / in val), class ids beyond the names, invalid lines, image formats Ultralytics skips, and a
    `path:` from another computer (the file's own folder is used — each run gets a copy of the file with an absolute
    path). The environment: Python, PyTorch, Ultralytics, the GPUs (with memory).
  - **Easy choices:** presets (*Quick look* · *Balanced* · *Best accuracy*); YOLO26 / YOLO11 / YOLOv8 in five sizes for
    the dataset's task, an earlier run's best.pt (fine-tune), your registered models, any .pt or model .yaml
    (weights are downloaded once into one shared folder); the image size suggested from the object sizes; batch
    *Auto* (fills the GPU); stop when val stops improving; device (a GPU, all GPUs, CPU); workers. *More settings*:
    image cache (with the RAM it needs), seed, a time limit, a share of the images, AMP, one class, and any
    Ultralytics setting as `key=value` (checked as you type — a name this environment's Ultralytics doesn't know is
    caught before the run starts).
  - **Augmentation and learning rate** (its own section): presets for each half — augmentation *Ultralytics default
    · Light · Strong · Off*, schedule *Ultralytics default · Fine-tuning · Long run (cosine)* — and every setting by
    hand: optimizer, start and final learning rate (the decay), linear or cosine decay, warm-up epochs, momentum,
    weight decay, frozen layers; mirror, upside-down, mosaic and when it switches off, MixUp, CutMix, copy-paste
    (segmentation), rotation, shift, zoom, shear, perspective, hue, saturation, brightness, channel swap. Only what
    differs from Ultralytics' defaults is passed on; the defaults are the environment's own, and a setting its
    Ultralytics doesn't have is switched off. **Text in the images** (class names like plate, digit, char, or single
    characters) switches mirroring off — mirrored text isn't real. *Fine-tune from it* applies the fine-tuning
    schedule (AdamW 0.0005, 1 warm-up epoch, cosine decay, the backbone frozen); a trained .pt chosen by hand offers
    it. A summary lists what changed; *Reset all* goes back; everything is remembered.
  - **Runs live on their own:** each run starts detached (also on Windows, out of the app's job object), keeps going
    when the window or AIO QC closes (closing asks: keep training or stop with the results), and is picked up again —
    a notification says when it finishes, stops or fails. A run whose process is gone shows as *interrupted*.
  - **Live:** epoch and batch progress with time left, GPU memory, the best scores (mAP50-95, mAP50, precision,
    recall, best epoch), loss and score charts (train solid, val dashed; hover for values; the best epoch marked),
    per-class results after the final check, the log, the run's settings. History of every run in the folder, also
    runs made outside AIO QC (from `results.csv`).
  - **Control:** *Stop* (finish now and keep the results: best.pt checked, plots made), *Pause* (after the next saved
    epoch; *Resume* continues there), *Stop now* (at once; the last saved epoch resumable), *Resume* an interrupted
    run. Then *Add to Models* (with its classes and image size: Model Check, Testing, Auto-Annotate use it),
    *Validate…* on val / test (per-class scores, plots), *Fine-tune from it*, *Open folder*. A failed run says what
    to do (GPU or RAM memory, a download, a data-loader worker, the environment, the dataset paths, a setting).
  - The model server's GPU memory is freed before a GPU run starts.
  - **Every ended run gives its resources back:** whether it finishes, stops, pauses, fails or is ended, the run
    empties the GPU cache, ends its data-loader worker processes and exits hard (a crashed run can otherwise hang on a
    worker or a thread and keep GPU memory); signals end it the same way. A watchdog checks every few seconds: a run
    whose process is still there 20 s after it ended has its whole process tree killed (only processes that really are
    training runs — a recycled process id is never touched). Stop now asks first, then kills the tree if needed.
  - **Running training is shown everywhere:** a chip next to the mode switch, in every mode ("Training · 12/100", or
    the number of runs) — click it for each run's progress, score, time, device and GPU memory, with **Show** (Train
    Model on that run) and **End** (frees its GPU and memory; resumable from the last saved epoch). A run that stops
    responding is marked and can be ended there.
- **Image Augmentation** (Models tab, Ctrl+Alt+A; works without an open dataset too) — build an augmentation
  recipe on your own images, then train with it in memory or write an augmented copy of the dataset:
  - **74 effects in 8 families.** 48 are built in (NumPy + OpenCV, nothing to install): *Geometry* — mirror,
    upside-down, quarter turns, rotate, shear, zoom, shift, perspective, random crop, elastic warp, piecewise warp,
    lens distortion; *Colour & exposure* — brightness, contrast, gamma, hue, saturation, grayscale, colour
    temperature, RGB shift, CLAHE, equalize, posterize, solarize, invert, channel shuffle, colour quantization;
    *Weather & lighting* — fog, rain, snow, glare, sun flare, shadow, vignette, low light; *Blur & focus* — Gaussian,
    motion, defocus, ghosting; *Noise & camera artifacts* — sensor noise, shot noise, salt & pepper, JPEG, pixelate,
    low resolution; *Occlusion* — cutout, pixel dropout, channel dropout. **albumentations** adds 18 (ISO noise,
    rain, snow, fog, sun flare, shadows, plasma shadow, uneven illumination, spatter, gravel, glass and zoom blur,
    chromatic aberration, Planckian jitter, tone curve, sharpen, ringing, emboss) and **imgaug** 8 (clouds, fog,
    snowflakes, rain, cartoon, superpixels, edge overlay, emboss) when they are installed — an **Install** button
    does it; each library effect is tried once first, so one that does not work with your NumPy / OpenCV is never
    offered (imgaug 0.4 is made to work on NumPy 2). The best of both libraries, and nothing depends on either.
  - **Labels follow every geometric effect** — rectangles, polygons, oriented boxes, circles, lines, points: all
    matrix effects are composed into one warp (one interpolation), the soft warps move label points exactly (to
    under a pixel); shapes are clipped to the image and dropped below a visible share / size you set. The uncovered
    area is grey (as YOLO pads), black, stretched or mirrored.
  - **A recipe:** each effect has a chance and a range; an image gets *a few at random* (1–3 by default), *every
    effect by its chance*, or *one effect per image* (the older pipeline's way); one **Strength** slider scales
    everything. Presets: Light · Balanced · Strong · Outdoor camera & weather · Night & low light · Poor camera ·
    Text & plates · Classic. Save / save as / delete, export / import as JSON — and **import the older imgaug
    pipeline's `aug_parameters.yaml`** (its four `apply_*` sections become a recipe).
  - **Live preview:** the original and five variations of the image shown in Image QC (◀ ▶ for others), with labels
    drawn, redrawn as you change anything; each says what was done to it ("Rotate 12.3° · JPEG 34") and how many
    objects stayed; **Shuffle** (R) draws again; the **eye** shows one effect alone from the low to the high end of
    its range; the time per image is shown. Find an effect by typing.
  - **Train with it — no files:** *Use in Train Model* (or Train Model ▸ Augmentation and learning rate ▸ *Extra
    effects*) applies the recipe to every training image **in memory**, differently each epoch, for all or a share
    of the images: no disk space at any dataset size. It is one step of Ultralytics' own pipeline (after mosaic /
    flips, before formatting), on the train split only, in every data-loader worker; boxes and masks follow its
    geometry (pose: pixel effects only); it survives *close mosaic* and Resume; a failing image goes through
    unchanged. Library effects the training environment lacks are named before you start and left out.
  - **Generate files** (for other trainers, or to review the result in AIO QC): copies per image, the train split
    only by default (val / test stay real), only images with chosen classes, **more copies for rare classes**
    (capped), a longer-side limit, a seed. **Paste objects from other images** first (the older pipeline's
    copy-paste): cut by their polygon (boxes optionally), resized, placed where they do not cover labelled objects,
    chosen classes only — shown in the preview. **Originals:** hard-linked into the new folder (no extra space on
    the same drive), only **referred to in `dataset.yaml`** (YOLO: nothing copied, on any drive), or left out.
    LabelMe, YOLO (boxes, polygons, oriented boxes) and class folders, written as the dataset is.
  - **For millions of images:** a pool of processes with a bounded queue (flat memory), every file written
    atomically, a per-image seed (the same settings give the same files on any number of workers), **resumable**
    (what exists is skipped; Stop any time), the size estimated and checked against free space first, progress with
    rate and time left, errors listed without stopping; the recipe and report are kept in `.aio_qc/augmentation.json`.
- **More options, everywhere the same:** extra options slide open above their button like a drawer (height eased per
  display refresh, the options fading in, a scroll area following along); the button then reads *Less options* /
  *Fewer settings* / *Hide …*. Used by Convert to YOLO, Train Model, the Frame Extractor's AI Annotate (Advanced
  settings) and libmpv (Places searched). Off with Settings ▸ General ▸ Animations.
- **Crop Viewer:** right-click an annotation ▸ *View crop…* (Shift+C): the shape's crop in its own window (as shown or
  original pixels, padding, outline, *Dim outside*, zoom / pan, ← / → through the annotations, *Follow selection*,
  copy / save).

### Added

- **Easy to start with (every mode):**
  - **Start pages** while nothing is open. Image QC's is the app's welcome: three cards (check annotations · frames
    from videos · clean a table) and the recent datasets, videos and tables of all three modes, each opened in its own
    mode. The Frame Extractor and Data Studio have their own (open, a folder of videos → frames, workspace …).
  - **Samples to learn with**, made on this computer: a street dataset with seven problems planted on purpose, a
    12-second street video, a small table with gaps and odd values (*Try a sample …*, Help ▸ Open a Sample …).
  - **A guided first run**: the first dataset / video / table shows a short tour that lights up one part of the window
    at a time (Next · Back · Skip, ← →, Esc); Help ▸ Quick Tour repeats it. The slides (Get Started) no longer
    open by themselves at the first start.
  - **Health check** (Image QC toolbar, Review menu, Ctrl+Alt+H): the common checks one after the other — files and
    images that cannot be read, missing partners, label names spelt two ways, files without shapes, duplicate images,
    objects marked twice — with each result in plain words, *Open* for the details (no second scan) and *Review the
    worst files first* (Issues Inbox).
  - **Simple menus** (View ▸ Simple Menus; on for a new install, off for existing users): the specialist commands
    leave the menus and the QC tabs; Ctrl+K still lists them and their shortcuts still work.
  - **Buttons explain themselves**: every tooltip has what the command does and its key; a disabled one also says why
    and what to do ("open a dataset first (Ctrl+O) …"). Menus show these tips too. The command palette shows a
    description under every command — and Data Studio now has one (Ctrl+K), and the key map (?) in every mode.
  - **Undo from the notification** after a change to many files at once (fixes from Model Check, rules, Dim / Area,
    Sequence, Similarity, Explorer) and after moving files to the trash.
  - **Help where you are**: a ? in Data Studio's panel headers opens that panel's part of the guide.
  - **One Help menu, the same in every mode**: Quick Tour · User Guide (F1) · Keyboard Shortcuts
    (? or Ctrl+F1 — Ctrl+/ in the Frame Extractor: every key and mouse gesture over the window, filter, click to
    run), then the sample. Gone as duplicates: the Get Started slides (the tour and the start pages do their job),
    the static Keyboard Shortcuts dialog and the Frame Extractor's shortcut page entry (the overlay has everything,
    the guide keeps the printable list), Key Map as a separate entry, and the Which tool for what? window (now section 6b of the
    guide, made from the tool list so it never drifts).
  - Errors say what happened in a notification with *Show details*, instead of a count in the status bar.
- **Data Studio in the new design**: the toolbar (outlined Open / Add step / Export, quiet icon buttons in the ink
  colour), panel headers like the other modes, the start page.
- **Notifications in the new design:** cards in the app's surface colour with its border and shadow (light and dark),
  a tinted badge for the kind (info · success · warning · error), a title over a muted second line, the app's buttons
  (the first action in teal) and a thin countdown line along the bottom in the kind's colour. **Details** open by height
  only, in one smooth movement upwards, at a fixed width (the message never re-wraps; the panel is laid out once and
  revealed), with **Copy** for the text.
- **Image QC ▸ (i):** the file's name, dimensions, type and file size are spaced apart with evenly placed dots.
- **New shapes like in an annotation tool:** **Create new bbox** (Ctrl+R — drag it; a crosshair across the image shows
  where the corner goes) and **Create new polygon** (Ctrl+N — click the corners). While a polygon is drawn: the first
  corner lights up and the pointer snaps to it when a click there would close it (as in LabelMe); Enter or a
  double-click closes it too; **Ctrl+Z** takes the last corner back and Ctrl+Shift+Z / Ctrl+Y puts it back (Backspace
  works too); **Shift** keeps an edge at 0 / 45 / 90°; **Esc** cancels; a faint edge previews how it closes, and the
  area fills as it grows. A hint over the image says what to do next and counts the corners; too few corners keeps
  the drawing going instead of throwing it away. Corners and lines stay the same size at any zoom. Also in
  the Edit menu and at the top of the image's right-click menu. The label is picked in a small **label picker** next to
  the new shape: search as you type, ↑ ↓ and Enter, 1–9 for the first nine labels, each with its colour and how often
  the dataset uses it, the last one used selected, and *+ Create "…"* for a new name (LabelMe). Relabel uses it too.
- **Delete asks first** (*Delete* / *Cancel*, with *Don't ask again*; *Settings ▸ Viewer ▸ Ask before deleting a shape*
  turns it back on). Undo still works either way.
- **Keyboard Shortcuts has a close button** (top right, every mode), next to its keyboard icon.

- **Image–Label Pairs** (QC drawer ▸ Dataset, Ctrl+Alt+K): every image matched with its label file, and everything
  that is not. LabelMe — JSONs whose image is missing (neither imagePath nor the same name next to it), pictures only
  embedded in the JSON, images no JSON uses, two JSONs on one image, imagePath naming another file, several images
  with the JSON's name. YOLO (images/ and labels/ mirrored) — labels without their image, images without a label
  (background or missing), labels whose name differs only in upper / lower case (pair on Windows, not on Linux),
  labels left in images/, several images on one label. Files that are not LabelMe / YOLO at all are listed too. The
  likely partner is searched across the dataset (another folder or split) and a safe fix proposed: point imagePath at
  the image, move the label next to its image, rename it to the image's case, create an empty label (background) or
  an empty LabelMe JSON — never overwriting, undoable; *Move to trash* is recoverable. 20,000 pairs in ~2 s.
- **Validation tells a wrong file from a right one:** a JSON that is not LabelMe says what it looks like (COCO, Label
  Studio, VIA, Supervisely, Create ML, Labelbox …), a .txt that is not YOLO too (class names instead of ids, pixel
  coordinates instead of 0…1, CSV, KITTI, DOTA, JSON / XML saved as .txt, a class list); every shape's structure is
  checked (label text, [x, y] points, the point count for its type, group_id, flags) and every YOLO line (class id
  range, value count, zero-size boxes, odd polygon coordinates, boxes and polygons mixed). Pairing moved from
  Validation and Empty Annotations to Image–Label Pairs (listed once).
- **Opening a mode** shows a large loading ring under the header while its code loads; switching between open modes
  cross-fades with one cheap blit per frame.
- *Settings ▸ General ▸ Unload the Frame Extractor when unused for … min* (15 by default): it closes with its state
  saved and reopens on the same video and frame.
- *Settings ▸ Viewer ▸ GPU image canvas* (OpenGL) for the smoothest zoom on big images and scaled screens.
- **Image details as an (i) icon** on the canvas that glides open on hover.
- **Feature windows** (scan windows, galleries, Settings, Testing, Data Prep, the Frame Extractor's and Data Studio's
  windows) get the same merged title bar and **rounded corners** (Windows 11 rounds them itself).
- A **Data Studio icon**; each mode's own icon on the window and taskbar (multi-size PNGs as well as SVG).
- *Save Flagged Files* as **CSV or TXT** (*Settings ▸ Review ▸ Flagged files list format*).
- Animated show / hide-all eye in the Labels header; tooltips in the app's colours.
- About links to the source code ("Source : GitHub").

### Changed

- **Image QC shortcuts** (Keyboard Shortcuts, the User Guide and the tooltips show the new keys):

  | Key | Now | Before |
  |---|---|---|
  | Ctrl+R | Create new bbox | B |
  | Ctrl+N | Create new polygon | — (new) |
  | Ctrl+E | Rename / Merge Label | Ctrl+R |
  | Ctrl+F | Fit Window | Ctrl+0 |
  | Ctrl+P | Search Files | Ctrl+F |
  | Ctrl+Alt+O | Fix in Editor | Ctrl+E |

  B and Ctrl+0 do nothing in Image QC any more. The Frame Extractor and Data Studio keep their keys.
- **The pointer read-out over the image** is drawn as key–value pairs (muted keys, bright numbers in the number font)
  with even gaps and dots between them.
- **Tooltips have round corners** (the corners outside the outline are see-through).
- **Pending review** has its own clock icon in the new icon style.
- **Smoother zoom:** the glide is clocked at the screen's refresh rate from the elapsed time, repaints in one pass, the
  sharp high-quality view is computed on a worker thread, and the garbage collector waits while the view glides.
  Background image checks and indexing wait while the view zooms or pans, so the first minutes with a large dataset
  no longer stutter; the YOLO class check of a dataset with `data.yaml` runs in the background.
- **Model Check, rearranged and steady:** three tabs — *Explorer*, *Findings* (a Cards | Table switch) and
  *Analysis*; the issue filter, the actions and the live thresholds sit inside the tabs they belong to, so switching
  tabs never moves anything; Accept / Reject / Undo stay as buttons and the rest (flag, file list, CSV, delete
  labelled shapes, rejected findings) is under *More*; *Parameters* is a toggle at the end of the run row and stays
  as you left it; reopening shows "Opening the last run…" and then "Last run restored"; the Explorer sidebar is as
  wide as its content and its check marks are drawn like the rest of the app.
- **Sketch / ROI tools take the place of the QC drawer's tab row** while the mode is on (they floated over the image).
- **Steady layout:** fixed areas for the toolbar path and the status-bar items; canvas read-outs no longer drift; the
  "Start here" card floats over the image (never resizes the canvas) and glides in / out.
- **Fits every screen:** every feature window is sized by the screen it opens on — workspaces (Model Check, Testing,
  galleries, Data Prep, Export, Models & Environment) take most of it, checks and lists a comfortable share, helpers
  less, with a cap on very large screens — instead of fixed sizes that were cramped on laptops and tiny on 4K. Rows of
  buttons and options wrap onto a second line on narrow screens, long hints wrap, and parameter forms scroll when the
  screen is short, so the results keep their room: every window's content fits a 1280 × 720 screen (checked at
  1280 × 720, 1366 × 768, 1920 × 1080, 2560 × 1440 and 3840 × 2160) without scrolling the whole window. The QC drawer
  lays its tools out by the width it has (never cut names).
- Settings: a quiet sidebar, section headings with icons and *Reset section*, settings in bordered cards.
- Menus: clean accent ticks; menu buttons show one drawn chevron.
- Labels and Annotations panels keep their size: long names scroll sideways, counts / #n stay pinned and names fade
  before them; panel headers have one height, float / close appear on hover.
- *Fix in Editor ▸ minimise* goes straight into the in-app mini window with a smooth glide.
- Resize arrows never stick (thin grips along frameless windows' edges).
- *Save Flagged Files*: the copy folder is always named after the dataset (`<dataset>_flagged`).
- Whole-image screenshots size outlines and tags by the picture (also on HiDPI screens and small images).
- QC drawer tabs look selected only while the drawer is open or pinned.
- Result cards in every gallery: finer outlines, and tags placed where they cover neither another tag nor another
  shape's outline.
- The welcome page has no status bar while it has nothing to say; it appears with a dataset, or while a background
  task runs.
- **Log files renamed:** `aio-qc.log` (was `aio_qc_tool.log`) and `aio-qc-crash.log` (a previous run's kept as
  `aio-qc-crash-<date>_<time>.log`).
- **The log file keeps no variable values:** crash tracebacks still have every frame, but no longer the values of
  local variables (`diagnose` off) — a key or token held in a variable can never end up in `aio-qc.log`.
- **For developers:**
  - The Image QC window is split by area: `app.py` keeps start-up and window events (220 lines, was 4,093), the rest
    is five modules in `AIO_QC/main_window/` (UI construction, dataset, display, tools, extras) and the tool
    catalogue `app_common.py` (`AIO_QC.app.QC_TOOLS` etc. still work).
  - Errors that core code handles by carrying on are no longer silent: `core.diag.ignored(exc)` writes one DEBUG
    line (where, type, message — secrets masked), capped per place so a million unreadable files give a handful of
    lines; plain file closes catch `OSError` only. The standalone scripts that run in your ML environment (training
    runner, inference servers, augmentation engine) are unchanged.
  - Lint with ruff (pyflakes rules, configured in `pyproject.toml`; the vendored editors are excluded): every unused
    import and dead assignment removed. Coverage configured (`pytest --cov`), a `dev` extra (pytest, pytest-qt,
    pytest-cov, pytest-timeout, ruff), and CI on every push: lint, then the tests on Windows (Python 3.10 and
    3.12), macOS and Linux, with coverage, crash logs kept when a run fails.
  - New direct tests for the least covered dialogs: Crop Gallery, the OCR validator editor and Data Studio's step
    form; Visual Similarity end to end with a toy model (a stand-in Python runs the real embedding server with a
    colour embedding: scan, cache, look-alike analysis, relabel / undo, Crop Gallery).
  - Testing (Model Lab) and the Frame Extractor window are split by area like Image QC:
    `widgets/testing/model_lab.py` (148 lines, was 4,000) + `model_lab_parts/` (layout, data, models, run, results,
    report, review) + `model_lab_common.py`; `tools/frame_extractor/app.py` (267 lines, was 3,273) +
    `main_window/` (ui, files, playback, engine, tools) + `app_common.py`. Their imports still work.
  - The same logging for errors handled by carrying on everywhere else (widgets, the Frame Extractor, utils): no
    `except Exception: pass` is left outside the standalone scripts and the start-up guards.
  - The image canvas is split by area like the windows: `widgets/canvas.py` keeps the class, its set-up and the Qt
    event overrides (384 lines, was 2,775), the rest is six modules in `widgets/canvas_parts/` (display, selection,
    overlay, zoom, input, editing) and the shared pieces in `widgets/canvas_common.py`. Its imports still work.
  - Coverage also measures the model servers that tests run as separate processes (`patch = ["subprocess"]`).
    New tests without the window for Data Prep (every importer — COCO incl. RLE masks, VOC, CVAT incl. tracks and
    masks, Label Studio — Combine & Flatten copy / move / undo, Unzip, orphans, integrity, exact duplicates, resize,
    EXIF rotation, Pick to label), the Frame Extractor's extraction (step / sharpest, blur + negatives, duplicates,
    scene cuts, ROI, chunked = one pass, stop + resume, markers, cross-video duplicates, model mining and
    temporal mining with a toy detector) and its command line, every Data Studio step option (pandas code runs,
    PySpark code is valid Python) and formula function, Clean Export in every format, Split Health and hand-off
    batches (every merge state). Coverage: 67 % → 70.6 %; CI fails below 68 %.
  - Model Check end to end on the real inference server: a stand-in Python runs `core/infer_server.py` with a toy
    Ultralytics (YOLO + SAM, `tests/infer_fake.py`), and the test drives every view of the window (findings, live
    thresholds, analysis and drill-downs, audit, accept / undo, reject, flag, report card); Model Check's coverage
    52 % → 79 %; all of AIO QC 70.6 % → 72.4 % (training runner 0 → 22 %, Fix in Editor 41 → 62 %, Models & Environment 42 → 74 %), CI floor 70 %. New modules: `core/improve.py`, `core/qc_presets.py`, `core/explain.py`, `widgets/model_assist.py`,
    `widgets/improve_dialog.py`, `widgets/qc_presets_dialog.py`, `widgets/errors.py` (`show_error`), `widgets/jobs.py`
    (`JobBar`, `eta`); scan windows emit `scanFinished` and can run `quiet` (no message boxes).
  - `tests/test_real_models.py` (opt-in, `AIOQC_REAL_PYTHON`): the model features with real models — a dataset
    made from Ultralytics' bundled photos with planted mistakes; Model Check finds them with YOLO11n, Model Assist
    adds the missing person, MobileSAM outlines a box, Convert to YOLO, a real training run, Improve Model on it
    (check, flag, pick from a pool, fine-tune). Runs in under a minute on a CPU.
  - New end-to-end tests for Fix in Editor (a stand-in labelme edits the working copies: fixed / rejected /
    unchanged, sync, undo, finish) and Models & Environment on the toy server (probe, add, load and map classes,
    test, export, install asked first). The training runner's copy is measured too (`AIOQC_RUNNER_DIR`,
    `[tool.coverage.paths]`).
  - Handled errors are counted per place and per error type: at exit the log gets one DEBUG line per place that
    caught something in that session (`core.diag.report()`), and `AIOQC_DIAG_DUMP=<file>` writes the same as JSON
    (also at the end of a test run) — the evidence for narrowing a handler to the errors it is really for.
  - **uv for development:** `uv.lock` pins every dependency (108 packages, one lock for Windows, macOS and Linux),
    `.python-version` picks Python 3.12, and `uv sync --all-extras` sets up the app, every optional part and the
    dev tools in one step. The dev tools moved from the `dev` extra to a `dev` dependency group (users never get
    them); `build` holds PyInstaller. CI installs from the lock (`uv sync --locked`) and checks the lock is up to
    date. Installing with pip works as before.

### Removed

- *Duplicate Shapes* as a separate check (now part of *Duplicate & Overlapping Shapes*, Ctrl+D).
- The always-visible zoom slider in the toolbar (now optional: *Settings ▸ Viewer*).
- The corner notification for "Start here" / "next check" (now a card over the image; corner notifications stay for
  results).
- The ? on Image QC's Files, Annotations and Labels panels — their names say what they are (Data Studio keeps its
  two).

- **Several shapes at once, and copy / paste, on the image (Image QC):**
  - **Select several:** Ctrl+click (⌘ on macOS) puts a shape in or takes it out of the selection, Shift+click adds
    it; a shape inside a selected one (a plate on a selected car) is reached too. **Ctrl+A** selects every shape —
    once one is selected, as in LabelMe. A plain click or Esc goes back to one / none. The shape clicked last is the
    primary one (its handles, the Annotations list).
  - **Together:** drag any of them or use the arrows (Shift ×10) and the whole selection moves, kept inside the
    image — one write, one Undo step; Del deletes them (asked once; one step); 1 … 9 / L relabels them all.
  - **Copy · cut · paste** (Edit menu; Ctrl+C / Ctrl+X / Ctrl+V and nothing else — ⌘C / X / V on
    macOS): shapes keep their label, type, points, group id, flags and description. Paste puts
    them at the same place in another image; over their own source they land 12 px down-right, further with every
    repeat; a shape from a larger image is brought inside (or skipped when nothing of it fits). Through the system
    clipboard they also go to another AIO QC window. The target's format decides: YOLO takes only its own classes,
    boxes (as 4-point polygons in a file of polygons) and polygons (not into a file of boxes), LabelMe every type —
    what could not be pasted is said, with the reason. The pasted shapes are selected, ready to move.
  - **Undo, one by one:** every pasted shape is its own exact-bytes Undo step (Ctrl+Z takes them back one at a
    time; Redo likewise); the file is refreshed on screen once per paste, not once per shape.
  - **Everywhere in the window:** Ctrl+C / X / V / A work wherever the focus is in Image QC — the image, the
    **Annotations panel** (which now selects several rows too: Ctrl+click, Shift+click for a range, Ctrl+A; the
    rows and the image always show the same selection; right-click offers copy / cut / paste and relabel / delete
    for all of them), the file list (Ctrl+V after moving to the next image; Ctrl+A there still selects files) —
    while text fields keep their own. Ctrl / Shift + click on a shape also selects in **Drag mode** (it switches to
    Select, as any selection does), and Ctrl+A works in both modes. Ctrl+click on a neighbour of the selected
    polygon selects it instead of adding a vertex to the selected one.

### Fixed

- **Train Model's charts** labelled their axis with odd values (0.00382, 0.137, 0.271 …) and could cut a label off
  on the left; the axes now step in round values (0.0, 0.2, 0.4 …) and the margin fits the widest label.
- **Local AI annotate models no longer run their own code without asking.** Florence-2, Moondream and similar
  models were loaded with `trust_remote_code=True`, which runs code from the model's download. Now it is off unless
  you allow that model once (*Allow this model to run its own code*, with a warning); the choice is kept per model
  on this computer and never read from a shared setup file. A model that needs it says so plainly instead of loading.
- **A model added with *Add to Models* (Train Model) could not be used by Model Check at once** ("No classes mapped")
  until it was loaded in Models & Environment; its classes are matched to the dataset's labels by name when it is
  added now.
- **Train Model could crash after it was closed** when an environment check finished afterwards.
- **Data Prep ▸ Unzip extracted the same archives again:** with "Move each extracted archive to _extracted_zips/"
  on, a second run found the moved archives and unpacked them a second time inside `_extracted_zips/`. Archives
  already put away are skipped now.
- **Frame Extractor batch: a stopped helper process could go unnoticed.** The progress loop treated every error of
  its message queue as "nothing yet"; only that is ignored now, and a broken queue ends the batch with an error
  (workers stopped) instead of waiting for the workers to finish.
- **Crop Gallery did not offer a label made in it:** after relabelling crops to a new label (or deleting the last
  ones of one), its label list still showed the old labels until it was reopened; it follows every change now.
- **Validators: a saved validator could not be deleted at once** — it was highlighted in the library but not loaded,
  so Delete / Copy stayed off until it was clicked again.
- **Train Model could keep the check of the wrong dataset:** with two dataset checks under way (the one remembered
  from last time and the one just chosen, or two picked quickly), the one that finished last won — so a slow check
  of the earlier dataset replaced the result for the one shown, and the run was prepared from it. Only the latest
  check counts now: the earlier one is told to stop and its result is ignored, and Start stays off until the shown
  dataset is checked. The same for the environment check when another Python is chosen meanwhile.
- **The training chip's *Show* did nothing** in Image QC and Data Studio (only the Frame Extractor's chip was
  connected): every chip now opens Train Model on the run itself, switching to Image QC first. Its list updates in
  place (a refresh never replaces the button under the mouse), stays on the screen, and *End* asks from the window
  (the list closes first) instead of from the list itself.
- **Train Model's left column could be narrower than its rows,** cutting off their labels: it keeps the width they
  need.
- **"QThread: Destroyed while thread … is still running" — the app aborted.** A background thread that belonged to a
  window or dialog was deleted with it while still working (Qt deletes a widget's children before anyone can step in),
  and Qt 6 answers that with an abort. Now every background thread of the app (Image QC, Frame Extractor, Data Studio,
  Testing — the dataset watcher, thumbnail / timeline / marker / detection / tracking workers, video engine, …) is
  owned by Python from the moment it starts until it ends: closing its window asks it to stop and lets it finish on its
  own; dropping the last reference or deleteLater() waits for its end; the log names it. At exit, a thread still stuck
  (e.g. a slow network read) no longer turns a normal exit into a crash.
- **"Windows fatal exception: page error" with a dataset on a network share or USB drive.** The dataset's databases
  (`.aio_qc/cache.sqlite`, history, backups, Testing and Frame Extractor databases) used SQLite's WAL journal, whose
  memory-mapped index network file systems do not support — reads of it failed (and risked damaging the database).
  On network and removable drives the classic rollback journal is used now (no memory mapping), local disks keep WAL;
  an existing database is switched over when it is opened.
- **Splits on diverse datasets:** when nearly every image had its own class combination, every image counted as
  "rare" and val / test stayed almost empty (both tabs of Convert & Split). Combinations that fine-grained are
  no longer protected (classes still are); if a split still falls far short, protection is relaxed step by step (a
  rare class may give one image, keeping two in train) and the report says so. A repair pass swaps in groups for
  features missing from val / test, and the planner is about 8× faster on large datasets (60,000 images: 9 s).
- **Split (Convert & Split), YOLO output from LabelMe:** `a.jpg` and `a.png` in one folder wrote the same label file; class
  names such as `no`, `yes`, `on` or `a: b` were read back as other values from `dataset.yaml` (names are quoted now,
  with `nc`); lines and points were written as boxes with no area; shapes outside the image were not clipped; pieces of
  one object (group ID) were separate objects; mask shapes were skipped; an image kept only in the JSON (imageData)
  was left out.
- Opening the dataset that is already open (Open Recent, Open, a start page) froze the window for about 30 s and then
  ran without its cache — on a big dataset a long re-index that could end in a crash. The open dataset is now closed
  first, and the cache no longer keeps a write lock after indexing.
- Reopening a big dataset waited for the folder watcher's full scan (it now stops at once), and the first image waited
  for the indexer's read of the whole cache (it now reads on its own connection).
- Every reopen left finished background workers and start-page rows behind; they are freed now.
- Data Studio's toolbar icons (save, Add step, undo, redo …) showed at a quarter of their size on high-DPI screens.
- Small empty windows flashed on the screen while the app started (ribbon buttons were shown before they had a place
  in the window).
- The Frame Extractor's start page showed through under the seek bar when no video was loaded; it now takes the
  player's place instead of lying over it.
- Image QC's QC tabs ran into the search field on a 1366 px screen (the last tab was cut, with scroll arrows); the
  search field now gives way first.
- The first-launch status-bar note about the QC Panel ("now a slim tab row …") is only shown to people who used the
  old one.
- Several notifications posted at once: all but the last could stay invisible (moving one that was still fading
  in stopped its fade).
- **Screen sizes:** the Frame Extractor needed a window 1545 px wide (its time / speed / zoom row and side panels),
  so on a 1366 or 1280 px laptop it ran off the screen; the step / speed / zoom group now moves under the time
  readout and the side panels' rows wrap (narrowest window 890 px). A window that grew for a mode's page slid off the
  right edge of a small screen (and the windows opened from it with it); it now stays on the screen. Menus and tool
  buttons that don't fit a narrow window sit behind a visible chevron (it was drawn blank, so Tools / Help seemed to
  vanish). The Auto-Annotate header no longer keeps room for empty info lines. Every window of every mode was opened
  on 1024×640, 1280×720, 1366×768 and 3840×2160 screens to check it fits.
- **Frame Extractor status bar:** a hint from hovering a button, or an error note, hid the background job shown at the
  left (timeline analysis …); the job now comes back as soon as the hint or note is gone, and the output line returns
  when the job is done.
- The toolbar path dropped the file name whenever files were added to or removed from the dataset folder outside AIO
  QC (another program saving, a sync client, frames being extracted) — it now only updates the file count.
- A crash at exit that came and went with the number of objects (PyQt's exit clean-up walking its table of widgets
  while deleting them): the window is now deleted while Python is still fully alive.
- Batch pre-annotations from a .pt model run in this Python used a fixed 640 px instead of the size it was trained at.
- **Windows:** a crash at start-up while the window restored its size and place.
- **Frame Extractor:** an access violation from libmpv (e.g. double-click zoom while playing); a missing import when
  the GPU canvas played with libmpv; the libmpv handle could be destroyed while its event thread still used it; Save
  Frame right after a video was closed raised an error.
- **Exit and unload:** a model / detector process still running logged an `AttributeError`; Data Studio and the Frame
  Extractor could abort the app when a background job outlived the exit wait (the job is now detached); closing an
  install window during `pip install` left the process running; the hard-frame export could poll a closed Frame
  Extractor.
- The window blinked when a mode was prepared or opened (its OpenGL surface now exists from the start); no flicker of
  stray windows (and of the console's focus) before the window appears.
- Model Check's *poor box fit* missed loose boxes that still overlapped the model's box well (IoU at or above the
  match threshold) — that is most loose boxes.
- The Issues Inbox, Coverage Map and Model Check's Explorer kept listening to the dataset after they closed.
- QC drawer tools could not be clicked after the drawer changed its layout.
- A rare crash after a delayed action (a timer's action cleared by Python's garbage collector).
- Counts in Flagged Screenshots and Save Flagged Files did not follow marks changed while the window was open.
- **Windows:** after opening a dataset (the folder dialog), a drag and drop, or with Get Started in front, the QC tabs
  did not open the drawer on hover and the image's (i) details did not unfold until something was clicked — Qt's
  Enter / Leave events and its "button held" state go stale there; hover now follows the cursor itself.
- Model Check: switching from the Explorer to another tab (and the last run arriving on reopen) re-laid the window
  out; the Explorer's controls stayed disabled, without saying why, while the last run loaded.
- Labels written with "&" (e.g. *Duplicate & Overlapping Shapes*, *Models & Environment*) lost it in menus and
  buttons.
- Label names in the Labels panel could be clipped at the end; a "QStatusBar::insertPermanentWidget" warning at start-up.
