# Fabric defect inspection

Web app that runs the fabric defect segmentation models of the fabrid benchmark on one photo. It
finds needle marks, holes, missing yarn and stains, draws a mask, a box, the class and the
confidence on every detection, and lists them in a table. Everything runs on the CPU. For the demo
it runs in a Kaggle notebook and is reached through a public Gradio link; it also runs on a laptop
as an offline copy.

Models, chosen in a dropdown:

| name | kind | weights file |
|---|---|---|
| YOLOv8m-seg (augmented) | Ultralytics | `yolov8m_seg_aug.pt` |
| FabriMSSA-Net | hybrid Mask R-CNN (torchvision) | `fabrimssa_net.pth` |

If a weights file is missing or does not load, that model is listed as unavailable with the
reason and the app still starts. At start each loaded model is also checked against its
reference pack (see Reference verification).

## The two weights files

They come from the output of the two Kaggle training notebooks and must carry these names:

| Kaggle notebook output | file name |
|---|---|
| `runs/baselines/yolov8m_aug_session_seed0/train/weights/best.pt` (job `yolov8m_aug`) | `yolov8m_seg_aug.pt` |
| `runs/ablation/hybrid_anchors_mssa_aux_session_seed0/checkpoints/best.pth` (job `hybrid_anchors_mssa_aux`) | `fabrimssa_net.pth` |

Use `best.pth`, not `last.pth`: `last.pth` holds the training state and the app refuses it. The
weights never go to GitHub: `.gitignore` excludes `weights/*.pt` and `weights/*.pth`.

The FabriMSSA-Net loader rebuilds the network with the training settings
(`configs/hybrid_anchors_mssa_aux.yaml`: anchors, MSSA and auxiliary heads on, preprocessing off,
input size 640) and loads the checkpoint strictly. A missing, unexpected or wrongly shaped key
stops loading with a message; there is no fallback to COCO or random weights. As in training,
the frozen batch norm layers use eps 0, which the COCO weights set during training and which is
not stored in the checkpoint.

## Run on Kaggle

One time setup:

1. **Weights dataset.** On Kaggle open Datasets, New Dataset, upload the two weights files with
   the file names above, keep it private and create it.
2. **GitHub token.** On GitHub create a fine grained personal access token with read access to
   the contents of `FaiazPronoy/fabrid-app` only.
3. **Notebook.** On Kaggle create a new notebook and import `kaggle/demo.ipynb` from this
   repository (File, Import Notebook).
4. **Input.** In the notebook's side panel choose Add Input and attach the weights dataset. The
   launcher searches all of `/kaggle/input` for the folder that holds both files, so the dataset
   name does not matter.
5. **Secret.** Open Add-ons, Secrets, add `GITHUB_TOKEN` with the token as its value, and tick it
   for this notebook. The notebook never prints the token and passes it to git only as an http
   header for the clone.
6. **Settings.** Accelerator: None (the app runs on the CPU). Internet: On (needed for the clone,
   the package install and the public link; Kaggle only allows it on phone verified accounts).

Each run:

1. Choose Save Version, Save & Run All. The run continues on Kaggle when the browser is closed.
2. Open the running version (the notebook's Versions list, or View Active Events) and watch its
   log. The launcher prints, in order: the Python version; which torch and torchvision are kept
   from Kaggle or installed from the pins; the installed version of every pinned package; the
   weights folder it found; the full `python -m defects.check` report; then the app's start log.
   If the check ends with `NOT READY`, a framed warning is printed and the app starts anyway.
3. Search the log for the line that starts with `PUBLIC URL:`. That `https://...gradio.live`
   address is the app; open it on the phone and on the projector laptop.
4. The link changes on every run, so share it only after the run has started. Kaggle stops a
   session after at most 12 hours; start the run one to two hours before the demo, not the day
   before. Stop it afterwards from View Active Events.

What the run stores: history goes to `/kaggle/working/data`. With Save & Run All, Kaggle keeps
the contents of `/kaggle/working` as the output of that version, so photos analysed during the
demo are saved with it. Delete that version's output afterwards if they should not be kept.

The notebook clones the default branch of `https://github.com/FaiazPronoy/fabrid-app.git`, so
push the final code there before the run. `kaggle/launch.py` keeps the torch and torchvision
that Kaggle already has when their versions match the pins (ignoring the build suffix such as
`+cu126`), otherwise it installs the pinned CPU builds, then installs the rest of
`requirements.txt`, and starts:

```
python -m defects.check --weights <folder> --data /kaggle/working/data
python app.py --share --weights <folder> --data /kaggle/working/data
```

## Run locally

Python 3.11 or newer (developed for 3.13). About 3 GB of free RAM is needed with both models
loaded. Put the two weights files in `weights/`, or anywhere and pass the folder.

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Open http://127.0.0.1:7860. Options, each also readable from the environment:

| option | environment variable | meaning |
|---|---|---|
| `--weights DIR` | `FABRID_APP_WEIGHTS` | folder holding the two weights files by their file names (default `./weights`) |
| `--data DIR` | `FABRID_APP_DATA` | folder for the history (default `./data`); use it when the code folder is read only |
| `--share` | | also create a public Gradio link |
| `--lan` | | listen on the local network so a phone can open `http://<computer IP>:7860` |
| `--port N` | | port (default 7860) |

After start the app prints `PUBLIC URL: <address>` (with `--share`) or `LOCAL URL: <address>` on
its own line. If `--share` was given but no link could be made, it prints a warning instead.

Tests need no weights, no dataset and no deep learning packages (numpy, Pillow, matplotlib and
pandas are enough; the interface tests run only when Gradio is installed):

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

```bash
ruff check . && ruff format --check .
```

`requirements.txt` pins Gradio 6.31.0 and the CPU builds of torch 2.11.0 and torchvision 0.26.0
with Ultralytics 8.4.209, the versions the models were trained with.

### Fallback: Gradio 5

The interface has not yet been run under Gradio 6.31.0. If the install fails or the interface
misbehaves, switch to the newest Gradio 5 release. Gradio 5.50.0 requires pandas below 3.0 and
Pillow below 12, so three pins change. Replace `requirements.txt` with this complete set (checked
against the PyPI metadata of each package), push, and run the notebook again:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
torchvision==0.26.0+cpu
ultralytics==8.4.209
opencv-python-headless==4.14.0.94
gradio==5.50.0
numpy==2.1.3
pillow==11.3.0
matplotlib==3.11.1
pandas==2.3.3
```

## Open it on a phone

Open the `PUBLIC URL` address. Tap the photo box: the upload button offers the phone camera or
the gallery, and the webcam button takes a picture inside the page. The camera inside the page
needs HTTPS, which the public link has. On the local network fallback (plain http) use the upload
button, which still offers "Take photo" on Android and iOS. Large phone photos are fine: the app
turns them upright from their EXIF orientation and scales them with antialiasing to the model
input size (long side 640 px) before inference. iPhone HEIC files are not readable; set the
camera format to Most Compatible or take the photo through the page.

## Demo day plan (16 November)

### Demo day checklist

**1. Start the Kaggle run (one to two hours before).** Save & Run All the demo notebook, open the
running version's log and wait for the line `PUBLIC URL: https://...gradio.live`. Open that
address on the phone and on the projector laptop and analyse one example image with each model
so both are warm.

**2. Read the check.** In the same log, the `python -m defects.check` report printed before the
app start must end with `READY`: both models loaded and `Verified against the benchmark
reference`, both weights files with their sha256, model card entries and default thresholds,
free disk space, and the inference time per model. If it says `NOT READY`, read the lines marked
`PROBLEM`; a model that failed is shown as unavailable and is not presented. On the laptop run the
same check for the offline copy:

```bash
python -m defects.check
```

then start it with `python app.py --lan` and leave it running.

**3. What to show, in this order.**

1. Analyse tab, YOLOv8m-seg selected. Click a picture in the row "Validation images from the
   dataset, from capture sessions not used in training", press Analyse. Point out the masks, the
   class and confidence on each, the table, the counts and the inference time.
2. Move the confidence slider and untick a class: the result changes at once, the model is not
   run again.
3. Choose FabriMSSA-Net (the slider jumps to its validated default) and analyse the same image.
4. Open Model input to show exactly what the model receives.
5. Take a live photo of a fabric sample with the phone and analyse it.
6. Create the PDF report, then show the History tab (reopen an analysis) and the Dashboard tab.
7. End on the About tab: limits, verification status and where each figure comes from.

**4. What to say about accuracy.** On validation fabric from capture sessions the models never
saw, both reach a mask AP50 of about 0.18 to 0.19, single seed, with wide 95% intervals that
overlap, so neither model is shown to be better. The models learned from one dataset of 2407
photos; on fabric, lighting or cameras unlike it they miss more defects and mark more wrongly.
The app is an aid for an inspector, not a final inspection. Say this before the live phone photo,
not after a miss.

**5. Offline fallback.** If the Kaggle run, the public link or the network fails, switch the
projector to the laptop at http://127.0.0.1:7860, put phone and laptop on one network (a phone
hotspot works) and open `http://<laptop IP>:7860` on the phone. Use the upload button's "Take
photo"; the in-page webcam button needs HTTPS and does not work here. Without any phone
connection, use the example images and the prepared photos on the laptop. The details follow
below.

### Full plan

Before the day:

1. Push the final code to GitHub, run the Kaggle notebook once, check that the log ends the check
   with `READY`, open the public link and run three photos per model. Note the inference times.
2. On a laptop with at least 8 GB RAM, install the requirements, copy the repository and both
   weights files, and check that `python app.py` works with the network switched off.
3. Put 10 to 15 test photos (good and defective fabric) on both the phone and the laptop.

One to two hours before:

1. Start the Kaggle run and wait for the `PUBLIC URL` line. Run one photo per model.
2. Start the laptop copy as well with `python app.py --lan` and keep it running.

If the Kaggle run, the public link or the venue network fails:

1. Connect the phone and laptop to the same network (a phone hotspot is enough; allow port 7860
   in the laptop firewall when asked).
2. Show the laptop browser at http://127.0.0.1:7860 on the projector and open
   `http://<laptop IP>:7860` on the phone.
3. On this plain http connection the webcam button inside the page does not work, because
   browsers only allow the camera on HTTPS pages. Tap the upload button and choose "Take photo"
   (Android) or "Take Photo or Video" (iOS) instead.
4. If no phone can connect, use the prepared photos on the laptop directly.

Never present a stored screenshot as a live result. Reopened history entries are marked in the
page as reopened, with no new inference run.

## Features

* Analyse tab: photo from the gallery or the camera, model choice with its validation figure,
  original and annotated image side by side, table of detections (class, confidence, box, mask
  area in pixels and as a share of the image), count per class and total, inference time.
* Examples row under the photo input, labelled "Validation images from the dataset, from capture
  sessions not used in training": the images found in `reference/*/images/`, one per file name.
  Clicking one only loads it as the photo; Analyse then runs the model on it like any upload.
  Nothing is precomputed. Without reference images the row is not shown.
* Confidence threshold slider and class filter: both re-filter the stored predictions without
  running the model again, and only the annotated image, summary, table and export files are
  sent back to the browser. Detections down to confidence 0.05 are kept for this. Choosing a
  model sets the slider to that model's default threshold (see Accuracy figures).
* Analysis size: the photo is scaled with antialiasing (Lanczos) so its long side is at most the
  model's input size from the registry (640 px for both models); smaller photos are not
  enlarged. Boxes, mask areas, the table, the CSV and the annotated image all use this size.
* Contrast stretch checkbox, off by default: rescales brightness between the 2nd and 98th
  percentile of the photo before inference, as an approximation of the "auto contrast via
  contrast stretching" step of the Roboflow export the models were trained on. Its settings in
  that export are unknown and this approximation has not been validated. Whether it was used is
  shown in the summary and stored in the history and the PDF report; its effect is visible in
  the Model input view.
* Model input view: the image exactly as the model receives it (Mask R-CNN resized and padded,
  with the channel normalisation undone for display; YOLO letterboxed by Ultralytics).
* Downloads: annotated image as PNG, table as CSV, and a PDF report (both images, the table,
  model, threshold, date).
* History tab: every analysis of the current browser session, with thumbnail, time, model,
  threshold and counts, can be reopened.
* Dashboard tab: defects per class, defects per image, and the confidence distribution over the
  session history.
* About tab: limits, verification status, the validation figure and default threshold of each
  model and the files they come from.

## History storage

History is written to `<data folder>/history/<session id>/`, by default `./data`, on Kaggle
`/kaggle/working/data`. A session is one opened browser page; reloading the page starts a new
one. Nothing survives the end of a Kaggle session except what Kaggle keeps as the version's
output (see Run on Kaggle). At every start the app deletes history sessions not changed for 24
hours; a failure of that cleanup is printed in the log and does not stop the app.

## Reference verification

The `reference/` folder holds, per model, a small pack made by the benchmark's own model code on
CPU in float32, from validation images whose long side is 640 px:

```
reference/<model_key>/detections.json
reference/<model_key>/images/<file name>.jpg
```

`detections.json` names the model key, the sha256 of the weights file, the minimum score (0.05)
and, per image, its file name, width, height and detections (label, score, box as x0 y0 x1 y1,
mask area in pixels). The images are committed as normal files.

At start, after loading, each model is checked against its pack. The sha256 of the weights file
must match first; then every image goes through the same code path as the interface. In both
directions every detection with score 0.30 or more needs a partner with the same label, box IoU
of at least 0.90, a score within 0.02 and a mask area within 10 percent. A model that passes is
shown as "Verified against the benchmark reference (n images)"; one that fails becomes
unavailable with the reason; one without a pack is usable and shown as "Not verified, no
reference pack". The same check runs from the command line and exits with a non zero code when a
check fails or a model cannot be loaded (`--weights` works here too):

```bash
python -m defects.verify
```

## Accuracy figures

The only accuracy figures in the interface come from `model_card.json`, read at start. Each value
was copied from a committed file in the benchmark repository, named in the entry:

| model | validation mask AP50 | source |
|---|---|---|
| YOLOv8m-seg (augmented) | 0.187, 95% interval 0.125 to 0.279 | `results/evaluation/yolov8m_aug_session_seed0/metrics.json`, `mask_ap50.mean` (benchmark evaluator) |
| FabriMSSA-Net | 0.182, 95% interval 0.127 to 0.254 | `results/evaluation/hybrid_anchors_mssa_aux_session_seed0/metrics.json`, `mask_ap50.mean` (benchmark evaluator) |

Both are measured on validation images from capture sessions not seen in training, single seed,
by the same benchmark evaluator with a 95% session bootstrap interval. The two intervals overlap
almost entirely, so these figures do not show that one model is better than the other.
A model without an entry shows "not evaluated yet". When the benchmark
repository sits next to this folder, a test checks the values against the source files.

Default confidence thresholds:

| model | default | source |
|---|---|---|
| YOLOv8m-seg (augmented) | 0.20 | mean of the per class `f1_threshold` values in `results/evaluation/yolov8m_aug_session_seed0/per_class.csv`, rounded to the nearest 0.05 |
| FabriMSSA-Net | 0.65 | mean of the per class `f1_threshold` values in `results/evaluation/hybrid_anchors_mssa_aux_session_seed0/per_class.csv`, rounded to the nearest 0.05 |

## Limits

The models were trained on one dataset of 2407 fabric photos, so accuracy drops on fabric,
lighting or cameras that differ from it. Defects can be missed and marks can be wrong. Treat the
results as an aid for a human inspector, not as a final inspection.

The training images were exported by Roboflow with auto contrast stretching. By default the app
does not reproduce that step, because the export does not record its exact settings, so phone
photos reach the model without it. The optional contrast stretch checkbox is an unvalidated
approximation.

## Adding a model

Add one `ModelSpec` to `MODELS` in `defects/registry.py`, put its weights file next to the others,
and add an entry with the same key to `model_card.json` when an evaluated result exists.
Supported kinds:

* `ultralytics`: any Ultralytics segmentation weights file whose class names are the four classes
  in order; `class_offset` 0.
* `maskrcnn`: a torchvision Mask R-CNN ResNet-50 FPN `state_dict` with five outputs (background
  plus four classes), as trained by `fabrid.maskrcnn`; `class_offset` 1.
* `fabrimssa`: the hybrid network, with its components in `settings`; `class_offset` 1.

The interface reads the registry, so it needs no change.

## Layout

| path | content |
|---|---|
| `app.py` | entry point: options, loads the models once, starts the interface |
| `kaggle/demo.ipynb` | Kaggle notebook: clones this repository and runs the launcher |
| `kaggle/launch.py` | installs the requirements, finds the weights, runs the check, starts the app |
| `defects/registry.py` | the model registry |
| `defects/models.py` | loading each registry entry, availability and reason |
| `defects/settings.py` | weights and data folders from options or the environment |
| `defects/yolo.py`, `defects/rcnn.py` | predictors for Ultralytics and torchvision models |
| `defects/fabrimssa.py` | inference copy of the hybrid network from the benchmark |
| `defects/checkpoint.py` | strict checkpoint loading |
| `defects/service.py` | analysis, filtering, history and exports, independent of the interface |
| `defects/verify.py` | checking each model against its reference pack |
| `defects/check.py` | readiness report for the demo, `python -m defects.check` |
| `defects/examples.py` | the example images listed under the photo input |
| `defects/views.py` | which values each interface event sends back |
| `defects/ui.py` | the Gradio interface |
| `reference/` | reference packs from the benchmark |
| `tests/` | tests with synthetic images and a fake predictor |
