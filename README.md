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
   weights folder it found; the full `python -m defects.check` report; the result of the
   interface check `python -m defects.uicheck`; then the app's start log. If the readiness check
   ends with `NOT READY`, a framed warning is printed and the app starts anyway. If the interface
   check fails, a framed warning is printed and the classic interface starts instead, so there is
   always a page.
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
python -m defects.uicheck
python app.py --share --weights <folder> --data /kaggle/working/data
```

`python -m defects.uicheck` builds the interface with a stand-in model, calls every event
handler once with realistic inputs (analyse, filters, mark style, model change, report, compare,
history open and clear, dashboard), checks the number of values each returns, and starts and stops
the server once with the real theme, stylesheet and script. It prints `UI CHECK PASSED` or names
each handler that failed. When it fails the launcher adds `--classic` to the app command.

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
| `--classic` | | start the previous, plain interface instead of the redesigned one |

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

1. Point at the header: each model's chip says Verified. If the page is dark on the projector
   browser, press Light / dark once; the browser then keeps the light page.
2. Inspect tab, YOLOv8m-seg selected. Click an example image under step 1, press Analyse. Read
   the verdict banner, the four class tiles, then drag the before and after handle. Open Defect
   close-ups and Defect details.
3. Open Display and advanced settings: move the confidence slider, untick a class, switch the
   mark style. The result changes at once; the model is not run again.
4. Choose FabriMSSA-Net (the slider jumps to its validated default) and analyse the same image.
   Open Model input to show exactly what the model receives.
5. Compare tab: run both models on one photo and read the agreement line, which says it is not an
   accuracy measurement.
6. Take a live photo of a fabric sample with the phone and analyse it in Inspect.
7. Downloads: create the PDF report. Then History (open an earlier card) and Dashboard.
8. End on the About tab: the defect types, the models with their figures and sources,
   verification and the limits.

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

The page has a header with the project title, an optional subtitle, institution, team and
supervisor lines from `branding.json` (empty fields are not shown), and one status chip per model:
Verified, Not verified or Unavailable. Tap or hover a chip for the reason. A short limits line sits
under every tab. The page opens in a light theme, whatever the visitor's system setting, because
projectors wash out dark pages. The "Light / dark" button next to the header switches to a dark
theme and the browser remembers the choice; both themes are checked for contrast. Charts and the PDF
report stay light. Every collapsible section shows a chevron that points right when closed and
down when open, and the whole header row can be tapped. The four class colours (needle mark wine, hole blue, missing yarn sand, stain green) are used for
the tiles, table chips, marks and charts. They were chosen so that every pair stays at least 20
CIELAB units apart under simulated protanopia, deuteranopia and tritanopia, and every label and
text colour reaches a contrast ratio of at least 4.5 to 1; `defects/palette.py` holds the check.

* Inspect tab, the default. Left: step 1 Photo (upload, camera, and the example strip "Validation
  images from the dataset, from capture sessions not used in training", built from
  `reference/*/images/`; clicking one only loads it, nothing is precomputed), step 2 Model (one
  line per model with its status and validation figure, and "More about this model" with the full
  card and limits), step 3 a large Analyse button that is disabled while a model runs, and a
  collapsed "Display and advanced settings" with the confidence threshold, classes to show, mark
  style and the contrast option. Right: a friendly empty state until the first analysis, then a
  green or amber verdict banner with model, inference time and analysis size, four class tiles,
  a before and after slider (full screen and download buttons), and the expanders Defect details,
  Defect close-ups, Model input and Downloads. A failed analysis replaces the verdict with an error
  card and hides the old result.
* Marks are drawn on the display image (long side up to 2048 px), not at 640 px: boxes and masks
  are scaled from analysis coordinates (masks with nearest neighbour), and label text is sized from
  the image on a solid background. Mark styles: Mask and box, Box only, Outline, Circle. Reported
  coordinates and areas stay in analysis pixels.
* Threshold, class filter and mark style re-draw the stored predictions without running the model
  again. An analysis is only re-filtered under its own model, and the slider shows the analysis'
  own threshold when its model is selected (also after reopening from history).
* Compare tab: one photo, every available model in turn with its own default threshold, side by
  side, with a neutral line on where the class counts agree and differ, and the note that this is
  not an accuracy measurement. Each run is saved to history.
* History tab: cards with thumbnail, time, model, verdict and counts; pick one and Open loads it
  into Inspect; Clear this session asks for confirmation first.
* Dashboard tab: tiles for analyses, defects shown and most frequent class above the charts of
  defects per class, defects per image and confidence.
* About tab: what the app does, the four defect types with their colours, the models with their
  figures and sources from `model_card.json`, how verification works, the limits, and the team.
* Analysis size: the photo is scaled with antialiasing (Lanczos) so its long side is at most the
  model's input size from the registry (640 px for both models); smaller photos are not enlarged.
* Contrast option, off by default: rescales brightness between the 2nd and 98th percentile of the
  photo before inference, as an approximation of the "auto contrast via contrast stretching" step
  of the Roboflow export the models were trained on. Its settings in that export are unknown and
  this approximation has not been validated. Whether it was used is shown in the verdict and stored
  in the history and the PDF report; its effect is visible in the Model input view.
* PDF report: branding, the verdict line, both images, the class tiles, the run details and the
  limits on page one, the defect table on page two.
* `python app.py --classic` starts the previous plain interface, kept in `defects/classic.py`.

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
| `defects/views.py` | which values each classic interface event sends back |
| `defects/screens.py` | which values each redesigned interface event sends back |
| `defects/summary.py` | verdict, class tiles, compare sentence, history captions, HTML pieces |
| `defects/marks.py` | marks drawn at display resolution in four styles |
| `defects/closeups.py` | close-up crops of each shown defect |
| `defects/palette.py` | theme and class colours with the contrast and colour blindness check |
| `defects/branding.py`, `branding.json` | title, subtitle, institution, team and supervisor |
| `defects/style.css` | the stylesheet of the redesigned interface |
| `defects/ui.py` | the redesigned Gradio interface |
| `defects/classic.py` | the previous interface, `python app.py --classic` |
| `defects/uicheck.py` | calls every interface handler once, `python -m defects.uicheck` |
| `reference/` | reference packs from the benchmark |
| `tests/` | tests with synthetic images and a fake predictor |
