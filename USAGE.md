# InterTreeSeg Usage Guide

**Language:** **English** · [中文](USAGE.zh.md)

This guide covers the three most common tasks: **running inference (on your own
test data)**, **training a new model**, and **evaluating results**. All commands
assume you run them inside the Docker `pointcept` environment.

---

## 0. Environment

Every command runs inside the configured Docker container (host `E:\` is mounted
at container `/workspace`). Use this wrapper:

```bash
# run any <command> inside the container
docker run --rm --gpus all -v E:/:/workspace ptv3:latest \
  bash -lc 'source activate pointcept && cd /workspace/InterTreeSeg && export PYTHONPATH=. && <command>'
```

Put every `python tools/xxx.py ...` below into the `<command>` slot.
(If you have an equivalent local environment via `pip install -e .`, you can run
the tools directly and skip the docker wrapper.)

**Directory conventions** (relative to `InterTreeSeg/`, created automatically on
first run):
- `log/<run_name>/` — logs and a copy of the config; `log/<run_name>/checkpoints/<run_name>.pth` — trained weights
- `result/<run_name>/<testname>/` — inference output (one `.ply` + `.h5` per scene)

---

## 1. Data formats

### 1.1 Scene files for inference (txt)
One point per row, whitespace- or comma-delimited. Column meanings are set by the
`data` block in the config. Defaults:

| column index | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| meaning | x | y | z | intensity | **treeID (instance GT)** | label2 |
| config key | `coord_channels` | | | (optional `feat_channels`) | `label_channel` | `aux_label_channel` |

- `label_channel` (default 4) = the tree-instance ID each point belongs to; `0`
  means ground / no instance. It is used as the ground truth for IoU.
- If your columns are ordered differently, change `data.coord_channels` /
  `data.label_channel` in the config — **no code changes needed**.

### 1.2 Training block files (HDF5)
Each `.h5` holds two datasets:
- `data`: `(B, N, C)` — B blocks, N points each, C channels (at least xyz, often
  with intensity, etc.)
- `label`: `(B, N)` — per-point label (foreground/background or semantic class)

Training reads from `data.data_dir/train/*.h5` and `data.data_dir/validate/*.h5`.
Raw scene txt → training block h5 is produced by `tools/prepare_data.py` (see §3.1).

---

## 2. Inference on your own test data

Inference needs no fixed data location — `--scene_dir` points at **any folder of
scene txt files**.

### 2.1 Simplest: scene folder + checkpoint

```bash
python tools/infer_scene.py \
  --cfg configs/default.yaml \
  --ckpt checkpoints/model.pth \
  --scene_dir /path/to/scenes \
  --log_dir my_infer_run \
  --n_clicks 5
```

This walks **every file** under `--scene_dir`, segments each scene, and writes
`.ply` + `.h5` to `result/my_infer_run/scenes/` (subfolder named after the scene
folder).

> Validated on a FORinstance test plot (~64 trees): 5 interactive click rounds,
> per-block mIoU rising from ~50 to ~72, full-scene merge IoU ~0.86.

### 2.2 Adding a batch of your own scenes
1. Put the scene txt files in one folder, e.g. `E:\mydata\myplots\`
   (`/workspace/mydata/myplots/` inside the container).
2. Make sure the column order matches the config (default `x y z intensity treeID label2`).
   If not, see §2.4.
3. Run:
```bash
python tools/infer_scene.py --cfg configs/default.yaml \
  --ckpt <weights.pth> --scene_dir /workspace/mydata/myplots --log_dir my_run --n_clicks 5
```

### 2.3 Common flags
- `--n_clicks N`: interactive click rounds per tree (more = more accurate, slower;
  the paper setting is often 5).
- `--force_divide`: force unclassified points onto their most likely instance
  (the legacy `--force_Divide`).
- `--grid_size 0.02`: voxelization grid size (default is already 0.02).
- `--ckpt`: explicit weights; if omitted, read from
  `log/<log_dir>/checkpoints/<log_dir>.pth` (i.e. your trained model).
- `--set key=value`: override any config value ad hoc, e.g.
  `--set inference.reclick_thr=0.95 data.bbox_delta=0.3`.

### 2.4 Different column order
No code changes — just override the config. E.g. if your txt is `x y z treeID`
(treeID in column 3):
```bash
python tools/infer_scene.py --cfg configs/default.yaml --ckpt <weights> \
  --scene_dir <folder> --log_dir my_run \
  --set data.label_channel=3
```
Or copy `configs/default.yaml`, edit `data.coord_channels` / `data.label_channel`,
and point `--cfg` at it.

---

## 3. Training a new model

### 3.1 Step 1: prepare training data (raw txt → training block h5)

Suppose your annotated raw scenes are in `E:\mydata\raw_train\` (comma-delimited,
columns `xyz, ?, ?, intensity, sem, inst` — same as FORinstance: inst in column 7,
sem in column 6):

```bash
# build train blocks
python tools/prepare_data.py \
  --src /workspace/mydata/raw_train \
  --dst data/trees/train/mytrain_ \
  --inst_col 7 --sem_col 6 --store_channels 6 --sample_size 32768

# build validate blocks the same way
python tools/prepare_data.py \
  --src /workspace/mydata/raw_val \
  --dst data/trees/validate/myval_ \
  --inst_col 7 --sem_col 6 --store_channels 6 --sample_size 32768
```

Afterwards the directory should look like:
```
data/trees/
  train/     mytrain_0.h5, mytrain_1.h5, ...
  validate/  myval_0.h5, ...
```
All `prepare_data.py` options (`--inst_col/--sem_col/--delta/--min_points/--store_channels/--delimiter`)
are configurable; run `python tools/prepare_data.py -h` for the full list.

### 3.2 Step 2: train

```bash
python tools/train.py \
  --cfg configs/default.yaml \
  --log_dir my_first_model \
  --loss combined
```
- Reads `train/` and `validate/` from `data.data_dir` (default `data/trees`).
- Evaluates on the validation set each epoch and saves to
  `log/my_first_model/checkpoints/my_first_model.pth` **whenever val mIoU improves**.
- Logs go to `log/my_first_model/logfile.log`; the fully-resolved config is written
  to `log/my_first_model/resolved_config.yaml` for reproducibility.

### 3.3 Commonly changed training settings
Edit `configs/default.yaml`, or override ad hoc with `--set`:
```bash
python tools/train.py --cfg configs/default.yaml --log_dir run2 \
  --loss combined \
  --set batch_size=32 epoch=200 optimizer.lr=0.0002 grid_size=0.02
```
- `loss`: `crossentropy | focal | tversky | combined`
- `optimizer`: `AdamW` (default) or `SGD`; `scheduler`: `Cosine` (default) or `OneCycleLR`
- `batch_size` / `epoch` / `num_point` / augmentation `data.augment.*`, etc.

> ⚠️ **Checkpoint-compat note**: the backbone params in `configs/default.yaml`
> (especially `mlp_ratio: 4` and `enc/dec_patch_size: 512`) match the released
> weights exactly. Changing them changes the network shape and the released
> weights will no longer `strict`-load — only touch them if you intend to train a
> brand-new architecture from scratch.

### 3.4 Infer with your trained model
Use the same `--log_dir` as training and `infer_scene` finds the weights
automatically:
```bash
python tools/infer_scene.py --cfg configs/default.yaml \
  --log_dir my_first_model --scene_dir <test_folder> --n_clicks 5
```

---

## 4. Multi-channel input (e.g. adding intensity)

By default only xyz is used (+2 click channels, `in_channels=2`). To also feed
intensity (source column 3) into the network:

```bash
# train a 3-channel model (in_channels = 1 intensity + 2 clicks = 3)
python tools/train.py --cfg configs/multichannel_intensity.yaml --log_dir intensity_model --loss combined
```
The only thing that matters is these two config lines (already set in
`configs/multichannel_intensity.yaml`):
```yaml
data:
  feat_channels: [3]          # source column of intensity -> in_channels becomes 3 automatically
  feat_norm: [standardize]    # standardize intensity; xyz is untouched
```
- `in_channels` is **derived from the config** — no code changes.
- Add more channels by appending column indices to `feat_channels` and giving each
  a normalization in `feat_norm` (`none`/`standardize`/`minmax`/`[scale, s]`).
- ⚠️ A 3-channel model **cannot** load 2-channel legacy weights (embedding stem
  size differs); train from scratch. To fine-tune, load with `strict=False` and
  re-initialize only `embedding.stem`.

---

## 5. Evaluating results

Inference already prints each scene's merge IoU in the log. For a standalone
**instance-level evaluation** (per-instance IoU, detection rate at IoU>0.7,
precision/recall/F1):

```bash
# evaluate a single result h5 (inference output: col 3 = GT instance, col 4 = predicted)
python tools/evaluate.py --pred result/my_run/scenes/scene1.h5 --detection_iou 0.7

# evaluate a whole directory with a summary
python tools/evaluate.py --pred result/my_run/scenes --detection_iou 0.7

# also export an error-colored PLY (green = correct, red = wrong)
python tools/evaluate.py --pred result/my_run/scenes/scene1.h5 --error_ply err.ply
```

---

## 6. Command cheat-sheet

| Task | Tool | Key arguments |
|---|---|---|
| Whole-scene interactive inference | `tools/infer_scene.py` | `--scene_dir` / `--testname`, `--ckpt`, `--n_clicks`, `--force_divide` |
| Training | `tools/train.py` | `--cfg`, `--log_dir`, `--loss`, `--set` |
| h5 validation-set evaluation | `tools/test_h5.py` | `--cfg`, `--log_dir`/`--ckpt` |
| Instance-level evaluation | `tools/evaluate.py` | `--pred` (file or dir), `--detection_iou` |
| Data prep txt→h5 | `tools/prepare_data.py` | `--src`, `--dst`, `--inst_col`, `--sem_col` |
| Batch over multiple test sets | `tools/run_batch.py` | `--log_dir`, `--testnames a b c` |

Add `-h` to any script for the full argument list. All configurable options live
in `configs/default.yaml` (fully commented); override any of them ad hoc with
`--set a.b=value`.
