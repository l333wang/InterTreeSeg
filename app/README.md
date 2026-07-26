# InterTreeSeg — Web App (Interactive Annotation)

> Part of the [InterTreeSeg](../README.md) project. This document covers the
> browser-based annotation app; see the repo root for the library, CLI, and
> training.

**Interactive point-cloud annotation for individual-tree (single-tree) instance segmentation in forestry.**

InterTreeSeg turns a promptable deep-learning model (Point Transformer V3) into an interactive labeling loop: frame a tree from the top view, click positive/negative prompts, get a real-time segmentation, refine it with more clicks or manual lasso/brush edits, then confirm it as a tree instance with attributes (species, height, crown width, DBH, …). Annotation is always done at **full resolution** — the render is decimated for speed, but labels, inference, and export cover every original point.

The UI is English by default with a **中文 (Chinese)** toggle in the top-right.

---

## Features

- **Top-view box → click → segment** — draw a rectangle in an orthographic top view (CloudCompare-style vertical cut), then place positive/negative clicks; the model segments the tree in real time (~120–250 ms warm).
- **Two refinement paths** — AI clicks (re-run the model) and manual lasso/brush editing (edit point labels directly), freely mixed.
- **Full-resolution annotation** — inference, per-point labels, manual edits, and export all operate on the full-resolution cloud; decimation only affects render density.
- **Instance management + attributes** — per-tree ID, species, growth status, DBH (manual) and auto-computed height / crown width / stem XY (editable, batch-recomputable).
- **Export** — full-resolution per-point instance labels (`.txt`) + a per-tree attribute table (`.csv`).
- **Large clouds** — millions of points render smoothly via decimation; the full cloud stays server-side with a KD-tree index.
- **Bilingual UI** — English / Chinese, switchable at runtime.

---

## Architecture

```
app/
├── backend/            FastAPI (REST + WebSocket), Python
│   ├── main.py         API endpoints
│   ├── session.py      full-resolution scene state, labels, instances
│   ├── inference/      InferenceService interface + Mock + PTv3 model
│   ├── io/             readers (TxtReader) + exporters (labels / CSV)
│   ├── spatial/        KD-tree crop/pick + screen-space projection selection
│   └── geometry.py     height / crown / stem auto-computation
└── frontend/           Vite + React + TypeScript + three.js
    └── src/
        ├── viewer/     three.js rendering, interaction, annotator controller
        ├── panels/     top bar, toolbar, instance panel, attribute dialog
        ├── state/      zustand store
        ├── api/        REST client
        └── i18n.ts     English / Chinese strings
```

The `InferenceService` interface decouples the UI from the model: a `MockInferenceService` (pure geometry, no GPU) drives the whole loop for development, and `Ptv3InferenceService` runs the real model. Selection is via the `ANNO_INFERENCE_BACKEND` environment variable — no code changes.

---

## Prerequisites

- **Windows** (tested) with an NVIDIA GPU + recent driver (CUDA 11.8-capable) for the real model. The mock backend needs no GPU.
- **[Miniconda/Anaconda](https://docs.conda.io/)** for the Python environments.
- **Node.js ≥ 20** for the frontend (a conda `nodejs` env works well on Windows).
- A trained Point Transformer V3 checkpoint (`.pth`) — download per [`../checkpoints/README.md`](../checkpoints/README.md). The model code is the packaged `intertreeseg` library (this repo).

---

## Installation

### 1. Backend — real model (GPU)

Create an environment with PyTorch (CUDA 11.8), the PTv3 dependencies, and the web stack:

```bash
conda create -n intertreeseg python=3.9 -y
conda activate intertreeseg

# PyTorch (CUDA 11.8) — match your CUDA/driver
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# PTv3 model deps
pip install addict timm spconv-cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu118.html

# numerics + web
pip install numpy pandas scipy fastapi "uvicorn[standard]" python-multipart websockets
```

> **flash-attention is NOT required.** The model is built with `enable_flash=False`, which uses the plain-softmax attention path (mathematically equivalent, so pretrained weights apply unchanged). This avoids the hard-to-build `flash_attn` dependency on Windows.

### 2. Backend — mock only (no GPU, optional)

If you just want to try the UI without the model:

```bash
conda create -n intertreeseg-mock python=3.9 -y
conda activate intertreeseg-mock
pip install numpy pandas scipy fastapi "uvicorn[standard]" python-multipart websockets
# then run with ANNO_INFERENCE_BACKEND=mock
```

### 3. Frontend

```bash
# with Node.js ≥ 20 on PATH (e.g. `conda create -n webnode -c conda-forge nodejs`)
cd app/frontend
npm install
```

---

## Configuration

Set these environment variables before starting the backend (all optional, sensible defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `ANNO_INFERENCE_BACKEND` | `mock` | `ptv3` for the real model, `mock` for the geometry placeholder. |
| `ANNO_PTV3_CKPT` | `../checkpoints/E100_B60_FOR05x4_novali_newmodel512.pth` | Path to the `.pth` checkpoint (download per `checkpoints/README.md`). |
| `ANNO_PTV3_GRID` | `0.02` | Serialization grid size (must match training). |
| `ANNO_PTV3_SIGMA` | `0.01` | Click Gaussian width in world metres. |
| `ANNO_MAX_RENDER_POINTS` | `1500000` | Render target: larger clouds are evenly sub-sampled to exactly this many points (labels/inference/export stay full-res). Lower it for smoother rendering. |

---

## Running

### One command (Windows PowerShell)

```powershell
cd app
./dev.ps1          # starts backend (real model) on :8000 and frontend on :5173
```

Then open **http://127.0.0.1:5173**.

### Manual

```bash
# Backend (real model) — port 8000
cd app
ANNO_INFERENCE_BACKEND=ptv3 conda run -n intertreeseg python -m uvicorn backend.main:app --port 8000

# Frontend — port 5173 (proxies /api → 8000)
cd app/frontend
npm run dev
```

The model loads on startup (~7 s), so the first click is not slow.

---

## Usage

1. **Open a file** — top bar → *Open File*, choose a `.txt` point cloud (6-column `x y z ...`). It uploads and renders, colored by height.
2. **Frame a tree** — pick the **Top-view box** tool (`B`). The view switches to orthographic top-down; drag a rectangle over one tree. It becomes a vertical block (blue box).
3. **Click prompts** — the tool auto-switches to **Positive** (`P`). Click on the trunk/crown (positive = this tree); switch to **Negative** (`N`) and click neighbours/ground to exclude them. With **Realtime** on, each click re-runs the model.
4. **Refine (optional)** — use **Lasso** (`L`) or **Brush** (`K`) to edit directly: left-drag adds points, `Alt`+drag removes. Manual edits act on full-resolution points.
5. **Confirm** — press `Enter` or *Confirm as a tree*. Fill in species / DBH / growth status (height, crown, stem are auto-filled). The tree gets an instance color and ID.
6. **Manage instances** — the right panel lists trees; select one to edit attributes, toggle visibility, delete, or lasso/brush-edit its points.
7. **Compute geometry** — *Σ Recompute geometry* recomputes height/crown/stem for all trees (manual overrides are preserved).
8. **Export** — *⬇ Point labels* (full-resolution points + instance-id column) and *⬇ Attributes CSV* (one row per tree).

### Keyboard shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| `B` | Top-view box | `L` | Lasso |
| `P` | Positive click | `K` | Brush |
| `N` | Negative click | `T` | Top view |
| `Enter` | Confirm tree | `Esc` | Discard current |
| `Space` | Orbit/Pan | `[` `]` | Brush radius |

### Language

Toggle **EN / 中文** in the top-right corner at any time.

---

## Export format

- **Point labels** (`labels.txt`): the original columns plus a trailing **instance-id** column (`0` = unlabeled). One line per **original** point — full resolution.
- **Attributes** (`trees.csv`): `instance_id, species, height, crown_width, stem_x, stem_y, dbh, health_status, point_count, notes` — `instance_id` matches the label column.

---

## Notes & limitations

- Input is currently `.txt` (6-column). Other formats (`.las/.laz/.ply`) can be added by implementing a new `PointCloudReader` — the rest of the pipeline is format-agnostic.
- Rendering uses uniform decimation (not yet an octree/LOD); very large clouds (tens of millions) will benefit from a Potree-style tiler in a later iteration. Annotation quality is unaffected — it is always full-resolution.
- The click Gaussian width (`ANNO_PTV3_SIGMA`) and grid size (`ANNO_PTV3_GRID`) replicate the training recipe; change only if you retrain.
- Single-user by default. The session layer is multi-session-ready; server deployment (auth, GPU queue, persistence) can be added around it without changing the core.
