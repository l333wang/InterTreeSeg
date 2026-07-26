# Checkpoints

Trained weights are **not stored in this repository** (they are large binary
files). Download them and place the `.pth` here.

## Download

| Checkpoint | in_channels | Notes | Link |
| --- | --- | --- | --- |
| `E100_B60_FOR05x4_novali_newmodel512.pth` | 2 (xyz + 2 clicks) | default model used by the web app and CLI | **[Google Drive](<ADD GOOGLE DRIVE LINK>)** |

<!-- TODO: replace <ADD GOOGLE DRIVE LINK> with the shared Google Drive URL. -->

## Setup

Place the downloaded file in this folder:

```
checkpoints/E100_B60_FOR05x4_novali_newmodel512.pth
```

The library, CLI, and web app all read a checkpoint path from configuration:

- **CLI** (`tools/infer_scene.py`, `tools/test_h5.py`): pass `--ckpt checkpoints/E100_B60_FOR05x4_novali_newmodel512.pth`.
- **Web app** (backend): set `ANNO_PTV3_CKPT` to the path (defaults to
  `checkpoints/E100_B60_FOR05x4_novali_newmodel512.pth`).

The default config (`configs/default.yaml`) reproduces this checkpoint's
architecture, so it loads with `strict=True`.
