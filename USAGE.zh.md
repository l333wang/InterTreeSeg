# InterTreeSeg 使用指南

**语言 / Language:** [English](USAGE.md) · **中文**

本指南覆盖三件最常见的事：**跑推理（加自己的测试数据）**、**训练新模型**、**评估结果**。
命令都以在 Docker `pointcept` 环境中运行为准。

---

## 0. 运行环境

所有命令都在配置好的 Docker 容器里跑（宿主机 `E:\` 挂到容器 `/workspace`）。约定一个 wrapper：

```bash
# 在容器里执行任意 <命令>
docker run --rm --gpus all -v E:/:/workspace ptv3:latest \
  bash -lc 'source activate pointcept && cd /workspace/InterTreeSeg && export PYTHONPATH=. && <命令>'
```

下文所有 `python tools/xxx.py ...` 都放到上面 `<命令>` 的位置。
（若你已装好等价的本地环境 `pip install -e .`，可直接运行，省去 docker 包裹。）

**目录约定**（相对 `InterTreeSeg/`，首次运行自动创建）：
- `log/<run_name>/` —— 日志、配置副本；`log/<run_name>/checkpoints/<run_name>.pth` —— 训练权重
- `result/<run_name>/<testname>/` —— 推理输出（每个场景一个 `.ply` + `.h5`）

---

## 1. 数据格式

### 1.1 推理用的场景文件（txt）
每行一个点，空白或逗号分隔。列的含义由配置里的 `data` 段决定，默认：

| 列索引 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 含义 | x | y | z | intensity | **treeID（实例GT）** | label2 |
| 配置项 | `coord_channels` | | | (可选 `feat_channels`) | `label_channel` | `aux_label_channel` |

- `label_channel`（默认 4）= 每个点所属的树实例 ID，`0` 表示地面/无实例。评估 IoU 时用它当 GT。
- 若你的数据列顺序不同，改 `configs` 里的 `data.coord_channels` / `data.label_channel` 即可，**不用改代码**。

### 1.2 训练用的块文件（HDF5）
每个 `.h5` 含两个 dataset：
- `data`：`(B, N, C)` —— B 个块、每块 N 个点、C 个通道（至少包含 xyz，通常还有 intensity 等）
- `label`：`(B, N)` —— 每个点的标签（前景/背景或语义类）

训练时从 `data.data_dir/train/*.h5` 和 `data.data_dir/validate/*.h5` 读取。
原始场景 txt → 训练块 h5 由 `tools/prepare_data.py` 生成（见 §3.1）。

---

## 2. 推理：用自己的测试数据

推理不需要把数据放到固定位置，`--scene_dir` 直接指向**任意存放场景 txt 的文件夹**即可。

### 2.1 最简单：指定场景目录 + 权重

```bash
python tools/infer_scene.py \
  --cfg configs/default.yaml \
  --ckpt checkpoints/model.pth \
  --scene_dir /path/to/scenes \
  --log_dir my_infer_run \
  --n_clicks 5
```

它会遍历 `--scene_dir` 下**所有文件**，逐个场景推理，把 `.ply` + `.h5` 写到
`result/my_infer_run/scenes/`（子目录名取自场景目录名）。

> 已在一个 FORinstance 测试样地（~64 棵树）上验证：5 轮交互点击，mIoU 从 ~50 提升到 ~72，整场景合并 IoU ~0.86。

### 2.2 添加你自己的一批测试数据
1. 把场景 txt 放进一个文件夹，例如 `E:\mydata\myplots\`（容器内即 `/workspace/mydata/myplots/`）。
2. 确认列顺序与配置一致（默认 `x y z intensity treeID label2`）。若不同，见 §2.4。
3. 运行：
```bash
python tools/infer_scene.py --cfg configs/default.yaml \
  --ckpt <权重.pth> --scene_dir /workspace/mydata/myplots --log_dir my_run --n_clicks 5
```

### 2.3 常用开关
- `--n_clicks N`：每棵树的交互点击轮数（越多越准、越慢；论文设置常用 5）。
- `--force_divide`：把未分类的点强制归到最可能的实例（对应旧代码 `--force_Divide`）。
- `--grid_size 0.02`：体素化网格尺寸（默认已是 0.02）。
- `--ckpt`：显式指定权重；不给则从 `log/<log_dir>/checkpoints/<log_dir>.pth` 读（即用你训练出来的）。
- `--set key=value`：临时覆盖任意配置，如 `--set inference.reclick_thr=0.95 data.bbox_delta=0.3`。

### 2.4 数据列顺序不同怎么办
不改代码，直接覆盖配置。例如你的 txt 是 `x y z treeID`（treeID 在第 3 列）：
```bash
python tools/infer_scene.py --cfg configs/default.yaml --ckpt <权重> \
  --scene_dir <目录> --log_dir my_run \
  --set data.label_channel=3
```
或者复制一份 `configs/default.yaml` 改 `data.coord_channels` / `data.label_channel` 后用 `--cfg` 指向它。

---

## 3. 训练新模型

### 3.1 第一步：准备训练数据（原始 txt → 训练块 h5）

假设原始标注场景在 `E:\mydata\raw_train\`（逗号分隔，列为 `xyz, ?, ?, intensity, sem, inst` —— 与 FORinstance 一致：inst 在第 7 列、sem 在第 6 列）：

```bash
# 生成 train 块
python tools/prepare_data.py \
  --src /workspace/mydata/raw_train \
  --dst data/trees/train/mytrain_ \
  --inst_col 7 --sem_col 6 --store_channels 6 --sample_size 32768

# 同样生成 validate 块
python tools/prepare_data.py \
  --src /workspace/mydata/raw_val \
  --dst data/trees/validate/myval_ \
  --inst_col 7 --sem_col 6 --store_channels 6 --sample_size 32768
```

生成后目录应形如：
```
data/trees/
  train/     mytrain_0.h5, mytrain_1.h5, ...
  validate/  myval_0.h5, ...
```
`prepare_data.py` 的参数（`--inst_col/--sem_col/--delta/--min_points/--store_channels/--delimiter` 等）都可调，`python tools/prepare_data.py -h` 看全部。

### 3.2 第二步：开始训练

```bash
python tools/train.py \
  --cfg configs/default.yaml \
  --log_dir my_first_model \
  --loss combined
```
- 从 `configs/default.yaml` 的 `data.data_dir`（默认 `data/trees`）读取 `train/` 和 `validate/`。
- 每个 epoch 在验证集上评估，**val mIoU 刷新最优时**保存到 `log/my_first_model/checkpoints/my_first_model.pth`。
- 日志写到 `log/my_first_model/logfile.log`；解析后的完整配置写到 `log/my_first_model/resolved_config.yaml`（方便复现）。

### 3.3 常改的训练配置
可以直接改 `configs/default.yaml`，或用 `--set` 临时覆盖：
```bash
python tools/train.py --cfg configs/default.yaml --log_dir run2 \
  --loss combined \
  --set batch_size=32 epoch=200 optimizer.lr=0.0002 grid_size=0.02
```
- `loss`：`crossentropy | focal | tversky | combined`
- `optimizer`：`AdamW`（默认）或 `SGD`；`scheduler`：`Cosine`（默认）或 `OneCycleLR`
- `batch_size` / `epoch` / `num_point` / 增强 `data.augment.*` 等

> ⚠️ **权重兼容提示**：`configs/default.yaml` 的 backbone 参数（尤其 `mlp_ratio: 4`、`enc/dec_patch_size: 512`）精确对应已发布权重。改动这些会改变网络结构，导致旧权重无法再 `strict` 加载——只有在你打算从头训练全新结构时才动它们。

### 3.4 用训练好的模型推理
`--log_dir` 用训练时的同名，`infer_scene` 会自动找到对应权重：
```bash
python tools/infer_scene.py --cfg configs/default.yaml \
  --log_dir my_first_model --scene_dir <测试目录> --n_clicks 5
```

---

## 4. 多通道输入（例如加 intensity）

默认只用 xyz（+2 个点击通道，`in_channels=2`）。要额外把 intensity（源数据第 3 列）喂进网络：

```bash
# 训练一个 3 通道模型（in_channels = 1 intensity + 2 clicks = 3）
python tools/train.py --cfg configs/multichannel_intensity.yaml --log_dir intensity_model --loss combined
```
关键就是配置里这两行（`configs/multichannel_intensity.yaml` 已写好）：
```yaml
data:
  feat_channels: [3]          # 源数据里 intensity 的列号 -> in_channels 自动变成 3
  feat_norm: [standardize]    # 对 intensity 做标准化；xyz 不受影响
```
- `in_channels` 由配置**自动推导**，无需手改任何代码。
- 想加更多通道就往 `feat_channels` 里加列号，并给 `feat_norm` 每个通道一个归一化方式（`none`/`standardize`/`minmax`/`[scale, s]`）。
- ⚠️ 3 通道模型**不能**加载 2 通道的旧权重（embedding stem 尺寸不同），需从头训练；若想微调，用 `strict=False` 只重置 `embedding.stem`。

---

## 5. 评估已有结果

推理会顺带在日志里打印每个场景的合并 IoU。要单独做**实例级评估**（per-instance IoU、IoU>0.7 检测率、precision/recall/F1）：

```bash
# 评估单个结果 h5（推理输出的 h5：列 3=GT实例, 列 4=预测实例）
python tools/evaluate.py --pred result/my_run/scenes/scene1.h5 --detection_iou 0.7

# 评估整个目录并给汇总
python tools/evaluate.py --pred result/my_run/scenes --detection_iou 0.7

# 顺便导出对错着色的 PLY（绿=对，红=错）
python tools/evaluate.py --pred result/my_run/scenes/scene1.h5 --error_ply err.ply
```

---

## 6. 命令速查

| 任务 | 工具 | 关键参数 |
|---|---|---|
| 整场景交互推理 | `tools/infer_scene.py` | `--scene_dir` / `--testname`, `--ckpt`, `--n_clicks`, `--force_divide` |
| 训练 | `tools/train.py` | `--cfg`, `--log_dir`, `--loss`, `--set` |
| h5 验证集评估 | `tools/test_h5.py` | `--cfg`, `--log_dir`/`--ckpt` |
| 实例级评估 | `tools/evaluate.py` | `--pred`（文件或目录）, `--detection_iou` |
| 数据预处理 txt→h5 | `tools/prepare_data.py` | `--src`, `--dst`, `--inst_col`, `--sem_col` |
| 批量跑多个测试集 | `tools/run_batch.py` | `--log_dir`, `--testnames a b c` |

任意脚本加 `-h` 看完整参数。所有可配置项集中在 `configs/default.yaml`（有逐项注释），临时改动用 `--set a.b=值`。
