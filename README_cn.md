# ForensicVision

多方法文档伪造检测系统（Multi-Method Document Forgery Detection System）

---

## 1. 项目简介

ForensicVision 是一个面向图像取证场景的文档伪造检测系统，主要针对证件、票据、合同等文档图像，自动判断其是否被篡改，并给出像素级的伪造区域定位与可视化结果。

当前实现基于 CASIA 伪造图像数据集，采用语义分割模型（如 Unet++ + EfficientNet-B3 骨干），结合 ELA（Error Level Analysis）等多种手段，提升对细微伪造区域的检测能力。

**核心能力：**
- 图像级真伪判定（真实 / 篡改）
- 像素级伪造区域分割与热力图可视化
- 训练、评估、推理（预测）全流程脚本与结果可视化

---

## 2. 特性概览

- **多模态输入**：
  - 支持将原始 RGB 图像与 ELA 噪声图进行融合，提高对低对比度篡改的感知能力。
- **像素级伪造定位**：
  - 输出 0–1 连续概率图，并可根据阈值生成二值掩码图。
- **图像级真伪判定**：
  - 基于伪造像素面积统计，给出“被篡改 / 未篡改”判定。
- **训练优化**：
  - 支持混合精度训练、梯度累积、学习率调度等策略，适配不同显存大小的 GPU。
- **评估工具链**：
  - 像素级 IoU/F1/Precision/Recall/Accuracy 等指标；
  - 图像级混淆矩阵与 F1；
  - 自动搜索最佳 F1 对应的概率阈值。
- **推理与可视化**：
  - 单图 / 批量预测脚本；
  - 自动生成多联图（原图、ELA 图、热力图、掩码等）。
- **结果归档**：
  - 训练日志保存在 `training_log.csv`；
  - 模型与划分索引保存在 `checkpoints/`；
  - 各种可视化图像与评估报告保存在 `results/`。

---

## 3. 目录结构

项目主要目录与文件说明如下（仅列出关键部分）：

```text
ForensicVision/
├── README.md                 # 英文版项目说明（默认）
├── README_cn.md              # 本文件，中文版项目说明
├── training_log.csv          # 训练日志（按 epoch 记录损失与指标）
├── checkpoints/
│   ├── best_model.pth        # 在验证集上表现最佳的模型权重
│   └── split_idx.pth         # 训练/验证集划分索引
├── data/
│   ├── Au/                   # 真实（未篡改）图像，例如 Au_ani_00001.jpg
│   ├── Tp/                   # 篡改（tampered）图像
│   └── CASIA2/               # 对应的伪造掩码/标注图
├── results/
│   ├── chart_loss_curve.png  # 训练/验证损失曲线
│   ├── chart_f1_curve.png    # 验证集 F1 曲线
│   ├── epoch_*_val.png       # 各 epoch 验证集可视化汇总
│   ├── epoch_*_val_preview.png
│   ├── evaluation_confusion_matrix.png
│   ├── final_pred_*.png      # 示例预测可视化结果
│   └── final_report.txt      # 最终评估文本报告（若脚本已生成）
├── scripts/
│   ├── train.py              # 训练脚本入口
│   ├── evaluate.py           # 评估脚本入口
│   └── predict.py            # 推理/预测脚本入口
└── src/
    ├── config.py             # 全局配置（数据路径、超参数、阈值等）
    ├── dataset.py            # 数据集定义与数据加载逻辑
    ├── ela.py                # ELA（Error Level Analysis）相关工具函数
    ├── model.py              # 模型构建（如 Unet++ + EfficientNet-B3）
    ├── utils.py              # 常用工具函数（度量计算、可视化等）
    └── __pycache__/          # Python 缓存文件（可忽略）
```

---

## 4. 环境与依赖

### 4.1 开发环境建议

- Python：3.9–3.11（按你本地环境调整）
- 操作系统：Linux / WSL / Windows（推荐有 NVIDIA GPU）
- CUDA：如使用 GPU，请确保 CUDA 与 PyTorch 版本匹配

### 4.2 主要依赖库

根据源码结构，一般需要以下依赖（实际以项目中的 `requirements.txt` 或 `pip freeze` 为准）：

- 深度学习与数据处理：
  - `torch`
  - `torchvision`
  - `segmentation_models_pytorch`
  - `numpy`
- 数据增强与图像处理：
  - `albumentations`
  - `opencv-python`
- 可视化与分析：
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tqdm`

### 4.3 安装示例

建议使用虚拟环境管理依赖（以 bash 为例）：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖（如果存在 requirements.txt）
pip install -r requirements.txt

# 若无 requirements.txt，可根据实际情况手动安装
pip install torch torchvision segmentation-models-pytorch \
    albumentations opencv-python matplotlib seaborn scikit-learn tqdm
```

> 提示：在 WSL 中运行时，路径形如 `/home/username/projects/ForensicVision`，与 Windows 盘符路径不同，请注意区分。

---

## 5. 数据准备

### 5.1 获取 CASIA 数据集

本项目默认基于 CASIA 图像伪造数据集（如 CASIA v2.0）。

- 推荐使用 Kaggle 上整理好的 CASIA 2.0 数据集版本：
  - CASIA 2.0 Image Tampering Detection Dataset（by divg07）  
    链接：https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
- 请从官方渠道或上述 Kaggle 链接获取数据集，并遵守其使用协议（通常限学术研究用途）。
- 下载完成后，将相关图像与掩码整理到本项目的 `data/` 目录下。

### 5.2 目录结构要求

建议整理为如下结构（相对于项目根目录）：

```text
data/
├── Au/         # 真实图像（Authentic），如 Au_ani_00001.jpg
├── Tp/         # 篡改图像（Tampered），对应 Au 中的原图
└── CASIA2/     # 掩码/标注图，名称与 Tp 能对应上（如加 _gt 等后缀）
```

- `Au/`：真实图片，命名一般以 `Au_...` 开头；
- `Tp/`：被篡改图片，命名一般以 `Tp_...` 开头，扩展名可能包含 `.jpg/.png/.tif` 等；
- `CASIA2/`：对应篡改掩码图，用于监督训练，命名需与 `Tp/` 中图像可匹配（具体规则见 `src/dataset.py`）。

### 5.3 配置路径

数据路径与部分超参数集中在 `src/config.py` 中：

- 典型字段（实际以文件内容为准）：
  - `DATA_ROOT`：数据根目录（默认指向项目下 `data/`）
  - `AU_DIR`：真实图像子目录名（如 `Au`）
  - `TP_DIR`：篡改图像子目录名（如 `Tp`）
  - `MASK_DIR`：掩码图像目录名（如 `CASIA2`）

如果你的数据不在默认位置，请根据实际情况修改对应配置。

---

## 6. 训练（Training）

### 6.1 训练前检查

在开始训练前，请确保：

1. 已正确放置数据到 `data/` 目录（或在 `config.py` 中修改为自己的路径）。
2. 已安装必要依赖，并可以在 Python 中导入 `torch` 等模块。
3. 根据硬件情况适当调整 `src/config.py` 中的关键超参数，例如：
   - `BATCH_SIZE`
   - `EPOCHS`
   - `LR`（学习率）
   - `IMAGE_SIZE`
   - `NUM_WORKERS`
   - 以及是否启用 AMP、梯度累积等选项（如果配置中提供）。

### 6.2 启动训练

在项目根目录下运行：

```bash
python scripts/train.py
```

脚本行为概览（以当前代码为准）：

- 构建训练集与验证集：
  - 通过 `src/dataset.py` 中的数据集类，从 `Au/`、`Tp/`、`CASIA2/` 读取样本；
  - 固定随机种子并生成 `split_idx.pth`，保证 train/val 划分可复现。
- 模型训练：
  - 构建语义分割模型（如 Unet++ + EfficientNet-B3）；
  - 使用复合损失函数、学习率调度等策略进行训练；
  - 每个 epoch 记录训练与验证指标。
- 模型与日志保存：
  - 最优模型权重保存到 `checkpoints/best_model.pth`；
  - 训练日志保存到根目录下的 `training_log.csv`；
  - 部分中间可视化结果保存在 `results/` 目录（如 `epoch_*_val.png`）。

### 6.3 监控训练过程

- 可以用任意 CSV 查看工具（如 Excel、Pandas）打开 `training_log.csv`，观察：
  - `Train_Loss` / `Val_Loss` 的变化趋势；
  - 验证集 F1、IoU 等指标是否持续提升或收敛。
- `results/chart_loss_curve.png` 与 `results/chart_f1_curve.png` 提供了直观的曲线图，用于分析是否过拟合或欠拟合。

---

## 7. 评估（Evaluation）

训练完成并生成 `checkpoints/best_model.pth` 后，可以在同一数据划分上进行更详细的评估。

### 7.1 运行评估脚本

在项目根目录下执行：

```bash
python scripts/evaluate.py
```

该脚本通常会：

1. 读取 `checkpoints/split_idx.pth`，重建原验证集划分；
2. 加载 `checkpoints/best_model.pth` 模型权重；
3. 在若干个概率阈值上评估像素级指标：
   - IoU、F1、Precision、Recall、Accuracy 等；
4. 找到像素级 F1 最高的最佳阈值 `best_thr`；
5. 基于伪造像素面积阈值（如 `IMG_TAMPER_MIN_PIXELS`）进行图像级真伪判定：
   - 计算图像级混淆矩阵与 F1；
6. 将关键结果以图像和文本形式保存到 `results/` 目录，例如：
   - `evaluation_confusion_matrix.png`
   - `final_report.txt`（若脚本中已实现）。

---

## 8. 预测 / 推理（Inference / Prediction）

### 8.1 脚本功能

`scripts/predict.py` 提供了基于训练好模型的推理功能，包括：

- 对单张图像进行伪造检测与可视化；
- 对文件夹中的多张图像批量推理（可限定数量）；
- 对结果进行多联图展示与保存（原图、ELA 图、热力图、掩码）。

### 8.2 常用参数（示例）

具体参数以脚本中的 `argparse` 定义为准，常见选项通常包括：

- `-i, --input`：输入图像或目录路径；
- `-o, --output`：输出结果保存目录（如 `results/preds/`）；
- `-c, --checkpoint`：模型权重文件路径（默认可为 `checkpoints/best_model.pth`）；
- `--limit`：当输入为目录时，最多处理的图像数量；
- `--seed`：随机抽样种子；
- `--show`：是否在推理过程中弹出窗口展示结果（如用 `matplotlib`）。

### 8.3 使用示例

在项目根目录下：

```bash
# 1) 对单张图像进行预测
python scripts/predict.py \
    -i data/Tp/Tp_S_NNN_S_B_nat00090_nat00090_11110.jpg \
    -o results/preds_single

# 2) 对一个目录中的若干图像进行预测（例如随机 50 张）
python scripts/predict.py \
    -i data/Tp \
    -o results/preds_batch \
    --limit 50 \
    --seed 42
```

推理完成后，可以在指定的输出目录中查看生成的可视化结果图片，文件名类似：

- `final_pred_0_*.png`
- `final_pred_1_*.png`

每张可视化图通常包含：

- 原始输入图；
- ELA 噪声图（如有启用）；
- 伪造概率热力图；
- 最终二值伪造掩码。

---

## 9. 结果可视化与分析

### 9.1 训练过程可视化

- `results/chart_loss_curve.png`
  - X 轴：epoch
  - Y 轴：损失值（训练/验证）
- `results/chart_f1_curve.png`
  - X 轴：epoch
  - Y 轴：验证集 F1 值

通过观察曲线，可以判断：

- 是否存在训练损失下降而验证损失上升（过拟合）；
- 指标是否趋于收敛，是否需要增加或减少训练轮数。

### 9.2 验证与最终预测示例

- `results/epoch_*_val.png` / `results/epoch_*_val_preview.png`：
  - 展示若干验证集样本的原图、真值掩码、预测结果等，用于直观对比模型在不同训练阶段的表现。
- `results/final_pred_*.png`：
  - 展示最终模型在若干样本上的预测效果，便于演示与分析。

### 9.3 量化评估结果

- `results/evaluation_confusion_matrix.png`：
  - 图像级真伪判定的混淆矩阵；
  - 行通常表示真实标签（真实/篡改），列表示模型预测结果；
  - 对角线元素代表预测正确的数量。
- `results/final_report.txt`（若生成）：
  - 文本形式总结：
    - 像素级 IoU/F1/Precision/Recall/Accuracy；
    - 图像级 Accuracy/F1 等关键指标；
    - 最佳阈值等信息。

---

## 10. 常见问题（FAQ）

### 10.1 显存不足 / CUDA OOM

- 尝试减小 `BATCH_SIZE`；
- 增大梯度累积步数（如果在 `config.py` 中有类似参数，如 `GRAD_ACCUM_STEPS`）；
- 如无 GPU，可将设备切换为 CPU（训练速度会明显变慢）。

### 10.2 Windows / WSL 多进程 DataLoader 报错

- 在 `config.py` 或构建 DataLoader 时，将 `NUM_WORKERS` 设置为 `0` 或较小值；
- 在 WSL/Windows 环境下，建议从较小的 `NUM_WORKERS` 逐步调试。

### 10.3 找不到数据或 checkpoint

- 确认 `data/` 目录下存在 `Au/`、`Tp/`、`CASIA2/` 子目录；
- 确认 `src/config.py` 中的数据路径与实际一致；
- 若运行评估/预测时报错找不到 `best_model.pth`，请先完成一次训练或手动放置模型权重到 `checkpoints/` 目录下。

---

## 11. 后续工作与扩展方向（可选）

- 支持更多类型文档与多源数据集的联合训练；
- 尝试更强的骨干网络或多任务学习（例如同时做篡改分类与定位）；
- 将训练与推理流程封装为简易 Web Demo 或命令行工具。

---