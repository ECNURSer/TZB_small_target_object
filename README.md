# YOLO26-OBB 遥感车辆检测

本项目使用 Ultralytics 官方 YOLO26 OBB 模型完成 10 类遥感车辆旋转框检测，分别支持：

- `yolo26n-obb.pt`
- `yolo26s-obb.pt`
- `yolo26m-obb.pt`

源码基于 `ultralytics v8.4.80`，位于 `ultralytics_src/`。s/m 精度实验在官方 box、DFL 和 angle loss 不变的基础上，为分类分支增加了可选的 class-balanced Focal Loss。官方 YOLO26 OBB 权重在第一次训练时自动下载，三个模型均使用官方 OBB 预训练权重初始化。

## 当前状态

- 项目代码和 Conda 环境脚本已建立。
- `dataset` 符号链接指向 `/data/work1/00_data/TZB/subject1/dataset`，原图读取自同级 `input_path`；不复制 25GB 原图。
- 原 YOLO26n 探索实验的 runs、日志和统一结果记录已删除，下一轮使用新的 700 epoch 优化配置重新训练。
- YOLO26s 已完成 500 epoch 基线复评，作为 26n 参数选择和模型对比依据。
- YOLO26m 已完成 fold 0 复评，mAP50 为 0.5994，mAP50-95 为 0.4463。
- 独立 test 评估要求 `test.json` 中包含真实 OBB 标注；无标注测试集只能运行 `predict.py`。

## 目录结构

```text
TZB-subject1-YOLO26-OBBV1.0/
├── configs/                    # n/s/m 三套训练配置
├── dataset -> /data/.../dataset # 外部五折 JSON 和 test.json
├── dataset_yolo/               # 转换后数据，脚本生成
├── docs/                       # 硬件和实验说明
├── results/                    # 统一实验结果表
├── tests/                      # 项目脚本测试
├── tools/                      # 结果汇总工具
├── ultralytics_src/            # 官方 Ultralytics v8.4.80
├── convert_to_yolo.py
├── train.py
├── evaluate_test.py
├── predict.py
├── plot_metrics.py
├── run.sh
└── setup_env.sh
```

## 环境

环境名称为 `yolo26-obb`：

```bash
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0
bash setup_env.sh
conda activate yolo26-obb
```

当前实测环境为 Python 3.11.15、`torch 2.11.0+cu126`、`torchvision 0.26.0+cu126`、Ultralytics 8.4.80，CUDA 可用且可识别 8 张 A100。运行以下命令复核：

```bash
conda activate yolo26-obb
python tools/check_env.py
python -m pip check
```

## 数据格式

实际标注和图像路径如下：

```text
/data/work1/00_data/TZB/subject1/
├── dataset/                   # 五折 JSON 和 test.json
├── input_path/                # 9445 张 TIFF 原图
└── gt/                        # 9445 个原始 XML，当前转换不重复解析
```

JSON 结构为：

```text
dataset/
├── fold_0/train.json
├── fold_0/val.json
├── ...
├── fold_4/train.json
├── fold_4/val.json
└── test.json                 # 可选；独立评估时必须含标注
```

JSON 每条记录需要包含 `data_path`、`lab` 和四点或闭合五点 `points`。转换后的标签格式为：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

转换全部 fold：

```bash
python convert_to_yolo.py --all
```

JSON 中 `data_path` 已是 `input_path/*.tif` 的绝对路径。转换默认创建指向原图的符号链接；需要独立复制数据时使用 `--copy-images`。

原图是带 GeoTIFF 标签的 RGBA TIFF，但 Alpha 全为 255。项目与 YOLO11 一样只加载 RGB 像素并转换为三通道 BGR，完全忽略地理坐标、投影和仿射标签。转换器沿用 YOLO11 的直接归一化规则，不裁剪坐标、不重排顶点、不额外过滤标注。

## 训练

```bash
# 先查看空闲 GPU；当前复核时 4-7 号卡空闲
nvidia-smi

# 当前准备运行的 YOLO26n 优化实验
bash run.sh train-n --fold 0 --name yolo26n_obb_fold0_balanced_focal_700ep_b64

# 覆盖配置
python train.py --model m --fold 0 --batch 32 --imgsz 1024 --device 0,1,2,3

# 仅检查最终参数和数据路径
python train.py --model n --fold 0 --dry-run

# 通过统一入口运行
bash run.sh train-n --fold 0
```

### YOLO26m 增强实验

`configs/yolo26m_obb.yaml` 当前采用以下关键设置：

```text
epochs=500, imgsz=1024, batch=72, optimizer=AdamW, cos_lr=True
patience=200, save_period=100
focal_gamma=1.5, focal_alpha=0.25, cls_pw=0.25
mosaic=0.25, scale=0.25, flipud=0.5, close_mosaic=50
```

其中 `cls_pw=0.25` 根据训练集类别频率计算温和的逆频率权重；`focal_gamma=1.5` 抑制容易负样本。将 `focal_gamma` 和 `cls_pw` 都设为 `0.0` 即恢复官方 BCE 分类损失。该实验与官方 Loss 基线的参数不同，不能直接当作只比较模型规模的 n/s/m 公平实验。

正式训练由用户手动启动：

```bash
bash run.sh train-m --fold 0 --name yolo26m_obb_fold0_balanced_focal
```

### YOLO26m fold 0 最终结果

本次实验实际完成 500 epoch，训练耗时 8.628 小时。训练结束时服务器环境错误地使用了 `TkAgg`，导致自动最终绘图退出；权重保存不受影响。2026-07-05 使用 `best.pt`、`imgsz=1024` 在 fold 0 验证集重新评估，评估集包含 1703 张图像和 86298 个实例。单卡评估的 `batch=18` 只影响资源占用，不影响精度指标。

训练记录中最高 mAP50-95 出现在 epoch 414。`results.csv` 后段包含验证异常产生的零值和 NaN，因此 `results.png` 保留这些原始异常点；下表不采用最后一行，而以 `best.pt` 的独立复评结果为最终指标。

| 权重 | Precision | Recall | F1 | mAP50 | mAP50-95 | 推理耗时/图 |
|---|---:|---:|---:|---:|---:|---:|
| `weights/best.pt` | 0.6494 | 0.5829 | 0.6143 | 0.5994 | 0.4463 | 10.61 ms |

逐类别结果：

| 类别 | 实例数 | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Bus | 356 | 0.7494 | 0.6770 | 0.7113 | 0.7102 | 0.6019 |
| Cargo Truck | 2983 | 0.7457 | 0.6527 | 0.6961 | 0.7079 | 0.5982 |
| Dump Truck | 6452 | 0.6848 | 0.6841 | 0.6845 | 0.7205 | 0.5768 |
| Excavator | 233 | 0.6223 | 0.6294 | 0.6259 | 0.6182 | 0.4721 |
| Small Car | 37730 | 0.7074 | 0.7969 | 0.7495 | 0.8015 | 0.5255 |
| Tractor | 27 | 0.5086 | 0.4603 | 0.4832 | 0.4059 | 0.3251 |
| Trailer | 260 | 0.4999 | 0.3577 | 0.4170 | 0.3265 | 0.2696 |
| Truck Tractor | 256 | 0.7744 | 0.6758 | 0.7218 | 0.7392 | 0.4889 |
| Van | 36841 | 0.7226 | 0.7603 | 0.7410 | 0.7913 | 0.4754 |
| other-vehicle | 1160 | 0.4789 | 0.1347 | 0.2103 | 0.1728 | 0.1300 |

最终评估与绘图产物位于：

```text
runs/yolo26m_obb_fold0_balanced_focal/
├── results.png                  # 500 epoch 训练曲线
└── final_eval/
    ├── metrics.json             # 总体指标
    ├── class_metrics.csv         # 逐类别指标
    ├── BoxPR_curve.png           # PR 曲线
    ├── BoxP_curve.png            # Precision-Confidence 曲线
    ├── BoxR_curve.png            # Recall-Confidence 曲线
    ├── BoxF1_curve.png           # F1-Confidence 曲线
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    └── val_batch*_pred.jpg       # 验证集预测样例
```

服务器运行入口默认设置 `MPLBACKEND=Agg`，后续训练和评估可以在无桌面环境中正常保存图表。

### 训练异常复盘

epoch 415 后的指标归零不是正常过拟合，而是中间 batch 出现 NaN 后污染了 EMA，导致后续 EMA 验证失效。EMA、batch、AMP 的含义、故障时间线、当前恢复逻辑缺陷和下一次训练处理顺序见 [YOLO26m 训练异常复盘](docs/YOLO26M_TRAINING_INCIDENT.md)。

训练器现已增加梯度、AMP、模型缓冲区和 EMA 的有限性检查，并支持 DDP 全 rank 同步回滚。后续实验使用全局 batch 64；若仍可复现 NaN，再降到 48 或关闭 AMP 进行 FP32 对照。降低 batch 和关闭 AMP 只能降低数值异常概率，不能修复已经损坏的 EMA。

### YOLO26n 优化实验配置

已完成的 YOLO26s 500 epoch 基线在 fold 0 复评得到 Precision=0.6071、Recall=0.5039、F1=0.5507、mAP50=0.5357、mAP50-95=0.3938，最高训练记录出现在 epoch 486，说明 500 epoch 时仍有缓慢提升空间。

新的 YOLO26n 配置参考上述 s/m 实测结果，使用 `epochs=700`、`imgsz=1024`、全局 `batch=64`、AdamW、余弦学习率和 balanced focal loss，并在最后 70 epoch 关闭 mosaic。核心 loss、输入尺寸、数据和增强策略与 s/m 实验一致；每 50 epoch 保存周期权重。

当前复评对比：

| 模型 | Epoch | mAP50 | mAP50-95 | Precision | Recall | 推理 ms/图 | 参数量 M |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26s 基线 | 500 | 0.5357 | 0.3938 | 0.6071 | 0.5039 | 3.29 | 9.76 |
| YOLO26m | 500 | 0.5994 | 0.4463 | 0.6494 | 0.5829 | 10.61 | 21.21 |

YOLO26m 当前精度更高；YOLO26s 推理更快、参数量更小。700 epoch 的新 n 实验完成后，再把 n/s/m 一起纳入最终选择。

### 使用 tmux 后台训练

启动前先确认 4-7 号 GPU 空闲：

```bash
nvidia-smi
```

创建 YOLO26n 训练会话：

```bash
tmux new -s yolo26n_train
```

进入 tmux 后加载环境并启动训练，同时将终端输出保存到独立日志：

```bash
source /home/dihan/miniconda3/etc/profile.d/conda.sh
conda activate yolo26-obb
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0

bash run.sh train-n \
  --fold 0 \
  --name yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  2>&1 | tee /home/dihan/yolo26n_obb_fold0_balanced_focal_700ep_b64.log
```

让训练留在后台并退出 tmux：先按 `Ctrl+B`，松开后再按 `D`。SSH 断开不会终止训练。

```bash
# 查看全部会话
tmux ls

# 重新进入训练会话
tmux attach -t yolo26n_train

# 不进入会话，直接查看训练日志
tail -f /home/dihan/yolo26n_obb_fold0_balanced_focal_700ep_b64.log
```

正常停止训练时，先进入会话再按 `Ctrl+C`，等待进程退出。仅在进程无法正常退出时强制删除会话：

```bash
tmux kill-session -t yolo26n_train
```

训练生成 events 文件后，可以创建单独的 TensorBoard 会话：

```bash
tmux new -s yolo26n_tb
source /home/dihan/miniconda3/etc/profile.d/conda.sh
conda activate yolo26-obb
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0

bash run.sh tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  --port 6007
```

不要在同一组 GPU 上同时启动多个训练任务。若出现 OOM，优先把 YOLO26n 降到 `--batch 48`。

训练目录由 `--name` 决定。本次目录为 `runs/yolo26n_obb_fold0_balanced_focal_700ep_b64/`。断点续训：

配置默认启用早停；n 优化实验和 s 基线使用 `patience=200, save_period=50`，m 增强实验使用 `patience=200, save_period=100`。`last.pt` 持续覆盖最新状态，`best.pt` 在验证 fitness 提升时覆盖。

```bash
python train.py --model n --fold 0 --name yolo26n_obb_fold0_balanced_focal_700ep_b64 --resume

# 从指定的周期权重恢复
python train.py --model n --fold 0 --resume runs/yolo26n_obb_fold0_balanced_focal_700ep_b64/weights/epoch100.pt
```

训练完整结束后，Ultralytics 会 strip `last.pt`/`best.pt` 中的 optimizer 和 epoch 状态，此时二者只能用于评估或微调，不能断点续训。训练入口会检查 checkpoint 的 `epoch` 和 optimizer 状态；对已 strip 的权重执行 `--resume` 会直接报错，不会回退到默认数据集重新训练。若要扩展已完成实验，只能从仍含 optimizer 的最近 `epochN.pt` 恢复，并通过 `--epochs` 指定新的总轮数。

## TensorBoard

训练开始后，目标实验目录会生成 `events.out.tfevents.*`。TensorBoard 必须明确指定包含该文件的目录：

```bash
# YOLO26n 700 epoch 优化实验
bash run.sh tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  --port 6007

# 等价的原生命令
tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  --port 6007 --bind_all

# 浏览器访问 http://服务器地址:6007
```

`run.sh` 会检查目录和 events 文件；新实验尚未生成 events 文件时会明确报错。

如果服务器端口不对外开放，可通过 SSH 转发：

```bash
ssh -L 6007:localhost:6007 user@server
```

当前 Ultralytics 版本无法稳定 trace YOLO26 OBB 的字典输出，因此项目关闭了 TensorBoard model graph tracing；训练 loss、学习率、Precision、Recall 和 mAP 标量监控保持启用。

## runs 目录内容

`runs/` 只保存运行产物，不保存源码或原始数据。正式训练后每个实验目录通常包含：

```text
runs/yolo26n_obb_fold0_balanced_focal_700ep_b64/
├── events.out.tfevents.*      # TensorBoard 标量日志
├── args.yaml                  # 本次训练的完整参数
├── results.csv                # 每个 epoch 的 loss 和评估指标
├── weights/
│   ├── best.pt                # 验证指标最优权重
│   ├── last.pt                # 最新训练状态
│   └── epoch*.pt              # 每 50 轮保存的周期权重
├── val_class_metrics.csv      # 逐类别验证指标
└── *.png / *.jpg              # 曲线、混淆矩阵和样本可视化
```

运行 `evaluate_test.py` 会产生 `runs/test/`，运行 `predict.py` 会产生 `runs/predict/`。之前的 `smoke`、`smoke_dataset`、`smoke_predict` 和 `smoke_test` 均为代码验证产物，现已删除。

## n/s/m 单折对比与最终选择

1. 当前结果表已有 fold 0 的 s/m；新的 26n 实验完成后加入同一结果表。
2. 汇总 fold 0 验证结果，不用 test 集反复调参：

```bash
bash run.sh compare --stage train_val --output results/FOLD0_MODEL_COMPARISON.md
```

3. 以 fold 0 验证 mAP50-95 为主指标。若候选模型差距小于 0.005，优先选择参数更少、推理更快的模型。
4. 架构和训练参数确定后，只对最终模型运行一次独立 test 评估，避免 test 数据泄漏。

最终记录的指标包括：mAP50-95、mAP50、Precision、Recall、F1、各类别 AP、单图推理耗时和参数量。单折实验不能估计跨划分标准差。训练结果追加到 `results/experiments.csv`；每次训练和 test 的逐类别指标分别写入 `val_class_metrics.csv`、`test_class_metrics.csv`。

## Test 评估与推理

有标注 test 集：

```bash
python evaluate_test.py \
  --model m --fold 0 \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --device 0
```

无标注图片或视频推理：

```bash
python predict.py \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --source /path/to/images \
  --device 0
```

训练验证和独立 test 指标都会追加到 `results/experiments.csv`。生成统一 Markdown 表：

```bash
python tools/summarize_results.py
```

## 参考

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics OBB 任务](https://docs.ultralytics.com/tasks/obb/)
- [Ultralytics OBB 数据格式](https://docs.ultralytics.com/datasets/obb/)
