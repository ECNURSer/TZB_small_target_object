# YOLO26-OBB 遥感车辆检测

本项目使用 Ultralytics 官方 YOLO26 OBB 模型完成 10 类遥感车辆旋转框检测，分别支持：

- `yolo26n-obb.pt`
- `yolo26s-obb.pt`
- `yolo26m-obb.pt`

源码基于 `ultralytics v8.4.80`，位于 `ultralytics_src/`。n/s 配置保持官方 OBB Loss；m 精度实验在官方 box、DFL 和 angle loss 不变的基础上，为分类分支增加了可选的 class-balanced Focal Loss。官方 YOLO26 OBB 权重在第一次训练时自动下载，三个模型均为 DOTAv1 预训练权重。

## 当前状态

- 项目代码和 Conda 环境脚本已建立。
- `dataset` 符号链接指向 `/data/work1/00_data/TZB/subject1/dataset`，原图读取自同级 `input_path`；不复制 25GB 原图。
- 当前已有 fold 0 的 YOLO26n 训练结果；该实验续训时修改过输入尺寸和 batch，只作为探索结果，不作为严格公平基线。
- YOLO26m 当前是面向小目标和长尾类别的增强实验，使用 `imgsz=1024`、全局 batch 72、AdamW 和 class-balanced Focal Loss。
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

# fold 0 首轮公平对比：依次运行，三者参数保持一致
python train.py --model n --fold 0 --device 4,5,6,7 --batch 32
python train.py --model s --fold 0 --device 4,5,6,7 --batch 32
python train.py --model m --fold 0 --device 4,5,6,7 --batch 32

# 覆盖配置
python train.py --model m --fold 0 --batch 32 --imgsz 1024 --device 0,1,2,3

# 覆盖早停和周期权重保存间隔
python train.py --model n --fold 0 --patience 50 --save-period 5

# 仅检查最终参数和数据路径
python train.py --model n --fold 0 --dry-run

# 通过统一入口运行
bash run.sh train-n --fold 0
```

### YOLO26m 增强实验

`configs/yolo26m_obb.yaml` 当前采用以下关键设置：

```text
epochs=1500, imgsz=1024, batch=72, optimizer=AdamW, cos_lr=True
patience=200, save_period=50
focal_gamma=1.5, focal_alpha=0.25, cls_pw=0.25
mosaic=0.25, scale=0.25, flipud=0.5, close_mosaic=50
```

其中 `cls_pw=0.25` 根据训练集类别频率计算温和的逆频率权重；`focal_gamma=1.5` 抑制容易负样本。将 `focal_gamma` 和 `cls_pw` 都设为 `0.0` 即恢复官方 BCE 分类损失。该实验与官方 Loss 基线的参数不同，不能直接当作只比较模型规模的 n/s/m 公平实验。

正式训练由用户手动启动：

```bash
bash run.sh train-m --fold 0 --name yolo26m_obb_fold0_balanced_focal
```

### 使用 tmux 后台训练

启动前先确认 4-7 号 GPU 空闲：

```bash
nvidia-smi
```

创建训练会话：

```bash
tmux new -s yolo26m_train
```

进入 tmux 后加载环境并启动训练，同时将终端输出保存到独立日志：

```bash
source /home/dihan/miniconda3/etc/profile.d/conda.sh
conda activate yolo26-obb
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0

bash run.sh train-m \
  --fold 0 \
  --name yolo26m_obb_fold0_balanced_focal \
  2>&1 | tee /home/dihan/yolo26m_obb_fold0_balanced_focal.log
```

让训练留在后台并退出 tmux：先按 `Ctrl+B`，松开后再按 `D`。SSH 断开不会终止训练。

```bash
# 查看全部会话
tmux ls

# 重新进入训练会话
tmux attach -t yolo26m_train

# 不进入会话，直接查看训练日志
tail -f /home/dihan/yolo26m_obb_fold0_balanced_focal.log
```

正常停止训练时，先进入会话再按 `Ctrl+C`，等待进程退出。仅在进程无法正常退出时强制删除会话：

```bash
tmux kill-session -t yolo26m_train
```

训练生成 events 文件后，可以创建单独的 TensorBoard 会话：

```bash
tmux new -s yolo26m_tb
source /home/dihan/miniconda3/etc/profile.d/conda.sh
conda activate yolo26-obb
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0

bash run.sh tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26m_obb_fold0_balanced_focal \
  --port 6007
```

不要在同一组 GPU 上同时启动 n/s/m。显存不足时应把三个实验统一降到 `--batch 16`，保证比较条件一致。

训练目录为 `runs/yolo26{n|s|m}_obb_fold{fold}/`。断点续训：

配置默认启用早停；n/s 使用 `patience=100, save_period=10`，m 增强实验使用 `patience=200, save_period=50`。`last.pt` 持续覆盖最新状态，`best.pt` 在验证 fitness 提升时覆盖。

```bash
python train.py --model n --fold 0 --resume

# 从指定的周期权重恢复
python train.py --model n --fold 0 --resume runs/yolo26n_obb_fold0/weights/epoch100.pt
```

训练完整结束后，Ultralytics 会 strip `last.pt`/`best.pt` 中的 optimizer 和 epoch 状态，此时二者只能用于评估或微调，不能断点续训。训练入口会检查 checkpoint 的 `epoch` 和 optimizer 状态；对已 strip 的权重执行 `--resume` 会直接报错，不会回退到默认数据集重新训练。若要扩展已完成实验，只能从仍含 optimizer 的最近 `epochN.pt` 恢复，并通过 `--epochs` 指定新的总轮数。

## TensorBoard

训练开始后，目标实验目录会生成 `events.out.tfevents.*`。TensorBoard 必须明确指定包含该文件的目录：

```bash
# YOLO26n
bash run.sh tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0 \
  --port 6006

# 等价的原生命令
tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0 \
  --port 6006 --bind_all

# 浏览器访问 http://服务器地址:6006
```

YOLO26s 和 YOLO26m 分别将 `--logdir` 改为 `runs/yolo26s_obb_fold0` 和 `runs/yolo26m_obb_fold0`。`run.sh` 会检查目录和 events 文件；未开始训练时会明确报错。

如果服务器端口不对外开放，可通过 SSH 转发：

```bash
ssh -L 6006:localhost:6006 user@server
```

当前 Ultralytics 版本无法稳定 trace YOLO26 OBB 的字典输出，因此项目关闭了 TensorBoard model graph tracing；训练 loss、学习率、Precision、Recall 和 mAP 标量监控保持启用。

## runs 目录内容

`runs/` 只保存运行产物，不保存源码或原始数据。正式训练后每个实验目录通常包含：

```text
runs/yolo26n_obb_fold0/
├── events.out.tfevents.*      # TensorBoard 标量日志
├── args.yaml                  # 本次训练的完整参数
├── results.csv                # 每个 epoch 的 loss 和评估指标
├── weights/
│   ├── best.pt                # 验证指标最优权重
│   ├── last.pt                # 最新训练状态
│   └── epoch*.pt              # 每 10 轮保存的周期权重
├── val_class_metrics.csv      # 逐类别验证指标
└── *.png / *.jpg              # 曲线、混淆矩阵和样本可视化
```

运行 `evaluate_test.py` 会产生 `runs/test/`，运行 `predict.py` 会产生 `runs/predict/`。之前的 `smoke`、`smoke_dataset`、`smoke_predict` 和 `smoke_test` 均为代码验证产物，现已删除。

## n/s/m 单折对比与最终选择

1. 只在 fold 0 分别训练 n/s/m，训练参数、数据、seed、输入尺寸和 batch 必须一致。
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
  --model n --fold 0 \
  --weights runs/yolo26n_obb_fold0/weights/best.pt \
  --device 0
```

无标注图片或视频推理：

```bash
python predict.py \
  --weights runs/yolo26n_obb_fold0/weights/best.pt \
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
