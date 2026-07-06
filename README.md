# YOLO26-OBB 遥感车辆检测

本项目使用 Ultralytics 8.4.80 的官方 YOLO26 OBB 架构和 n/s/m 预训练权重，完成 10 类遥感车辆旋转框检测。Ultralytics 源码固定在 `ultralytics_src/`，训练增加了可选的 class-balanced Focal Loss、TensorBoard 标量监控、周期权重和 NaN/Inf/EMA 恢复保护。

## 当前结论

n/s/m 均已完成 fold 0 训练和比赛评分基线。比赛主指标是全数据集 `F1@polygon-IoU0.3`，不是 mAP 或 Ultralytics 派生 F1。

| 模型 | Epoch | imgsz | mAP50 | mAP50-95 | 比赛 F1@0.3 |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 700 | 1024 | 0.4474 | 0.3143 | 0.6484 |
| YOLO26s | 500 | 1024 | 0.5357 | 0.3938 | 0.7035 |
| YOLO26m | 500 | 1024 | 0.5994 | 0.4463 | 0.7290 |

当前主方案使用 YOLO26m `best.pt`：

```text
imgsz=1280
max_det=600
conf=0.3250801563
device=7  # 单卡评估时的物理 GPU
```

- fold 0 val：Precision=0.75373，Recall=0.75713，F1=0.755425。
- 独立 test：Precision=0.74805，Recall=0.75690，F1=0.752449。
- test 使用 val 上冻结的 conf，没有在 test 上重新寻优。

完整 TP/FP/FN、参数搜索和逐类别结果见 [results/COMPETITION_RESULTS.md](results/COMPETITION_RESULTS.md)。评分 PDF 未说明全部匹配细节，本地评分器当前按同类别、置信度降序、GT 一对一匹配，IoU >= 0.3 记为 TP。

## 目录

```text
configs/                    n/s/m 训练配置
dataset -> /data/...        原始 fold JSON 符号链接
dataset_yolo/               转换后的 YOLO OBB 数据
docs/                       实验计划、系统信息和故障复盘
results/                    可提交的结果文档
runs/                       训练、评估、TensorBoard 和权重产物
tests/                      项目回归测试
tools/                      环境检查、冒烟训练和诊断结果汇总
ultralytics_src/            项目固定的 Ultralytics 8.4.80 源码
competition_scoring.py      OBB polygon IoU 和比赛 F1 评分核心
evaluate_competition.py     预测缓存、conf 搜索和固定阈值评分
convert_to_yolo.py          JSON 到 YOLO OBB 数据转换
train.py                    训练和断点续训入口
evaluate_test.py            Ultralytics mAP 诊断评估
predict.py                  无标注图像/视频推理
run.sh                      统一命令入口
```

## 环境

环境名称为 `yolo26-obb`。当前实测组合是 Python 3.11.15、PyTorch 2.11.0+cu126、TorchVision 0.26.0+cu126、Ultralytics 8.4.80。

```bash
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0
bash setup_env.sh
conda activate yolo26-obb
python tools/check_env.py
python -m pip check
```

详细硬件和 CUDA 信息见 [docs/SYSTEM_INFO.md](docs/SYSTEM_INFO.md)。GPU 占用会变化，每次训练或推理前都应运行 `nvidia-smi`。

## 数据

原始数据位于：

```text
/data/work1/00_data/TZB/subject1/
├── dataset/       fold_0..4 的 train/val JSON 和 test.json
├── input_path/    TIFF/GeoTIFF 原图
└── gt/            原始 XML
```

项目只使用 TIFF 像素，忽略地理坐标、投影和仿射信息。RGBA 原图按三通道 BGR 输入模型。转换规则与 YOLO11 对比项目保持一致：直接归一化四点坐标，不重排顶点、不额外裁剪或过滤标注。

```bash
# 转换全部 fold 和 test
python convert_to_yolo.py --all
```

默认为原图创建符号链接，不复制大体积 TIFF。使用 `--copy-images` 才会复制图像。当前 fold 0 包含 6792 张 train、1703 张 val，独立 test 包含 944 张。

## 训练

三个配置都使用 AdamW、cosine LR、`degrees=15`、`mixup=0`、`mosaic=0.25`、AMP 和 balanced focal。默认 GPU 为 4-7 号卡，必须根据当时资源通过 `--device` 覆盖。

```bash
# 检查最终参数，不加载权重、不训练
python train.py --model m --fold 0 --dry-run

# 训练
bash run.sh train-n --fold 0 --name yolo26n_obb_fold0_balanced_focal_700ep_b64
bash run.sh train-s --fold 0 --name yolo26s_obb_fold0_balanced_focal_b64
bash run.sh train-m --fold 0 --name yolo26m_obb_fold0_balanced_focal

# 命令行覆盖资源参数
python train.py --model m --fold 0 --batch 64 --device 4,5,6,7
```

早停由 `patience` 控制。`last.pt` 保存最新状态，`best.pt` 在验证 fitness 提升时覆盖，`save_period` 额外保留 `epochN.pt`。

### 断点续训

```bash
# 自动定位当前实验 last.pt
python train.py --model n --fold 0 \
  --name yolo26n_obb_fold0_balanced_focal_700ep_b64 --resume

# 指定周期 checkpoint
python train.py --model n --fold 0 \
  --resume runs/yolo26n_obb_fold0_balanced_focal_700ep_b64/weights/epoch600.pt
```

已完成训练的 `best.pt`/`last.pt` 会被 strip，不含 optimizer，不能用于断点续训。此时必须使用仍含 optimizer 的 `epochN.pt`。

### tmux

```bash
tmux new -s yolo26_train
source /home/dihan/miniconda3/etc/profile.d/conda.sh
conda activate yolo26-obb
cd /home/dihan/TZB-subject1-YOLO26-OBBV1.0
bash run.sh train-n --fold 0 \
  --name yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  2>&1 | tee /home/dihan/yolo26n_obb_fold0_balanced_focal_700ep_b64.log
```

按 `Ctrl+B`，再按 `D` 退出会话。使用 `tmux attach -t yolo26_train` 返回，使用 `tmux kill-session -t yolo26_train` 终止会话。

## TensorBoard

必须将 `--logdir` 指向包含 `events.out.tfevents.*` 的具体实验目录：

```bash
bash run.sh tensorboard \
  --logdir /home/dihan/TZB-subject1-YOLO26-OBBV1.0/runs/yolo26n_obb_fold0_balanced_focal_700ep_b64 \
  --port 6007
```

本地浏览器通过 SSH 转发访问：

```bash
ssh -L 6007:localhost:6007 user@server
# 打开 http://localhost:6007
```

TensorBoard 保留 loss、学习率、Precision、Recall 和 mAP 标量；YOLO26 OBB 字典输出的 model graph tracing 已关闭。

## 评估

### Ultralytics 诊断指标

`evaluate_test.py` 输出 mAP50、mAP50-95、Precision 和 Recall，用于训练诊断，不代替比赛 F1。

```bash
bash run.sh test \
  --model m --fold 0 \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --imgsz 1024 --device 7
```

训练和该诊断评估会追加 `results/experiments.csv`。`bash run.sh summary` 可将其生成为 `results/TRAINING_DIAGNOSTICS.md`；这两者都不用于比赛排名。

### 比赛 F1

比赛评分入口只接受单个 GPU 编号。传入 `4,5,6,7` 会直接报错。

```bash
# val：生成低阈值预测缓存并精确搜索全局 conf
bash run.sh competition \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --fold 0 --split val --device 7 --batch 8 --chunk-size 8 \
  --imgsz 1280 --min-conf 0.05 --max-det 600 \
  --cache runs/competition/yolo26m_fold0_img1280_maxdet600.json \
  --output runs/competition/yolo26m_fold0_img1280_maxdet600_metrics.json

# test：使用 val 冻结的 conf，禁止在 test 上重新寻优
bash run.sh competition \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --fold 0 --split test --device 7 --batch 8 --chunk-size 8 \
  --imgsz 1280 --min-conf 0.05 --max-det 600 \
  --fixed-conf 0.32508015632629395 \
  --cache runs/competition/yolo26m_test_img1280_maxdet600.json \
  --output runs/competition/yolo26m_test_img1280_maxdet600_fixedconf_metrics.json
```

YOLO26 OBB 检测头为 end-to-end NMS-free，`nms_iou` 对当前权重不生效。`GPU 0` 出现在部分日志中时，可能是物理 GPU 7 经 `CUDA_VISIBLE_DEVICES` 重映射后的进程内 `cuda:0`。

## 无标注推理

`predict.py` 默认使用当前主方案的 `imgsz=1280、conf=0.325080、max_det=600`。

```bash
bash run.sh predict \
  --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt \
  --source /path/to/images --device 7 --name final_predict
```

## 文档

- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)：比赛优化顺序和未完成任务。
- [docs/SYSTEM_INFO.md](docs/SYSTEM_INFO.md)：服务器、CUDA 和 Python 环境。
- [docs/YOLO26M_TRAINING_INCIDENT.md](docs/YOLO26M_TRAINING_INCIDENT.md)：YOLO26m epoch 415 后 EMA/NaN 异常复盘。
- [results/COMPETITION_RESULTS.md](results/COMPETITION_RESULTS.md)：n/s/m、参数搜索和独立 test 结果。

## 测试

```bash
conda activate yolo26-obb
pytest -q tests
```

仅运行项目根目录 `tests/`。直接运行无路径的 `pytest` 会额外收集 `ultralytics_src/tests/` 中的大量上游训练和导出测试。
