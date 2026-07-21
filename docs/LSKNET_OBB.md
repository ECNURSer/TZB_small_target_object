# LSKNet-OBB 接入说明

## 方案定位

本项目新增两个可直接训练的模型：

| 模型 | 骨干参数 | 全模型参数 | 特征层 | 推荐用途 |
|---|---:|---:|---|---|
| LSKNet-T-OBB | 3.99M | 9.67M | P2/4、P3/8、P4/16、P5/32 | 快速验证模型方向 |
| LSKNet-S-OBB | 14.35M | 20.27M | P2/4、P3/8、P4/16、P5/32 | 主精度实验 |

骨干结构和 ImageNet-300 epoch 预训练权重来自 [zcablii/LSKNet](https://github.com/zcablii/LSKNet)。官方检测配置是 LSKNet + FPN + Oriented R-CNN，基于 MMRotate 0.3.4、MMDetection 2.x 和 MMCV-Full。这里为了兼容当前项目的 PyTorch 2.11、YOLO OBB 数据、断点续训、预测缓存和比赛评分器，使用的是：

```text
官方 LSKNet-T/S backbone
          ↓
P2-P5 PAN/FPN neck
          ↓
Ultralytics OBB head
```

因此模型名称是 `LSKNet-T/S-OBB（Ultralytics head）`，不是官方 Oriented R-CNN 复现。增加 P2/4 检测层是针对本项目绝大多数目标小于 16 像素的特点。

LSKNet 上游代码和权重采用 CC BY-NC 4.0，仅允许非商业使用；商业用途需要取得作者许可。

## 训练

项目的训练、监控和评估命令统一由 `tmux_job.sh` 后台启动。每个任务都必须使用唯一会话名，并将终端输出保存到 `logs/`。

先通过 tmux 检查配置，不下载权重、不启动训练：

```bash
bash tmux_job.sh start lsknet_t_dryrun logs/lsknet_t_dryrun.log \
  bash run.sh train-lsk-t --fold 0 --dry-run

bash tmux_job.sh start lsknet_s_dryrun logs/lsknet_s_dryrun.log \
  bash run.sh train-lsk-s --fold 0 --dry-run
```

建议先训练 T 版确认收益，T 版结束后再启动 S 版；默认配置都会占用后四张 GPU（4、5、6、7），不能同时运行：

```bash
bash tmux_job.sh start lsknet_t_train logs/lsknet_t_obb_full_fold0_img1280.log \
  bash run.sh train-lsk-t \
  --fold 0 --no-test \
  --name lsknet_t_obb_full_fold0_img1280

bash tmux_job.sh start lsknet_s_train logs/lsknet_s_obb_full_fold0_img1280.log \
  bash run.sh train-lsk-s \
  --fold 0 --no-test \
  --name lsknet_s_obb_full_fold0_img1280
```

默认会从 LSKNet 官方 Hugging Face 地址下载预训练骨干，并缓存到 PyTorch Hub 的 `checkpoints` 目录。也可以使用本地文件：

```bash
bash tmux_job.sh start lsknet_t_local_pretrain logs/lsknet_t_local_pretrain.log \
  bash run.sh train-lsk-t \
  --fold 0 --no-test \
  --name lsknet_t_obb_local_pretrain \
  --pretrained-backbone /path/to/lsk_t_backbone.pth.tar
```

仅用于消融实验时可从随机初始化开始：

```bash
bash tmux_job.sh start lsknet_t_scratch logs/lsknet_t_scratch.log \
  bash run.sh train-lsk-t \
  --fold 0 --no-test \
  --name lsknet_t_obb_scratch \
  --pretrained-backbone none
```

tmux 会话管理：

```bash
# 查看全部项目任务
bash tmux_job.sh status

# 进入训练终端；按 Ctrl+B，再按 D 可退出但不中断任务
bash tmux_job.sh attach lsknet_t_train

# 明确终止任务
bash tmux_job.sh stop lsknet_t_train

# 不进入会话，直接跟踪日志
tail -f logs/lsknet_t_obb_full_fold0_img1280.log
```

默认资源和稳定性设置：

| 配置 | T | S |
|---|---:|---:|
| epochs | 800 | 800 |
| imgsz | 1280 | 1280 |
| 全局 batch | 16 | 8 |
| GPU | 4、5、6、7 | 4、5、6、7 |
| AdamW lr0 | 0.0006 | 0.0004 |
| weight decay | 0.05 | 0.05 |
| focal gamma | 1.0 | 1.0 |
| validation conf | 0.05 | 0.05 |
| validation max_det | 600 | 600 |

默认 batch 按每卡 T=4、S=2 保守设置，因为 P2/4 高分辨率特征和密集小目标分配都会明显增加显存占用。首次正式训练前仍应先检查后四张卡的 `nvidia-smi`；确认训练稳定且显存有余量后，可通过 `--batch` 在 tmux 命令中逐步上调。

## F1@0.3 与 TensorBoard

每个验证 epoch 会复用当轮预测，按项目现有评分器执行：

```text
同类别 → 置信度降序 → GT 一对一 → polygon IoU >= 0.3
```

随后精确搜索当前 val 的全局最佳置信度，不会重复跑一次模型推理。`best.pt` 和早停 fitness 使用 `F1@0.3`，标准 Ultralytics fitness 仍作为诊断标量保留。

TensorBoard 新增：

```text
metrics/competition_precision(B)
metrics/competition_recall(B)
metrics/F1@0.3(B)
metrics/best_conf@0.3(B)
metrics/standard_fitness(B)
```

启动方式与 YOLO26 完全相同：

```bash
bash tmux_job.sh start lsknet_tensorboard logs/lsknet_tensorboard.log \
  bash run.sh tensorboard \
  --logdir runs/lsknet_t_obb_full_fold0_img1280 \
  --port 6007
```

通过 `bash tmux_job.sh attach lsknet_tensorboard` 查看服务终端。浏览器使用 SSH 端口转发访问 `http://localhost:6007`。

## 输出

训练目录保持现有组织形式：

```text
runs/<name>/
├── args.yaml
├── events.out.tfevents.*
├── results.csv
├── results.png
├── val_class_metrics.csv
├── val_metrics.json
└── weights/
    ├── best.pt
    ├── last.pt
    └── epochN.pt
```

全局 `results/experiments.csv` 新增：

```text
competition_precision
competition_recall
competition_f1_03
competition_conf
```

旧结果表会在首次追加新结果时自动迁移表头，原有行不会删除。

## 最终比赛评估

训练期 F1 用于观察曲线和选择 checkpoint；正式 val/test 结果仍通过统一的低阈值预测缓存生成：

```bash
bash tmux_job.sh start lsknet_t_val_eval logs/lsknet_t_val_eval.log \
  bash run.sh competition \
  --weights runs/lsknet_t_obb_full_fold0_img1280/weights/best.pt \
  --fold 0 --split val --device 7 --batch 8 --chunk-size 8 \
  --imgsz 1280 --min-conf 0.05 --nms-iou 0.7 --max-det 600 \
  --cache runs/competition/lsknet_t_fold0_img1280.json \
  --output runs/competition/lsknet_t_fold0_img1280_metrics.json
```

再把 val 输出中的最佳 `confidence` 固定到 test：

```bash
bash tmux_job.sh start lsknet_t_test_eval logs/lsknet_t_test_eval.log \
  bash run.sh competition \
  --weights runs/lsknet_t_obb_full_fold0_img1280/weights/best.pt \
  --fold 0 --split test --device 7 --batch 8 --chunk-size 8 \
  --imgsz 1280 --min-conf 0.05 --nms-iou 0.7 --max-det 600 \
  --fixed-conf <VAL_BEST_CONF> \
  --cache runs/competition/lsknet_t_test_img1280.json \
  --output runs/competition/lsknet_t_test_img1280_fixedconf_metrics.json
```

LSKNet 模型使用常规 OBB NMS，`nms_iou` 对它有效，这一点与当前 NMS-free YOLO26 不同。

同一块评估 GPU 上不要同时启动 val 和 test。先等待 `lsknet_t_val_eval` 结束，从输出 JSON 读取 `best.confidence`，替换 `<VAL_BEST_CONF>` 后再启动 test 会话。

## 无标注推理

1300 轮 LSKNet-T 的默认推理入口为 `predict_lsknet.py`。默认参数使用当前最优权重：

- 权重：`runs/lsknet_t_obb_full_fold0_img1280/weights/best.pt`
- `imgsz=1280`
- `conf=0.30566`
- `iou=0.7`
- `max_det=600`

tmux 启动示例：

```bash
bash tmux_job.sh start lsknet_t_predict logs/lsknet_t_predict.log \
  bash run.sh predict-lsk-t \
  --source /path/to/images \
  --device 7 \
  --name lsknet_t_1300_predict
```

输出默认保存到 `runs/predict/lsknet_t_1300_predict/`，包括可视化结果和带置信度的 YOLO OBB txt。
