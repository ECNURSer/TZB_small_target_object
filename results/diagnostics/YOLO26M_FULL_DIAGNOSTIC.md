# YOLO26m 全量训练诊断

评估对象：

- 训练目录：`runs/yolo26m_obb_full_fold0_img1280_deg180_1500ep_b96-4`
- 权重：`weights/best.pt`
- 训练参数核心：`imgsz=1280`，`degrees=180`，`batch=96`，`device=0,1,2,3,4,5,6,7`，`amp=true`
- 比赛对齐指标：同类别一对一匹配，polygon IoU 阈值 `0.3`，统计 `F1@0.3`

## 1. 本地比赛评分结果

`best.pt` 是第 1009 轮最优权重。训练在第 1309 轮早停，不是完整跑满 1500 轮。

| split | conf 来源 | conf | P | R | F1@0.3 |
| --- | --- | ---: | ---: | ---: | ---: |
| val | val 上寻优 | 0.3230 | 0.7807 | 0.7656 | 0.7731 |
| test | 固定 val conf | 0.3230 | 0.7733 | 0.7666 | 0.7700 |

结论：本地 val/test 表现接近，没有明显本地过拟合崩塌。线上隐藏集表现差，更可能来自隐藏集分布差异、类别定义差异或弱类别泛化不足。

## 2. 类别瓶颈

val 上最弱类别：

| class | GT | TP | FP | FN | P | R | F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other-vehicle | 1160 | 244 | 189 | 916 | 0.5635 | 0.2103 | 0.3063 |
| Trailer | 260 | 96 | 51 | 164 | 0.6531 | 0.3692 | 0.4717 |
| Tractor | 27 | 11 | 2 | 16 | 0.8462 | 0.4074 | 0.5500 |
| Excavator | 233 | 157 | 52 | 76 | 0.7512 | 0.6738 | 0.7104 |
| Cargo Truck | 2983 | 2085 | 634 | 898 | 0.7668 | 0.6990 | 0.7313 |

test 上最弱类别：

| class | GT | TP | FP | FN | P | R | F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other-vehicle | 665 | 110 | 102 | 555 | 0.5189 | 0.1654 | 0.2509 |
| Tractor | 27 | 12 | 4 | 15 | 0.7500 | 0.4444 | 0.5581 |
| Truck Tractor | 24 | 12 | 5 | 12 | 0.7059 | 0.5000 | 0.5854 |
| Trailer | 141 | 73 | 19 | 68 | 0.7935 | 0.5177 | 0.6266 |

主要问题不是整体小目标检测能力，而是少数类/混淆类召回不足，尤其 `other-vehicle`。

## 3. 像素尺度瓶颈

尺度定义：`sqrt(OBB polygon area in pixels)`。

val：

| scale | GT | TP | FP | FN | P | R | F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lt16 | 81694 | 62683 | 17727 | 19011 | 0.7795 | 0.7673 | 0.7734 |
| 16_32 | 4553 | 3351 | 828 | 1202 | 0.8019 | 0.7360 | 0.7675 |
| 32_64 | 51 | 35 | 4 | 16 | 0.8974 | 0.6863 | 0.7778 |

test：

| scale | GT | TP | FP | FN | P | R | F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lt16 | 45345 | 34812 | 10163 | 10533 | 0.7740 | 0.7677 | 0.7709 |
| 16_32 | 2180 | 1621 | 514 | 559 | 0.7593 | 0.7436 | 0.7513 |
| 32_64 | 24 | 19 | 7 | 5 | 0.7308 | 0.7917 | 0.7600 |

尺度总体不是最大的瓶颈；真正差的是“弱类别 + 小尺度”组合：

- val `other-vehicle lt16`: F1 0.2494，FN 776
- test `other-vehicle lt16`: F1 0.1971，FN 494
- val `Trailer lt16`: F1 0.2927
- test `Trailer lt16`: F1 0.1481
- val/test `Cargo Truck lt16`: F1 约 0.56-0.58

## 4. 训练突变原因

`results.csv` 中第 330-410 轮出现过指标坍塌：

- 第 330-390 轮 mAP50 接近 0
- 同期 `val/cls_loss` 异常升到几百
- `train/box_loss` 和 `val/box_loss` 没有同步爆炸
- 第 400 轮后逐步恢复

训练日志还显示：

- `Non-finite training state detected`：71 次
- `Gradient norm is NaN/Inf`：多次恢复
- `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`：71 次
- 最优 epoch：1009
- 早停 epoch：1309

判断：这不是正常收敛波动。更可能是 `batch=96 + imgsz=1280 + 密集小目标图像` 导致 TaskAlignedAssigner 显存压力过大，叠加 AMP fp16、focal/class-weight 分类分支，训练中间出现数值不稳定。项目里的恢复逻辑把模型从 `last.pt` 拉回来了，所以最终 `best.pt` 没坏，但后期训练有效收益很低。

## 5. 后续优化优先级

1. 先稳定训练
   - 把全量训练 `batch` 从 96 降到 64 或 80。
   - 如果仍出现 NaN/Inf，把 `amp=false` 做一组稳定性对照；此时必须同步降低 batch。
   - 把 `lr0` 从 0.0012 降到 0.0008-0.0010，减少后期震荡。
   - 把 `focal_gamma=1.5` 对照为 `1.0` 或 `0.0`，确认 focal 是否引入分类分支不稳定。

2. 针对比赛 F1@0.3 搜索推理参数
   - 固定 val 作为寻优集，只在 val 上搜索 `conf/nms_iou/max_det/augment`。
   - test 只用 val 选出的参数做一次固定评估，避免测试集泄露。
   - 当前 per-class threshold 只从 val F1 0.7731 提到 0.7738，收益很小，不是主突破口。

3. 优先改善弱类别召回
   - 对 `other-vehicle`、`Trailer`、`Tractor`、`Truck Tractor` 做误检/漏检可视化清单。
   - 检查这些类与 `Cargo Truck`、`Dump Truck`、`Van` 的标注边界是否一致。
   - 做按图像的 rare-class oversampling，而不是只调 class loss。
   - 尝试弱类别 copy-paste/裁剪拼贴，但只在 val F1@0.3 上保留有效方案。

4. 做隐藏集分布诊断
   - 没有隐藏 GT 时，比较隐藏提交预测与 val/test 的类别分布、置信度分布、目标像素尺度分布、每图目标密度。
   - 如果隐藏集 `other-vehicle` 或稀有类别占比更高，当前线上掉分就是合理结果。

5. 模型层面
   - 保留 YOLO26m 作为主模型。
   - 用 2-3 个稳定 seed 或不同训练 epoch 的 best checkpoint 做轻量 ensemble，只用 val F1@0.3 判断是否值得。
   - 不建议优先上更大图或 SAHI；已有结果显示尺寸/切片不一定收益，且容易引入更多 FP。

## 6. 产物

- `results/diagnostics/yolo26m_full_b96-4_val_metrics.json`
- `results/diagnostics/yolo26m_full_b96-4_val_diagnostic/by_class.csv`
- `results/diagnostics/yolo26m_full_b96-4_val_diagnostic/by_scale.csv`
- `results/diagnostics/yolo26m_full_b96-4_val_diagnostic/by_class_scale.csv`
- `results/diagnostics/yolo26m_full_b96-4_test_fixedconf_metrics.json`
- `results/diagnostics/yolo26m_full_b96-4_test_diagnostic/by_class.csv`
- `results/diagnostics/yolo26m_full_b96-4_test_diagnostic/by_scale.csv`
- `results/diagnostics/yolo26m_full_b96-4_test_diagnostic/by_class_scale.csv`
- `tools/diagnose_competition_cache.py`
