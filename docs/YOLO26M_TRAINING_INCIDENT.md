# YOLO26m 训练异常复盘

## 结论

`yolo26m_obb_fold0_balanced_focal` 并不是在 400 epoch 后正常过拟合或精度真实归零。训练在 epoch 415 的第 21 个 batch 首次出现四项 loss 同时为 NaN，非有限状态随后进入 EMA。验证默认使用 EMA，因此 epoch 415-500 的验证 loss 为 NaN，Precision、Recall 和 mAP 被记录为 0。

损坏后的验证记录无效。最终指标采用损坏前保存的 `best.pt` 在 fold 0 验证集重新评估的结果：

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.6494 | 0.5829 | 0.6143 | 0.5994 | 0.4463 |

## 关键概念

### EMA

EMA 是 Exponential Moving Average，即模型参数和浮点缓冲区的指数移动平均。每次参数更新后，训练器使用下面的形式更新一份独立模型：

```text
EMA_new = decay * EMA_old + (1 - decay) * model_current
```

EMA 可以平滑单个 batch 带来的参数波动，验证和 `best.pt` 通常使用 EMA 模型，结果一般比直接使用当前训练模型稳定。

EMA 的风险是 NaN/Inf 具有传播性。只要一次更新把 NaN 合入 EMA，后续有限数值无法将它恢复：

```text
decay * NaN + finite = NaN
```

本项目的 EMA 还会平均 BatchNorm 的 `running_mean`、`running_var` 等浮点缓冲区。某个异常 forward 即使被 AMP 的 GradScaler 阻止更新参数，也可能已经污染这些缓冲区。

### Batch

Batch 是一次前向、反向传播共同处理的图像数量。本次 `batch=72` 是 4 卡 DDP 的全局 batch，通常每张卡处理 18 张图像。

降低到 64 或 48 后，每张卡分别约处理 16 或 12 张图像，作用包括：

- 减少特征图、梯度和 OBB 匹配过程的峰值显存。
- 减少一个 step 内同时参与 TaskAlignedAssigner 的目标数量。
- 给极端密集样本和临时张量留下显存余量。
- 降低显存紧张时 CUDA 算子回退或异常的概率。

日志在 epoch 189 已出现 `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`，说明 `batch=72` 的运行余量不大。不过 batch 大小主要解决资源压力，并不直接限制浮点数的取值范围，因此降低 batch 不能保证消除 NaN，也不能修复已经损坏的 EMA。

### AMP

AMP 是 Automatic Mixed Precision，即自动混合精度。启用 `amp: true` 后，部分计算使用 FP16，关键状态和部分运算仍使用 FP32，并通过 GradScaler 降低梯度下溢风险。

AMP 的优点：

- 降低显存占用。
- 利用 A100 Tensor Core 提高训练速度。
- 通常允许使用更大的 batch。

FP16 的数值范围和精度低于 FP32。遇到极端激活值、密集目标、特殊 loss 或不稳定算子时，更容易出现 overflow、Inf 或 NaN。设置 `amp: false` 会让训练主要使用 FP32，通常更稳定，但会增加显存占用并降低速度，因此不应该在没有复现数值异常时默认关闭。

## 事件时间线

| 阶段 | 现象 | 判断 |
|---|---|---|
| epoch 414 | mAP50=0.5991，mAP50-95=0.4453 | 最后一个完整有效 epoch |
| epoch 415 batch 1-20 | loss 正常 | 模型仍可训练 |
| epoch 415 batch 21 | box/cls/dfl/angle loss 同时变为 NaN | 首次数值异常 |
| epoch 415 验证 | val loss=NaN，所有精度指标为 0 | EMA 已损坏 |
| epoch 415-500 | 日志持续报告 `EMA contains NaN/Inf` | 后续验证全部无效，checkpoint 保存被跳过 |
| 训练结束 | `TkAgg` 在无桌面服务器绘图失败 | 与 EMA NaN 是两个独立问题 |
| 独立复评 | `best.pt` 得到 mAP50=0.5994 | 损坏前权重有效，模型没有真实归零 |

## 为什么自动恢复没有生效

当前恢复逻辑只读取当前 epoch 最后一个 batch 的 `self.loss`，并且要求 loss 非有限与 fitness 异常同时成立。epoch 415 中间 batch 出现 NaN 后，后续 batch 的即时 loss 可以恢复有限值，但 epoch 平均 loss 和 EMA 已经被污染，因此没有进入回滚分支。

当前 EMA 更新也没有在合入模型状态前检查全部参数和浮点缓冲区是否有限。一旦 EMA 出现 NaN，保存逻辑只能跳过 checkpoint，不能修复 EMA。

## 解决方案

### 已实施的训练保护

训练器已增加以下保护：

1. 每次 optimizer step 检查裁剪后的梯度范数和 AMP GradScaler 状态。
2. 梯度偶发溢出但模型缓冲区仍正常时，只跳过当前 batch 的 optimizer 和 EMA 更新。
3. 每次 EMA 更新前检查 BatchNorm 等浮点缓冲区，禁止非有限缓冲区进入 EMA。
4. 用整个 epoch 的 `tloss` 检测异常，不再只检查最后一个 batch。
5. 单独检查 EMA 是否有限；EMA 非有限本身即可触发回滚。
6. DDP 下汇总所有 rank 的检查结果，任意 rank 异常时所有 rank 执行相同处理。
7. 模型状态、epoch loss 或 EMA 已污染时，从最近有效 `last.pt` 恢复模型、optimizer、GradScaler 和 EMA，并重跑当前 epoch。
8. 首个 epoch 没有有效 checkpoint 可回滚，或连续恢复超过 3 次时，直接停止并报告错误。

### 下一次训练参数顺序

1. 使用已增加有限性检查和自动回滚的训练器。
2. 将全局 batch 从 72 降到 64；仍出现 NaN 时降到 48。
3. 保持 `amp: true` 做短程稳定性验证。
4. 如果在同一区间再次出现 NaN，再设置 `amp: false` 验证是否为 FP16 数值问题。
5. 若关闭 AMP 后仍异常，暂时将 `focal_gamma=0.0`、`cls_pw=0.0`，使用官方 BCE 基线定位自定义分类 loss 的影响。
6. 将 `save_period` 设为 25 或 50，保留更多可恢复节点。

需要关闭 AMP 时，先在 `configs/yolo26m_obb.yaml` 中设置：

```yaml
batch: 48
amp: false
```

再启动新实验：

```bash
bash run.sh train-m \
  --fold 0 \
  --name yolo26m_obb_fold0_fp32
```

当前 `train.py` 支持从命令行覆盖 batch，但没有提供 `--amp` 参数，因此不能使用 `--amp false` 作为命令行参数。

### 训练中检查

```bash
# 持续查看日志
tail -f /home/dihan/yolo26m_obb_fold0_balanced_focal.log

# 搜索数值异常和显存异常
rg -n "nan|NaN|Inf|OutOfMemory|EMA contains" \
  /home/dihan/yolo26m_obb_fold0_balanced_focal.log

# 查看 GPU 显存和利用率
watch -n 2 nvidia-smi
```

出现以下任意情况时应停止训练并检查，不应只看训练进程是否仍在运行：

- 任一 loss 为 NaN/Inf。
- 验证 loss 为 NaN。
- Precision、Recall、mAP 同时变为 0。
- 连续出现 `EMA contains NaN/Inf`。
- checkpoint 连续被跳过。

## 本次实验如何使用

- 模型选择使用 `runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt`。
- 最终指标使用 `runs/yolo26m_obb_fold0_balanced_focal/final_eval/metrics.json`。
- epoch 415-500 的验证指标不参与模型比较。
- `results.png` 保留异常点用于复盘，不能将曲线末尾的 0 当作最终精度。
- `TkAgg` 绘图问题已通过 `run.sh` 默认设置 `MPLBACKEND=Agg` 解决，它与训练 NaN 无直接关系。
