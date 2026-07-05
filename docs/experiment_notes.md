# 实验记录

截至 2026-07-05：

- 外部数据位于 `/data/work1/00_data/TZB/subject1`，项目通过符号链接读取，不复制原图。
- n/s/m 只使用 fold 0：train 6792 图/334854 标注，val 1703 图/86298 标注。
- 独立 test：944 图/47549 标注。
- `3057.tif` 为 0 字节；`85.tif` 和 `3250.tif` 已截断。三张图在各自划分中按 YOLO11 转换规则跳过，不单独修复。
- 标签完整性检查：断链、非 9 列标签、非法类别和越界归一化坐标均为 0。
- YOLO26n 历史探索实验的 runs、日志和统一结果记录已删除，不参与后续比较。
- YOLO26s 500 epoch 基线复评：Precision=0.6071、Recall=0.5039、F1=0.5507、mAP50=0.5357、mAP50-95=0.3938。
- YOLO26m 500 epoch 有效 `best.pt` 复评：Precision=0.6494、Recall=0.5829、F1=0.6143、mAP50=0.5994、mAP50-95=0.4463。
- m 在 epoch 415 后 EMA 被 NaN 污染，后续训练记录无效；最终结果来自损坏前 `best.pt` 的独立复评。
- 下一轮 YOLO26n 使用 700 epoch、imgsz 1024、batch 64、AdamW、余弦学习率和 balanced focal loss。
- 当前指标均为 fold 0 val，不是独立 test 结果。

训练完成后，以 `results/experiments.csv` 为机器可读主记录，以 `results/EXPERIMENT_RESULTS.md` 为展示表。不得把官方 DOTA 指标或配置预估值填写为本项目实测结果。
