#!/usr/bin/env python3
"""LSKNet pretrained-weight loading and competition-aligned OBB validation."""

from __future__ import annotations

import fcntl
import os
from collections import defaultdict
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from competition_scoring import MatchRecord, ObjectAnnotation, best_confidence, match_image
from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import LSKNet
from ultralytics.utils import LOGGER, RANK, ops


def _checkpoint_path(source: str, filename: str) -> Path:
    """Resolve a local checkpoint or download a URL once with a process lock."""
    local = Path(source).expanduser()
    if local.is_file():
        return local.resolve()
    if urlparse(source).scheme not in {"http", "https"}:
        raise FileNotFoundError(f"LSKNet 预训练权重不存在: {local}")

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / filename
    lock_path = output.with_suffix(output.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not output.is_file():
            temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
            LOGGER.info(f"Downloading LSKNet pretrained backbone to {output}")
            try:
                torch.hub.download_url_to_file(source, temporary, progress=RANK in {-1, 0})
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return output


def load_lsknet_pretrained(model: torch.nn.Module) -> Path | None:
    """Load an official ImageNet LSKNet checkpoint into layer 0 of an OBB model."""
    backbone = model.model[0]
    if not isinstance(backbone, LSKNet):
        return None

    override = os.getenv("LSKNET_PRETRAINED", "").strip()
    if override.lower() in {"none", "false", "0"}:
        LOGGER.warning("LSKNet backbone pretrained loading disabled; training from random initialization.")
        return None
    source = override or str(model.yaml.get("pretrained_backbone", ""))
    if not source:
        raise ValueError("LSKNet model YAML does not define pretrained_backbone")
    filename = str(model.yaml.get("pretrained_filename") or Path(urlparse(source).path).name)
    checkpoint_path = _checkpoint_path(source, filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid LSKNet checkpoint payload: {checkpoint_path}")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise TypeError(f"LSKNet checkpoint has no state_dict: {checkpoint_path}")

    target = backbone.state_dict()
    compatible = {}
    for key, value in state.items():
        normalized = key.removeprefix("module.").removeprefix("backbone.")
        if normalized in target and target[normalized].shape == value.shape:
            compatible[normalized] = value
    missing, unexpected = backbone.load_state_dict(compatible, strict=False)
    coverage = len(compatible) / max(len(target), 1)
    if coverage < 0.9:
        raise RuntimeError(
            f"LSKNet checkpoint coverage too low ({len(compatible)}/{len(target)}); "
            f"missing examples: {missing[:5]}, unexpected examples: {unexpected[:5]}"
        )
    LOGGER.info(
        f"Loaded LSKNet ImageNet backbone: {checkpoint_path} "
        f"({len(compatible)}/{len(target)} tensors, {coverage:.1%})"
    )
    return checkpoint_path


class CompetitionOBBValidator(yolo.obb.OBBValidator):
    """OBB validator that also computes the project's exact optimized F1@polygon-IoU0.3."""

    competition_iou = 0.3

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Reset standard and competition metric state."""
        super().init_metrics(model)
        self.competition_records: list[MatchRecord] = []
        self.competition_gt: dict[int, int] = defaultdict(int)

    @staticmethod
    def _objects(boxes: torch.Tensor, classes: torch.Tensor, confidences: torch.Tensor | None = None):
        """Convert xywhr tensors to competition scorer objects."""
        polygons = ops.xywhr2xyxyxyxy(boxes).detach().cpu().tolist()
        class_ids = classes.detach().cpu().tolist()
        scores = confidences.detach().cpu().tolist() if confidences is not None else [1.0] * len(polygons)
        return [
            ObjectAnnotation(
                class_id=int(class_id),
                confidence=float(score),
                polygon=tuple((float(x), float(y)) for x, y in polygon),
            )
            for polygon, class_id, score in zip(polygons, class_ids, scores, strict=True)
        ]

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """Collect exact polygon matches, then run the standard probabilistic-IoU metrics."""
        predictions = self._objects(preds["bboxes"], preds["cls"], preds["conf"])
        targets = self._objects(batch["bboxes"], batch["cls"])
        records, gt_counts = match_image(predictions, targets, iou_threshold=self.competition_iou)
        self.competition_records.extend(records)
        for class_id, count in gt_counts.items():
            self.competition_gt[class_id] += count
        return super()._process_batch(preds, batch)

    def gather_stats(self) -> None:
        """Gather both standard and competition records under distributed validation."""
        super().gather_stats()
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(
                (self.competition_records, dict(self.competition_gt)),
                gathered,
                dst=0,
            )
            self.competition_records = []
            self.competition_gt = defaultdict(int)
            for records, gt_counts in gathered:
                self.competition_records.extend(records)
                for class_id, count in gt_counts.items():
                    self.competition_gt[class_id] += count
        elif RANK > 0:
            dist.gather_object((self.competition_records, dict(self.competition_gt)), None, dst=0)
            self.competition_records = []
            self.competition_gt = defaultdict(int)

    def get_stats(self) -> dict:
        """Return standard metrics plus exact F1@0.3 and use that F1 as checkpoint fitness."""
        stats = super().get_stats()
        total_gt = sum(self.competition_gt.values())
        score = best_confidence(self.competition_records, total_gt)
        standard_fitness = float(stats.get("fitness", 0.0))
        stats.update(
            {
                "metrics/competition_precision(B)": score.precision,
                "metrics/competition_recall(B)": score.recall,
                "metrics/F1@0.3(B)": score.f1,
                "metrics/best_conf@0.3(B)": score.confidence,
                "metrics/standard_fitness(B)": standard_fitness,
                "fitness": score.f1,
            }
        )
        self.metrics.competition_score = score
        self.metrics.competition_iou = self.competition_iou
        LOGGER.info(
            "Competition F1@0.3: "
            f"P={score.precision:.5f}, R={score.recall:.5f}, F1={score.f1:.5f}, "
            f"conf={score.confidence:.5f}, TP={score.tp}, FP={score.fp}, FN={score.fn}"
        )
        return stats


class LSKNetOBBTrainer(OBBTrainer):
    """OBB trainer with official LSKNet pretraining and F1@0.3 checkpoint selection."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the model and load the official backbone when starting a new run."""
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if weights is None:
            load_lsknet_pretrained(model)
        return model

    def get_validator(self):
        """Use the competition-aligned validator for every validation epoch."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss"
        return CompetitionOBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
