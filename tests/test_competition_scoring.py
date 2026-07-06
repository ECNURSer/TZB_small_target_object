from pathlib import Path

import pytest

from competition_scoring import (
    MatchRecord,
    ObjectAnnotation,
    best_class_confidences,
    best_confidence,
    load_yolo_ground_truth,
    match_image,
    polygon_iou,
    score_records,
)


def box(x1: float, y1: float, x2: float, y2: float, class_id: int = 0, confidence: float = 1.0):
    return ObjectAnnotation(class_id, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)), confidence)


def test_polygon_iou_exact_geometry():
    assert polygon_iou(box(0, 0, 10, 10).polygon, box(0, 0, 10, 10).polygon) == pytest.approx(1.0)
    assert polygon_iou(box(0, 0, 10, 10).polygon, box(10, 0, 20, 10).polygon) == pytest.approx(0.0)
    assert polygon_iou(box(0, 0, 10, 10).polygon, box(5, 0, 15, 10).polygon) == pytest.approx(1 / 3)


def test_iou_threshold_is_inclusive():
    target = box(0, 0, 10, 10)
    prediction = box(5, 0, 15, 10, confidence=0.9)
    iou = polygon_iou(prediction.polygon, target.polygon)
    records, _ = match_image([prediction], [target], iou_threshold=iou)
    assert records[0].true_positive
    records, _ = match_image([prediction], [target], iou_threshold=iou + 1e-5)
    assert not records[0].true_positive


def test_duplicate_and_wrong_class_predictions_are_false_positives():
    targets = [box(0, 0, 10, 10, class_id=0), box(20, 0, 30, 10, class_id=1)]
    predictions = [
        box(0, 0, 10, 10, class_id=0, confidence=0.9),
        box(0, 0, 10, 10, class_id=0, confidence=0.8),
        box(20, 0, 30, 10, class_id=0, confidence=0.7),
    ]
    records, counts = match_image(predictions, targets)
    score = score_records(records, sum(counts.values()), confidence=0.0)
    assert (score.tp, score.fp, score.fn) == (1, 2, 1)
    assert score.f1 == pytest.approx(0.4)


def test_best_confidence_uses_complete_equal_confidence_group():
    targets = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    predictions = [
        box(0, 0, 10, 10, confidence=0.9),
        box(0, 0, 10, 10, confidence=0.8),
        box(20, 0, 30, 10, confidence=0.7),
    ]
    records, counts = match_image(predictions, targets)
    best = best_confidence(records, sum(counts.values()))
    assert best.confidence == pytest.approx(0.7)
    assert (best.tp, best.fp, best.fn) == (2, 1, 0)
    assert best.f1 == pytest.approx(0.8)


def test_empty_predictions_and_yolo_label_loading(tmp_path: Path):
    label = tmp_path / "image.txt"
    label.write_text("2 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8")
    targets = load_yolo_ground_truth(label, width=100, height=200)
    assert targets == [ObjectAnnotation(2, ((10.0, 40.0), (30.0, 40.0), (30.0, 80.0), (10.0, 80.0)))]
    assert score_records([], total_gt=1, confidence=0.5).f1 == 0.0
    with pytest.raises(FileNotFoundError):
        load_yolo_ground_truth(tmp_path / "missing.txt", width=100, height=200)


def test_class_specific_confidences_maximize_global_micro_f1():
    records = [
        MatchRecord(0.9, 0, True),
        MatchRecord(0.8, 0, False),
        MatchRecord(0.7, 0, False),
        MatchRecord(0.6, 1, True),
        MatchRecord(0.5, 1, True),
    ]
    thresholds, score = best_class_confidences(records, {0: 1, 1: 2})

    assert thresholds == {0: 0.9, 1: 0.5}
    assert (score.tp, score.fp, score.fn) == (3, 0, 0)
    assert score.f1 == 1.0
