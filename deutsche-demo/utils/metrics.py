"""
Metrics module for PII detection evaluation.

Provides precision, recall, and F-beta score calculation for comparing
detected entities against ground truth annotations.
"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class AnnotatedEntity:
    """Represents a ground truth or detected entity annotation."""
    start: int
    end: int
    entity_type: str
    text: str


def load_ground_truth(path: Path) -> list[AnnotatedEntity]:
    """
    Load ground truth annotations from a JSON file.

    Expected format:
    {
        "annotations": [
            {"start": 123, "end": 145, "entity_type": "PERSON", "text": "Hans Müller"},
            ...
        ]
    }

    Args:
        path: Path to the JSON annotations file.

    Returns:
        List of AnnotatedEntity objects.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        AnnotatedEntity(
            start=ann["start"],
            end=ann["end"],
            entity_type=ann["entity_type"],
            text=ann["text"],
        )
        for ann in data.get("annotations", [])
    ]


def _calculate_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate Intersection over Union for two spans."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    union = (end1 - start1) + (end2 - start2) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_matches(
    detected: list[AnnotatedEntity],
    ground_truth: list[AnnotatedEntity],
    overlap_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Calculate true positives, false positives, and false negatives.

    Match logic: A detection is a true positive if:
    - It has ≥50% IoU overlap with a ground truth entity
    - AND the entity types match

    Each ground truth entity can only be matched once (greedy matching).

    Args:
        detected: List of detected entities.
        ground_truth: List of ground truth entities.
        overlap_threshold: Minimum IoU for a match (default 0.5).

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for det in detected:
        matched = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue

            # Check entity type match
            if det.entity_type != gt.entity_type:
                continue

            # Check overlap
            iou = _calculate_iou(det.start, det.end, gt.start, gt.end)
            if iou >= overlap_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn


def precision_recall_fbeta(
    tp: int, fp: int, fn: int, beta: float = 2.0
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F-beta score.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        beta: Beta value for F-score (default 2.0 for F2, emphasizing recall).

    Returns:
        Tuple of (precision, recall, f_beta).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f_beta = 0.0
    else:
        beta_sq = beta ** 2
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    return precision, recall, f_beta


def get_missed_entities(
    detected: list[AnnotatedEntity],
    ground_truth: list[AnnotatedEntity],
    overlap_threshold: float = 0.5,
) -> list[AnnotatedEntity]:
    """
    Return the ground truth entities that were NOT detected (false negatives).

    Args:
        detected: List of detected entities.
        ground_truth: List of ground truth entities.
        overlap_threshold: Minimum IoU for a match (default 0.5).

    Returns:
        List of AnnotatedEntity objects that were missed.
    """
    matched_gt = set()

    for det in detected:
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue

            # Check entity type match
            if det.entity_type != gt.entity_type:
                continue

            # Check overlap
            iou = _calculate_iou(det.start, det.end, gt.start, gt.end)
            if iou >= overlap_threshold:
                matched_gt.add(i)
                break

    # Return all ground truth entities that were not matched
    return [gt for i, gt in enumerate(ground_truth) if i not in matched_gt]


def evaluate_results(
    detected: list[AnnotatedEntity],
    ground_truth: list[AnnotatedEntity],
    overlap_threshold: float = 0.5,
    beta: float = 2.0,
) -> dict:
    """
    Convenience function to evaluate detection results against ground truth.

    Args:
        detected: List of detected entities.
        ground_truth: List of ground truth entities.
        overlap_threshold: Minimum IoU for a match (default 0.5).
        beta: Beta value for F-score (default 2.0).

    Returns:
        Dictionary with keys: tp, fp, fn, precision, recall, f_beta.
    """
    tp, fp, fn = calculate_matches(detected, ground_truth, overlap_threshold)
    precision, recall, f_beta = precision_recall_fbeta(tp, fp, fn, beta)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
    }
