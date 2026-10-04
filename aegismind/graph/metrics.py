"""Attack-path evaluation metrics (see report section 14)."""
from __future__ import annotations


def _edges(path: list[str]) -> set[tuple[str, str]]:
    return set(zip(path, path[1:]))


def topk_hit(predicted: list[list[str]], truth: list[str], k: int) -> float:
    return 1.0 if any(p == truth for p in predicted[:k]) else 0.0


def reciprocal_rank(predicted: list[list[str]], truth: list[str]) -> float:
    for i, p in enumerate(predicted, start=1):
        if p == truth:
            return 1.0 / i
    return 0.0


def edge_precision_recall(predicted: list[str], truth: list[str]) -> tuple[float, float, float]:
    pe, te = _edges(predicted), _edges(truth)
    if not pe or not te:
        return 0.0, 0.0, 0.0
    tp = len(pe & te)
    precision = tp / len(pe)
    recall = tp / len(te)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1


def next_hop_accuracy(predicted: list[str], truth: list[str], observed_steps: int) -> float:
    """Given the first ``observed_steps`` hops are known, is the next hop right?"""
    i = observed_steps
    if i + 1 >= len(truth) or i + 1 >= len(predicted):
        return 0.0
    return 1.0 if predicted[i + 1] == truth[i + 1] else 0.0
