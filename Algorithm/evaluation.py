"""
Algorithm/evaluation.py
=======================
Evaluation metrics for comparing Brute-Force (Ground Truth) vs. LSH.
Calculates Precision, Recall (Accuracy), F1-Score, Speedup, and Trade-off summaries.
Zero external algorithm libraries. Easy to explain in viva.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def evaluate_accuracy(
    retrieved_matches: List[Dict[str, Any]],
    ground_truth_matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes Recall (Accuracy), Precision, F1-Score, and identifies missed pairs.
    
    Viva Explanation:
      - Ground Truth = All true pairs found by exhaustive Brute Force.
      - Retrieved = Pairs returned by LSH above threshold.
      - Recall (Accuracy) = |Retrieved ∩ Ground Truth| / |Ground Truth|
      - Precision = |Retrieved ∩ Ground Truth| / |Retrieved|
      - Speedup = Brute Force Execution Time / LSH Execution Time
    """
    def _to_key(m: Dict[str, Any]) -> Tuple[str, str]:
        if "doc_a" in m and "doc_b" in m:
            return (min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"]))
        return (m.get("doc_id", ""), "")

    gt_keys = {_to_key(m) for m in ground_truth_matches}
    ret_keys = {_to_key(m) for m in retrieved_matches}

    true_positives = gt_keys & ret_keys
    false_negatives = gt_keys - ret_keys  # Missed by LSH
    false_positives = ret_keys - gt_keys

    recall = (len(true_positives) / len(gt_keys) * 100.0) if gt_keys else 100.0
    precision = (len(true_positives) / len(ret_keys) * 100.0) if ret_keys else 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy_pct": round(recall, 1),
        "recall": round(recall, 1),
        "recall_pct": round(recall, 1),
        "precision": round(precision, 1),
        "precision_pct": round(precision, 1),
        "f1_score": round(f1, 2),
        "true_positives_count": len(true_positives),
        "false_negatives_count": len(false_negatives),
        "false_positives_count": len(false_positives),
        "ground_truth_count": len(gt_keys),
        "retrieved_count": len(ret_keys),
        "missed_keys": false_negatives,
    }


def calculate_speedup(bf_time_ms: float, lsh_time_ms: float) -> Tuple[float, str]:
    """Calculates speedup multiplier of LSH over Brute Force."""
    bf = max(bf_time_ms, 0.001)
    lsh = max(lsh_time_ms, 0.0001)
    ratio = round(bf / lsh, 1)
    return ratio, f"{ratio:.1f}x"
