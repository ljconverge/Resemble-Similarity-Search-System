from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def evaluate_accuracy(
    retrieved_matches: List[Dict[str, Any]],
    ground_truth_matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
       #RECALL: the fraction of brute-force ground-truth matches
       #retrieved by LSH: |Retrieved ∩ Ground Truth| / |Ground Truth|..

    def _to_key(m: Dict[str, Any]) -> Tuple[str, str]:
        if "doc_a" in m and "doc_b" in m:
            return (min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"]))
        return (m.get("doc_id", ""), "")

    gt_keys = {_to_key(m) for m in ground_truth_matches}
    ret_keys = {_to_key(m) for m in retrieved_matches}

    true_positives = gt_keys & ret_keys
    false_negatives = gt_keys - ret_keys  # Missed by LSH
    false_positives = ret_keys - gt_keys

    has_ground_truth = len(gt_keys) > 0
    if has_ground_truth:
        recall = (len(true_positives) / len(gt_keys)) * 100.0
        recall_val = round(recall, 1)
        recall_display = f"{recall_val:.1f}%"
    else:
        # Zero reference matches in ground truth: Recall is N/A
        recall = None
        recall_val = 0.0
        recall_display = "N/A (0 reference matches)"

    precision = (len(true_positives) / len(ret_keys) * 100.0) if ret_keys else 100.0
    r_for_f1 = recall if recall is not None else 0.0
    f1 = (2 * precision * r_for_f1 / (precision + r_for_f1)) if (precision + r_for_f1) > 0 else 0.0

    return {
        "recall": recall,
        "recall_pct": recall_val,
        "recall_display": recall_display,
        "has_ground_truth": has_ground_truth,
        "accuracy_pct": recall_val if has_ground_truth else 100.0,
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
    # Calculates speedup multiplier of LSH over Brute Force.
    bf = max(float(bf_time_ms), 0.0001)
    lsh = max(float(lsh_time_ms), 0.0001)
    ratio = round(bf / lsh, 2)
    return ratio, f"{ratio:.1f}x"
