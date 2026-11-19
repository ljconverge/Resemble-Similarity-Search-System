from __future__ import annotations

import re
import math
from typing import Dict, List, Optional
import numpy as np


def clean_text(text: Optional[str]) -> str:
    # Cleans raw text by lowercasing and removing punctuation/special characters.

    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_term_frequencies(
    text: str,
    shingle_k: int = 1,
    shingle_mode: str = "word",
) -> Dict[str, float]:
    # Tokenize text and extracts word frequencies or k-shingle frequencies.

    cleaned = clean_text(text)
    if not cleaned:
        return {}

    tokens = cleaned.split(" ")
    if shingle_mode == "word":
        if shingle_k <= 1 or len(tokens) < shingle_k:
            shingles = tokens
        else:
            shingles = [
                " ".join(tokens[i : i + shingle_k])
                for i in range(len(tokens) - shingle_k + 1)
            ]
    else:  # char shingles
        cleaned_str = "".join(tokens)
        if shingle_k <= 1 or len(cleaned_str) < shingle_k:
            shingles = list(cleaned_str)
        else:
            shingles = [
                cleaned_str[i : i + shingle_k]
                for i in range(len(cleaned_str) - shingle_k + 1)
            ]

    freqs: Dict[str, float] = {}
    for item in shingles:
        if item:
            freqs[item] = freqs.get(item, 0.0) + 1.0
    return freqs


def probability_jaccard(weights_a: Dict[str, float], weights_b: Dict[str, float]) -> float:
    # Compute exact Probability Jaccard Similarity between two weighted word distributions.    
    # Formula: J_P(p, q) = sum_{i in support_intersection} 1 / ( sum_j max(p_j / p_i, q_j / q_i) )
    
    if not weights_a and not weights_b:
        return 1.0
    if not weights_a or not weights_b:
        return 0.0

    tot_a = sum(weights_a.values())
    tot_b = sum(weights_b.values())
    if tot_a <= 0.0 or tot_b <= 0.0:
        return 0.0

    # Normalized distributions so sum(p) == 1.0 and sum(q) == 1.0
    p_dict = {k: v / tot_a for k, v in weights_a.items()}
    q_dict = {k: v / tot_b for k, v in weights_b.items()}

    # Support intersection
    set_a = set(p_dict.keys())
    set_b = set(q_dict.keys())
    common = set_a & set_b
    if not common:
        return 0.0

    # Terms unique to each document
    p_only = set_a - common
    q_only = set_b - common

    # Base sums for terms where one distribution is zero:
    # If p_j == 0 (q_only), ratio p_j / q_j = 0 <= p_i / q_i for all i, contributing q_j / q_i
    q_base = sum(q_dict[k] for k in q_only)
    # If q_j == 0 (p_only), ratio p_j / q_j = inf > p_i / q_i for all i, contributing p_j / p_i
    p_tail = sum(p_dict[k] for k in p_only)

    # Sort common terms in ascending order of ratio r = p / q
    common_items = [(p_dict[k] / q_dict[k], p_dict[k], q_dict[k]) for k in common]
    common_items.sort(key=lambda x: x[0])

    c_p = np.array([x[1] for x in common_items], dtype=np.float64)
    c_q = np.array([x[2] for x in common_items], dtype=np.float64)

    # Prefix sums of q and suffix sums of p over common terms
    prefix_q = q_base + np.cumsum(c_q)
    prefix_c_p = np.cumsum(c_p)
    suffix_p = p_tail + (prefix_c_p[-1] - prefix_c_p)

    # Exact denominators and summation
    denoms = (prefix_q / c_q) + (suffix_p / c_p)
    sim = float(np.sum(1.0 / denoms))

    return max(0.0, min(1.0, sim))
