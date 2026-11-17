"""
Algorithm/preprocessing.py
==========================
Pure from-scratch text preprocessing and Probability Jaccard similarity.
Zero third-party algorithm libraries used.
"""

import re
import math
from typing import Dict, List, Optional


def clean_text(text: Optional[str]) -> str:
    """Cleans raw text by lowercasing and removing punctuation/special characters.
      Step 1: Convert all characters to lowercase.
      Step 2: Replace punctuation and non-alphanumeric characters with spaces.
      Step 3: Strip leading/trailing whitespaces and collapse multiple spaces.
    """
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
    """Tokenizes text and extracts word frequencies or k-shingle frequencies.
      - If k=1: Counts single word frequencies (bag-of-words).
      - If k>1: Groups consecutive k words (n-grams/shingles) to preserve local phrase order.
    """
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
    """Computes exact Probability Jaccard Similarity between two weighted word distributions.
    Reference: Otmar Ertl (IEEE TKDE 2022).
    
    Formula:
        J_P(p, q) = sum_{i in support_intersection} 1 / ( sum_j max(p_j / p_i, q_j / q_i) )
      1. Normalize term frequencies so each document forms a probability distribution sum(p) = 1.
      2. Find words present in both documents (intersection of supports).
      3. For each overlapping word i, compute denominator: sum of max(p_j/p_i, q_j/q_i) over all words j.
      4. Sum 1 / denominator to get the exact similarity in range [0.0, 1.0] (0% to 100%).
    """
    if not weights_a and not weights_b:
        return 1.0
    if not weights_a or not weights_b:
        return 0.0

    total_a = sum(weights_a.values())
    total_b = sum(weights_b.values())
    if total_a <= 0.0 or total_b <= 0.0:
        return 0.0

    # Step 1: Normalize distributions
    p = {k: v / total_a for k, v in weights_a.items()}
    q = {k: v / total_b for k, v in weights_b.items()}

    # Step 2: All unique words in union
    all_words = set(p.keys()) | set(q.keys())

    # Step 3: Support intersection (words present in both documents)
    common_words = [w for w in all_words if w in p and w in q]
    if not common_words:
        return 0.0

    # Step 4: Compute Otmar Ertl's Probability Jaccard summation
    sim = 0.0
    for i in common_words:
        pi = p[i]
        qi = q[i]
        denom = 0.0
        for j in all_words:
            pj_over_pi = p.get(j, 0.0) / pi
            qj_over_qi = q.get(j, 0.0) / qi
            denom += max(pj_over_pi, qj_over_qi)
        if denom > 0.0:
            sim += 1.0 / denom

    return max(0.0, min(1.0, float(sim)))
