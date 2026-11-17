"""
Algorithm/brute_force.py
========================
Pure Brute-Force similarity evaluation without any hashing or indexing.
Compares every document pair directly using exact Probability Jaccard.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from Algorithm.preprocessing import extract_term_frequencies, probability_jaccard


def brute_force_all_pairs(
    documents: Dict[str, str],
    threshold: float = 0.25,
    shingle_k: int = 1,
    shingle_mode: str = "word",
    cached_freqs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Exhaustively compares all N*(N-1)/2 pairs in O(N^2) time.

      1. Preprocess each document into word frequencies: {word: count}.
      2. Use two nested loops to compare every doc_i with doc_j (where i < j).
      3. Compute exact Probability Jaccard similarity for each pair.
      4. Any pair with similarity >= threshold is marked as a true match.
      5. This serves as our 100% accurate Ground Truth baseline.
    """
    doc_ids = sorted(documents.keys())
    n = len(doc_ids)
    possible_pairs = n * (n - 1) // 2

    # Step 1: Preprocess documents into term frequencies
    if cached_freqs is not None:
        term_freqs = cached_freqs
    else:
        term_freqs = {d_id: extract_term_frequencies(documents[d_id]) for d_id in doc_ids}

    t0 = time.perf_counter()
    matches: List[Dict[str, Any]] = []
    all_pairs: List[Dict[str, Any]] = []
    comparisons = 0

    # Step 2: Nested loops comparing all pairs (O(N^2) exhaustive scan)
    for i in range(n):
        id_a = doc_ids[i]
        freq_a = term_freqs[id_a]
        for j in range(i + 1, n):
            id_b = doc_ids[j]
            freq_b = term_freqs[id_b]
            comparisons += 1

            sim = probability_jaccard(freq_a, freq_b)
            pair_record = {
                "doc_a": id_a,
                "doc_b": id_b,
                "similarity": round(sim * 100.0, 2),
                "similarity_raw": sim,
                "is_match": sim >= threshold,
            }
            all_pairs.append(pair_record)
            if sim >= threshold:
                matches.append(pair_record)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    matches.sort(key=lambda x: -x["similarity_raw"])
    all_pairs.sort(key=lambda x: -x["similarity_raw"])

    return {
        "matches": matches,
        "all_pairs": all_pairs,
        "comparisons": comparisons,
        "possible_pairs": possible_pairs,
        "execution_ms": round(elapsed_ms, 3),
        "term_freqs": term_freqs,
    }


def brute_force_query(
    query_text: str,
    documents: Dict[str, str],
    threshold: float = 0.0,
    top_k: int = 5,
    shingle_k: int = 1,
    shingle_mode: str = "word",
    cached_freqs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Exhaustive linear scan: compares 1 query document against all N documents in O(N) time."""
    t0 = time.perf_counter()
    q_freq = extract_term_frequencies(query_text)

    if cached_freqs is None:
        cached_freqs = {d_id: extract_term_frequencies(text) for d_id, text in documents.items()}

    hits = []
    for doc_id, text in documents.items():
        sim = probability_jaccard(q_freq, cached_freqs[doc_id])
        if sim >= threshold:
            hits.append({
                "doc_id": doc_id,
                "similarity": round(sim * 100.0, 2),
                "similarity_raw": sim,
                "snippet": text[:200].strip(),
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    hits.sort(key=lambda x: -x["similarity_raw"])

    return {
        "results": hits[:top_k],
        "execution_ms": round(elapsed_ms, 3),
        "total_documents": len(documents),
        "comparisons": len(documents),
    }
