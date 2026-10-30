"""
Algorithm/lsh_engine.py
========================
High-Performance Locality-Sensitive Hashing (LSH) & MinHash Similarity Engine
with Dual Comparison: Standard LSH (MinHash + Banding) vs. Custom LSH (SHA-256 Modulo 4)
evaluated against an Exact Brute-Force (N²) Baseline.

Includes:
- Unicode NFKC normalization and punctuation-stripped word/char shingling.
- Carter-Wegman universal hashing with Mersenne prime p = 2^61 - 1.
- Chunked NumPy vectorized MinHash signature generation.
- Banded LSH bucketing, candidate generation, and collision tracking (Standard LSH).
- Deterministic SHA-256 modulo partitioning across 4 buckets (Custom LSH).
- Exact Jaccard verification and brute-force evaluation baseline.
- Side-by-side performance, recall, precision, and pruning benchmarks for both approaches.
- Interactive query-against-corpus LSHEngine with real Precision@K and Recall@K.
- Robust multi-backend document loading.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import statistics
import struct
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    import mmh3  # type: ignore
except ImportError:
    mmh3 = None

# Mersenne prime M61 = 2^61 - 1 (2,305,843,009,213,693,951)
# Strictly exceeds uint32 shingle IDs (2^32 - 1) to eliminate aliasing collisions.
MERSENNE_PRIME: int = (1 << 61) - 1
MAX_HASH: int = (1 << 32) - 1
NUM_HASHES_DEFAULT: int = 100
VALID_BANDS: Tuple[int, ...] = (10, 20, 25, 50, 100)
MINHASH_METHODS: Tuple[str, ...] = ("fast", "notebook")
_FAST_CHUNK: int = 512

ShingleMode = Literal["word", "char"]
MinHashMethod = Literal["fast", "notebook"]


# ---------------------------------------------------------------------------
# Text Preprocessing & Shingling
# ---------------------------------------------------------------------------

def preprocess_text(text: Optional[str]) -> str:
    """
    Normalizes ligatures (NFKC), joins hyphenated line breaks, removes
    punctuation and symbols, and collapses whitespace into a lowercase string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("Expected string or None for preprocess_text.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"-\s*\r?\n\s*", "", normalized)
    normalized = normalized.lower()
    cleaned_chars = [
        ch
        for ch in normalized
        if not unicodedata.category(ch).startswith("P")
        and not unicodedata.category(ch).startswith("S")
    ]
    cleaned = "".join(cleaned_chars)
    return re.sub(r"\s+", " ", cleaned).strip()


def make_shingles(
    text: str,
    mode: str = "word",
    k: int = 3,
    already_cleaned: bool = False,
) -> Set[str]:
    """
    Creates word (k=1..20) or character (k=1..30) shingles from text.
    Short texts with fewer than k tokens return a single shingle with all tokens.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("Shingle size k must be an integer.")
    if mode == "word":
        if not (1 <= k <= 20):
            raise ValueError("For word shingles, k must be between 1 and 20.")
    elif mode == "char":
        if not (1 <= k <= 30):
            raise ValueError("For char shingles, k must be between 1 and 30.")
    else:
        raise ValueError(f"Invalid shingle mode '{mode}'. Choose 'word' or 'char'.")

    cleaned = text if already_cleaned else preprocess_text(text)
    if not cleaned:
        return set()

    if mode == "word":
        words = cleaned.split(" ")
        if not words or words == [""]:
            return set()
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    # mode == "char"
    if len(cleaned) < k:
        return {cleaned}
    return {cleaned[i : i + k] for i in range(len(cleaned) - k + 1)}


def _hash32(value: Union[str, bytes], seed: int = 42) -> int:
    """Computes an unsigned 32-bit integer hash via mmh3 with a deterministic SHA-256 fallback."""
    if mmh3 is not None:
        return int(mmh3.hash(value, seed=seed, signed=False))
    raw = value.encode("utf-8") if isinstance(value, str) else value
    seed_prefix = struct.pack("<I", seed & 0xFFFFFFFF)
    digest = hashlib.sha256(seed_prefix + raw).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def shingle_ids(shingles: Iterable[str], seed: int = 42) -> np.ndarray:
    """Hashes string shingles into a 1D uint32 NumPy array."""
    items = list(shingles)
    if not items:
        return np.empty(0, dtype=np.uint32)
    return np.fromiter(
        (_hash32(s, seed=seed) for s in items),
        dtype=np.uint32,
        count=len(items),
    )


# ---------------------------------------------------------------------------
# Similarity & Carter-Wegman MinHash
# ---------------------------------------------------------------------------

def jaccard(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Exact Jaccard similarity |A ∩ B| / |A ∪ B|. Two empty sets return 1.0."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    return inter / union if union else 1.0


jaccard_similarity = jaccard


def make_hash_params(
    num_hashes: int = NUM_HASHES_DEFAULT, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates deterministic (a, b) universal hash coefficients for Carter-Wegman hashing:
        h(x) = (a * x + b) mod p, with p = 2^61 - 1.
    """
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes <= 0:
        raise ValueError("num_hashes must be a positive integer.")
    rng = random.Random(seed)
    a = np.array(
        [rng.randint(1, MERSENNE_PRIME - 1) for _ in range(num_hashes)],
        dtype=object,
    )
    b = np.array(
        [rng.randint(0, MERSENNE_PRIME - 1) for _ in range(num_hashes)],
        dtype=object,
    )
    return a, b


def minhash_signatures(
    doc_shingle_ids: Sequence[np.ndarray],
    num_hashes: int = NUM_HASHES_DEFAULT,
    method: str = "fast",
    seed: int = 42,
    hash_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[List[int]]:
    """
    Computes MinHash signatures for a sequence of shingle-ID arrays.

    Parameters:
        doc_shingle_ids: Sequence of 1D uint32 arrays.
        num_hashes: Number of MinHash values in signature (default: 100).
        method: 'fast' (Carter-Wegman universal hashing) or 'notebook' (seeded mmh3).
        seed: Random seed for hash coefficient generation.
        hash_params: Optional cached (a, b) tuple from make_hash_params.

    Returns:
        List of integer signatures of length num_hashes for each document.
    """
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes <= 0:
        raise ValueError("num_hashes must be a positive integer.")
    if method not in MINHASH_METHODS:
        raise ValueError(f"Unknown MinHash method '{method}'. Choose from {MINHASH_METHODS}.")
    if not doc_shingle_ids:
        return []

    for arr in doc_shingle_ids:
        if len(arr) == 0:
            raise ValueError("Cannot compute MinHash signature for an empty shingle array.")

    signatures: List[List[int]] = []

    if method == "notebook":
        for arr in doc_shingle_ids:
            u32_arr = np.asarray(arr, dtype=np.uint32)
            byte_items = [x.tobytes() for x in u32_arr]
            sig: List[int] = []
            for i in range(num_hashes):
                min_val = min(_hash32(b_item, seed=i) for b_item in byte_items)
                sig.append(int(min_val))
            signatures.append(sig)
        return signatures

    # method == "fast"
    if hash_params is not None:
        a, b = hash_params
    else:
        a, b = make_hash_params(num_hashes, seed)

    a_col = a[:, None]
    b_col = b[:, None]
    chunk_size = max(int(_FAST_CHUNK), 1)

    for arr in doc_shingle_ids:
        x_all = np.asarray(arr, dtype=object)
        current_min: Optional[np.ndarray] = None
        for start in range(0, len(x_all), chunk_size):
            chunk = x_all[start : start + chunk_size][None, :]
            hashed = (a_col * chunk + b_col) % MERSENNE_PRIME
            chunk_min = hashed.min(axis=1)
            if current_min is None:
                current_min = chunk_min
            else:
                current_min = np.minimum(current_min, chunk_min)
        assert current_min is not None
        signatures.append([int(v) for v in current_min])

    return signatures


def estimate_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Estimates Jaccard similarity as the fraction of matching hash slots."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be non-empty and of equal length.")
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


# ---------------------------------------------------------------------------
# Standard LSH Banding & Candidate Generation
# ---------------------------------------------------------------------------

def generate_candidates(
    signatures: Dict[str, Sequence[int]],
    bands: int = 20,
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[int, Tuple[int, ...]], List[str]]]:
    """Partitions signatures into bands and derives candidate pairs from collisions."""
    if isinstance(bands, bool) or not isinstance(bands, int) or bands <= 0:
        raise ValueError("bands must be a positive integer.")
    if not signatures:
        return set(), {}

    first_sig = next(iter(signatures.values()))
    num_hashes = len(first_sig)
    if num_hashes == 0 or num_hashes % bands != 0:
        raise ValueError(
            f"Signature length ({num_hashes}) must be divisible by bands ({bands})."
        )

    rows = num_hashes // bands
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = {}

    for doc_id, sig in signatures.items():
        if len(sig) != num_hashes:
            raise ValueError("All signatures must have the same length.")
        for b_idx in range(bands):
            start = b_idx * rows
            band_key = (b_idx, tuple(int(x) for x in sig[start : start + rows]))
            buckets.setdefault(band_key, []).append(doc_id)

    candidates: Set[Tuple[str, str]] = set()
    for doc_list in buckets.values():
        if len(doc_list) > 1:
            ordered = sorted(doc_list)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    candidates.add((ordered[i], ordered[j]))

    return candidates, buckets


def shared_band_counts(
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]],
) -> Dict[Tuple[str, str], int]:
    """Counts how many bands each candidate document pair collided in."""
    counts: Dict[Tuple[str, str], int] = {}
    for doc_list in buckets.values():
        if len(doc_list) > 1:
            ordered = sorted(doc_list)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pair = (ordered[i], ordered[j])
                    counts[pair] = counts.get(pair, 0) + 1
    return counts


def candidate_probability(s: float, bands: int, rows: int) -> float:
    """S-curve probability 1 - (1 - s^r)^b that a pair with similarity s collides."""
    return float(1.0 - (1.0 - (s ** rows)) ** bands)


def approx_threshold(bands: int, rows: int) -> float:
    """Approximate S-curve inflection threshold (1 / bands) ** (1 / rows)."""
    if bands <= 0 or rows <= 0:
        raise ValueError("bands and rows must be positive.")
    return float((1.0 / bands) ** (1.0 / rows))


# ---------------------------------------------------------------------------
# Custom LSH: Deterministic SHA-256 Modulo Partitioning (Option A)
# ---------------------------------------------------------------------------

def custom_sha256_candidates(
    documents: Dict[str, str],
    num_buckets: int = 4,
) -> Tuple[Set[Tuple[str, str]], Dict[int, List[str]]]:
    """
    Partitions documents into deterministic SHA-256 modulo buckets and derives candidate pairs.
    Documents falling into the exact same bucket index form candidate pairs.

    Parameters:
        documents: Mapping of doc_id to document text (preprocessed or raw).
        num_buckets: Positive integer modulus (default: 4).

    Returns:
        Tuple of (candidates_set, buckets_dictionary mapping bucket_id to list of doc_ids).
    """
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    buckets: Dict[int, List[str]] = {b: [] for b in range(num_buckets)}
    for doc_id, text in sorted(documents.items()):
        raw = text.encode("utf-8")
        hash_hex = hashlib.sha256(raw).hexdigest()
        hash_int = int(hash_hex, 16)
        bucket_id = hash_int % num_buckets
        buckets[bucket_id].append(doc_id)

    candidates: Set[Tuple[str, str]] = set()
    for doc_list in buckets.values():
        if len(doc_list) > 1:
            ordered = sorted(doc_list)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    candidates.add((ordered[i], ordered[j]))

    return candidates, buckets


# ---------------------------------------------------------------------------
# Brute Force & Verification
# ---------------------------------------------------------------------------

def brute_force_all_pairs(
    shingle_sets: Dict[str, Set[Any]],
    threshold: float = 0.0,
) -> Tuple[List[Tuple[str, str, float]], int]:
    """Computes exact Jaccard for all n(n-1)/2 pairs and returns matches >= threshold."""
    names = sorted(shingle_sets.keys())
    n = len(names)
    comparisons = n * (n - 1) // 2
    matches: List[Tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = names[i], names[j]
            score = jaccard(shingle_sets[d1], shingle_sets[d2])
            if score >= threshold:
                matches.append((d1, d2, score))

    matches.sort(key=lambda item: (-item[2], item[0], item[1]))
    return matches, comparisons


def _norm_pair(item: Any) -> Tuple[str, str]:
    if isinstance(item, dict):
        a, b = str(item["doc_a"]), str(item["doc_b"])
    else:
        a, b = str(item[0]), str(item[1])
    return (a, b) if a <= b else (b, a)


def evaluate(found_pairs: Iterable[Any], true_pairs: Iterable[Any]) -> Dict[str, Any]:
    """Computes recall, precision, F1, false negatives, and false positives without zero-division."""
    found_set = {_norm_pair(p) for p in found_pairs}
    true_set = {_norm_pair(p) for p in true_pairs}

    tp = len(found_set & true_set)
    fp = len(found_set - true_set)
    fn = len(true_set - found_set)

    if not true_set and not found_set:
        recall = 1.0
        precision = 1.0
        f1 = 1.0
    elif not true_set:
        recall = 1.0
        precision = 0.0 if found_set else 1.0
        f1 = 0.0
    elif not found_set:
        recall = 0.0
        precision = 0.0
        f1 = 0.0
    else:
        recall = round(tp / len(true_set), 4)
        precision = round(tp / len(found_set), 4)
        denom = len(found_set) + len(true_set)
        f1 = round(2 * tp / denom, 4) if denom > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _pct_text(score: float, threshold: float) -> str:
    """Formats a 0..1 similarity score as a percentage without rounding across the threshold."""
    pct = score * 100.0
    thresh_pct = threshold * 100.0
    decimals = 2
    if score < threshold:
        while decimals < 10 and round(pct, decimals) >= round(thresh_pct, decimals):
            decimals += 2
    return f"{pct:.{decimals}f}"


# ---------------------------------------------------------------------------
# End-to-End Pairwise Collection Analysis (With Dual Comparison)
# ---------------------------------------------------------------------------

def run_analysis(
    documents: Dict[str, str],
    threshold: float = 0.8,
    bands: int = 20,
    shingle_mode: str = "word",
    shingle_size: int = 3,
    num_hashes: int = NUM_HASHES_DEFAULT,
    minhash_method: str = "fast",
    seed: int = 42,
    extraction_ms: float = 0.0,
    source_label: str = "Uploaded documents",
    timing_repeats: int = 1,
    custom_buckets: int = 4,
) -> Dict[str, Any]:
    """
    Runs document preprocessing, Standard LSH (MinHash + Banding), Custom LSH (SHA-256 Modulo 4),
    and exact Brute-Force baseline evaluation with stage timings and empirical metrics.
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not (0.0 <= float(threshold) <= 1.0)
    ):
        raise ValueError("threshold must be a number between 0.0 and 1.0.")
    if isinstance(timing_repeats, bool) or not isinstance(timing_repeats, int) or timing_repeats < 1:
        raise ValueError("timing_repeats must be an integer >= 1.")
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes <= 0:
        raise ValueError("num_hashes must be a positive integer.")
    if (
        isinstance(bands, bool)
        or not isinstance(bands, int)
        or bands <= 0
        or num_hashes % bands != 0
    ):
        raise ValueError(f"bands ({bands}) must evenly divide num_hashes ({num_hashes}).")
    if minhash_method not in MINHASH_METHODS:
        raise ValueError(f"minhash_method must be one of {MINHASH_METHODS}.")

    make_shingles("validation check", shingle_mode, shingle_size)

    if not documents:
        raise ValueError("No documents provided.")

    t_prep0 = time.perf_counter()
    warnings: List[str] = []
    doc_summaries: List[Dict[str, Any]] = []
    shingle_sets: Dict[str, Set[str]] = {}
    cleaned_docs: Dict[str, str] = {}

    for name in sorted(documents.keys()):
        raw_text = documents[name]
        cleaned = preprocess_text(raw_text)
        shingles = make_shingles(cleaned, shingle_mode, shingle_size, already_cleaned=True)
        if not shingles:
            warnings.append(f"Skipped '{name}': no usable text after cleaning.")
            continue
        shingle_sets[name] = shingles
        cleaned_docs[name] = cleaned
        word_count = len(cleaned.split(" ")) if cleaned else 0
        doc_summaries.append(
            {
                "name": name,
                "words": word_count,
                "characters": len(cleaned),
                "shingles": len(shingles),
            }
        )

    if len(shingle_sets) < 2:
        raise ValueError("At least two non-empty documents are required for similarity search.")

    preprocessing_ms = (time.perf_counter() - t_prep0) * 1000.0
    doc_names = [d["name"] for d in doc_summaries]

    # Timing accumulators for repeated benchmarking
    hashing_times: List[float] = []
    minhash_times: List[float] = []
    banding_times: List[float] = []
    verify_times: List[float] = []
    brute_times: List[float] = []
    custom_times: List[float] = []

    candidates: Set[Tuple[str, str]] = set()
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = {}
    verified_pairs: List[Dict[str, Any]] = []
    rejected_pairs: List[Dict[str, Any]] = []
    brute_matches: List[Tuple[str, str, float]] = []
    brute_comparisons: int = 0

    # Custom LSH state
    custom_cands: Set[Tuple[str, str]] = set()
    custom_bucket_map: Dict[int, List[str]] = {}
    custom_verified_pairs: List[Dict[str, Any]] = []
    custom_rejected_pairs: List[Dict[str, Any]] = []

    cached_hash_params = (
        make_hash_params(num_hashes, seed) if minhash_method == "fast" else None
    )

    for _ in range(timing_repeats):
        # 1. Standard LSH Timing
        t0 = time.perf_counter()
        id_arrays = [shingle_ids(shingle_sets[name], seed=42) for name in doc_names]
        t_hash = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        sig_list = minhash_signatures(
            id_arrays,
            num_hashes=num_hashes,
            method=minhash_method,
            seed=seed,
            hash_params=cached_hash_params,
        )
        sig_map = {name: sig for name, sig in zip(doc_names, sig_list)}
        t_mh = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        candidates, buckets = generate_candidates(sig_map, bands=bands)
        band_counts = shared_band_counts(buckets)
        t_band = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        cur_verified: List[Dict[str, Any]] = []
        cur_rejected: List[Dict[str, Any]] = []
        for doc_a, doc_b in sorted(candidates):
            score = jaccard(shingle_sets[doc_a], shingle_sets[doc_b])
            pct_str = _pct_text(score, threshold)
            sim_num = round(score * 100.0, 2)
            shared_b = band_counts.get((doc_a, doc_b), 1)
            row = {
                "doc_a": doc_a,
                "doc_b": doc_b,
                "similarity": sim_num,
                "similarity_raw": score,
                "similarity_text": pct_str,
                "shared_buckets": shared_b,
                "is_match": score >= threshold,
                "status": "Near duplicate" if score >= threshold else "Below threshold",
            }
            if score >= threshold:
                cur_verified.append(row)
            else:
                cur_rejected.append(row)

        cur_verified.sort(key=lambda r: (-r["similarity_raw"], r["doc_a"], r["doc_b"]))
        cur_rejected.sort(key=lambda r: (-r["similarity_raw"], r["doc_a"], r["doc_b"]))
        t_ver = (time.perf_counter() - t3) * 1000.0

        # 2. Brute-Force Baseline Timing
        t_bf0 = time.perf_counter()
        brute_matches, brute_comparisons = brute_force_all_pairs(
            shingle_sets, threshold=threshold
        )
        t_bf = (time.perf_counter() - t_bf0) * 1000.0

        # 3. Custom LSH (SHA-256 Modulo 4) Timing
        t_c0 = time.perf_counter()
        custom_cands, custom_bucket_map = custom_sha256_candidates(
            cleaned_docs, num_buckets=custom_buckets
        )
        cur_custom_verified: List[Dict[str, Any]] = []
        cur_custom_rejected: List[Dict[str, Any]] = []
        for doc_a, doc_b in sorted(custom_cands):
            score = jaccard(shingle_sets[doc_a], shingle_sets[doc_b])
            row = {
                "doc_a": doc_a,
                "doc_b": doc_b,
                "similarity": round(score * 100.0, 2),
                "similarity_raw": score,
                "similarity_text": _pct_text(score, threshold),
                "is_match": score >= threshold,
                "status": "Near duplicate" if score >= threshold else "Below threshold",
            }
            if score >= threshold:
                cur_custom_verified.append(row)
            else:
                cur_custom_rejected.append(row)

        cur_custom_verified.sort(key=lambda r: (-r["similarity_raw"], r["doc_a"], r["doc_b"]))
        cur_custom_rejected.sort(key=lambda r: (-r["similarity_raw"], r["doc_a"], r["doc_b"]))
        t_custom = (time.perf_counter() - t_c0) * 1000.0

        # Record run latencies
        hashing_times.append(max(t_hash, 1e-4))
        minhash_times.append(max(t_mh, 1e-4))
        banding_times.append(max(t_band, 1e-4))
        verify_times.append(max(t_ver, 1e-4))
        brute_times.append(max(t_bf, 1e-4))
        custom_times.append(max(t_custom, 1e-4))

        verified_pairs = cur_verified
        rejected_pairs = cur_rejected
        custom_verified_pairs = cur_custom_verified
        custom_rejected_pairs = cur_custom_rejected

    # Median Latencies
    hashing_ms = round(statistics.median(hashing_times), 3) or 0.001
    minhash_ms = round(statistics.median(minhash_times), 3)
    banding_ms = round(statistics.median(banding_times), 3)
    verify_ms = round(statistics.median(verify_times), 3)
    lsh_ms = round(hashing_ms + minhash_ms + banding_ms + verify_ms, 3)
    brute_ms = round(statistics.median(brute_times), 3)
    custom_lsh_ms = round(statistics.median(custom_times), 3) or 0.001

    # Standard LSH Evaluations
    verified_key_set = {(p["doc_a"], p["doc_b"]) for p in verified_pairs}
    bf_pair_dicts: List[Dict[str, Any]] = []
    for d1, d2, score in brute_matches:
        bf_pair_dicts.append(
            {
                "doc_a": d1,
                "doc_b": d2,
                "similarity": round(score * 100.0, 2),
                "similarity_raw": score,
                "similarity_text": _pct_text(score, threshold),
                "found_by_lsh": (d1, d2) in verified_key_set,
            }
        )

    ev = evaluate(
        [(p["doc_a"], p["doc_b"]) for p in verified_pairs],
        [(d1, d2) for d1, d2, _ in brute_matches],
    )

    n_docs = len(doc_names)
    possible_pairs = n_docs * (n_docs - 1) // 2
    cand_count = len(candidates)
    rows_per_band = num_hashes // bands
    collision_buckets = sum(1 for docs in buckets.values() if len(docs) >= 2)
    cand_precision = (
        1.0
        if (cand_count == 0 and len(brute_matches) == 0)
        else (0.0 if cand_count == 0 else round(len(verified_pairs) / cand_count, 4))
    )
    cand_reduction = (
        round((1.0 - cand_count / possible_pairs) * 100.0, 1) if possible_pairs else 0.0
    )
    speedup = round(brute_ms / max(lsh_ms, 1e-6), 2)

    # Custom LSH Evaluations
    custom_ev = evaluate(
        [(p["doc_a"], p["doc_b"]) for p in custom_verified_pairs],
        [(d1, d2) for d1, d2, _ in brute_matches],
    )
    custom_cand_count = len(custom_cands)
    custom_cand_reduction = (
        round((1.0 - custom_cand_count / possible_pairs) * 100.0, 1) if possible_pairs else 0.0
    )
    custom_cand_precision = (
        1.0
        if (custom_cand_count == 0 and len(brute_matches) == 0)
        else (0.0 if custom_cand_count == 0 else round(len(custom_verified_pairs) / custom_cand_count, 4))
    )
    custom_speedup = round(brute_ms / max(custom_lsh_ms, 1e-6), 2)

    # Closest pairs fallback (ensures Jaccard similarities never disappear when 0 pairs meet threshold)
    closest_pairs: List[Dict[str, Any]] = []
    max_similarity = 0.0
    if brute_matches:
        max_similarity = brute_matches[0][2]
    else:
        all_raw_pairs, _ = brute_force_all_pairs(shingle_sets, threshold=0.0)
        if all_raw_pairs:
            max_similarity = all_raw_pairs[0][2]
            for d1, d2, s in all_raw_pairs[:5]:
                closest_pairs.append(
                    {
                        "doc_a": d1,
                        "doc_b": d2,
                        "similarity": round(s * 100.0, 2),
                        "similarity_raw": s,
                        "similarity_text": _pct_text(s, threshold),
                        "shared_buckets": band_counts.get((d1, d2), 0),
                        "is_match": False,
                        "status": "Below threshold",
                    }
                )

    thresh_display: Union[int, float] = round(threshold * 100.0, 2)
    if thresh_display == int(thresh_display):
        thresh_display = int(thresh_display)

    # Consolidated Metrics Dictionary
    metrics: Dict[str, Any] = {
        "document_count": n_docs,
        "num_documents": n_docs,
        "num_hashes": num_hashes,
        "bands": bands,
        "num_bands": bands,
        "rows_per_band": rows_per_band,
        "shingle_mode": shingle_mode,
        "shingle_size": shingle_size,
        "shingle_unit": "words" if shingle_mode == "word" else "characters",
        "minhash_method": minhash_method,
        "seed": seed if minhash_method == "fast" else None,
        "threshold": thresh_display,
        "approx_lsh_threshold": round(approx_threshold(bands, rows_per_band) * 100.0, 1),
        "possible_pairs": possible_pairs,
        "candidate_count": cand_count,
        "last_candidate_count": cand_count,
        "near_duplicate_count": len(verified_pairs),
        "found_pairs": len(verified_pairs),
        "true_pairs": len(brute_matches),
        "lsh_comparisons": cand_count,
        "brute_comparisons": brute_comparisons,
        "bucket_count": len(buckets),
        "collision_bucket_count": collision_buckets,
        "candidate_reduction": cand_reduction,
        "candidate_precision": cand_precision,
        "recall": ev["recall"],
        "precision": ev["precision"],
        "f1": ev["f1"],
        "false_negatives": ev["false_negatives"],
        "false_positives": ev["false_positives"],
        "max_similarity": round(max_similarity * 100.0, 2),
        "max_similarity_text": _pct_text(max_similarity, threshold),
        "timing_repeats": timing_repeats,
        "extraction_ms": round(extraction_ms, 2),
        "preprocessing_ms": round(preprocessing_ms, 2),
        "hashing_ms": hashing_ms,
        "minhash_ms": minhash_ms,
        "banding_ms": banding_ms,
        "verify_ms": verify_ms,
        "lsh_ms": lsh_ms,
        "brute_ms": brute_ms,
        "speedup": speedup,
        # Custom LSH Metrics
        "custom_lsh_ms": custom_lsh_ms,
        "custom_comparisons": custom_cand_count,
        "custom_candidate_reduction": custom_cand_reduction,
        "custom_candidate_precision": custom_cand_precision,
        "custom_found_pairs": len(custom_verified_pairs),
        "custom_recall": custom_ev["recall"],
        "custom_precision": custom_ev["precision"],
        "custom_f1": custom_ev["f1"],
        "custom_speedup": custom_speedup,
        "custom_buckets": custom_buckets,
    }

    # Side-by-Side 3-Way Algorithmic Comparison Structure
    comparison: Dict[str, Any] = {
        "brute_force": {
            "name": "Brute-Force Baseline (N²)",
            "technique": "All Pairwise Exact Jaccard",
            "execution_ms": brute_ms,
            "comparisons": brute_comparisons,
            "pruned_pct": 0.0,
            "pairs_found": len(brute_matches),
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "speedup": 1.0,
        },
        "standard_lsh": {
            "name": "Standard LSH (MinHash Banding)",
            "technique": f"Carter-Wegman MinHash ({bands} bands × {rows_per_band} rows)",
            "execution_ms": lsh_ms,
            "comparisons": cand_count,
            "pruned_pct": cand_reduction,
            "pairs_found": len(verified_pairs),
            "recall": ev["recall"],
            "precision": ev["precision"],
            "f1": ev["f1"],
            "speedup": speedup,
        },
        "custom_lsh": {
            "name": "Custom LSH (SHA-256 Modulo 4)",
            "technique": "Deterministic Modulo Partitioning (int(SHA256, 16) % 4)",
            "execution_ms": custom_lsh_ms,
            "comparisons": custom_cand_count,
            "pruned_pct": custom_cand_reduction,
            "pairs_found": len(custom_verified_pairs),
            "recall": custom_ev["recall"],
            "precision": custom_ev["precision"],
            "f1": custom_ev["f1"],
            "speedup": custom_speedup,
        },
    }

    return {
        "source_label": source_label,
        "documents": doc_summaries,
        "pairs": verified_pairs,
        "closest_pairs": closest_pairs,
        "rejected_pairs": rejected_pairs,
        "brute_force": {"pairs": bf_pair_dicts},
        "warnings": warnings,
        "metrics": metrics,
        "custom_lsh": {
            "pairs": custom_verified_pairs,
            "rejected_pairs": custom_rejected_pairs,
            "candidates_count": custom_cand_count,
            "buckets": custom_bucket_map,
            "metrics": {
                "custom_lsh_ms": custom_lsh_ms,
                "custom_comparisons": custom_cand_count,
                "custom_reduction": custom_cand_reduction,
                "custom_recall": custom_ev["recall"],
                "custom_precision": custom_ev["precision"],
                "custom_f1": custom_ev["f1"],
                "custom_speedup": custom_speedup,
            },
        },
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Document Extraction & Loading
# ---------------------------------------------------------------------------

def extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """Extracts text from .txt or .pdf byte streams using PyMuPDF or UTF-8 decode."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return raw_bytes.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import pymupdf as fitz_mod  # type: ignore
        except ImportError:
            try:
                import fitz as fitz_mod  # type: ignore
            except ImportError:
                fitz_mod = None

        if fitz_mod is not None:
            with fitz_mod.open(stream=raw_bytes, filetype="pdf") as doc:
                return "\n".join(page.get_text("text") for page in doc)

        # Fallback to pypdf if available
        try:
            import io
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(raw_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            pass

    raise ValueError(f"Unsupported file extension: {suffix}")


def load_folder(folder: Union[str, Path]) -> Dict[str, str]:
    """Loads all .pdf and .txt documents from a directory preserving filename extensions."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")

    try:
        import document_loader  # type: ignore
        if hasattr(document_loader, "load_folder"):
            return document_loader.load_folder(folder_path, preserve_extension=True)
        if hasattr(document_loader, "load_documents"):
            res = document_loader.load_documents(folder_path, preserve_extension=True)
            if isinstance(res, tuple):
                return res[0]
            if isinstance(res, dict):
                return res
    except Exception:
        pass

    docs: Dict[str, str] = {}
    for path in sorted(folder_path.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}:
            docs[path.name] = extract_text_from_bytes(path.name, path.read_bytes())
    return docs


# ---------------------------------------------------------------------------
# Interactive Query Search Engine
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    doc_id: str
    score: float
    exact_jaccard: float
    estimated_similarity: float
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "exact_jaccard": round(self.exact_jaccard, 4),
            "estimated_similarity": round(self.estimated_similarity, 4),
            "snippet": self.snippet,
        }


@dataclass
class PerformanceMetrics:
    num_documents: int = 0
    num_hashes: int = NUM_HASHES_DEFAULT
    num_bands: int = 20
    rows_per_band: int = 5
    index_time_ms: float = 0.0
    last_query_lsh_time_ms: float = 0.0
    last_query_linear_time_ms: float = 0.0
    last_candidate_count: int = 0
    speedup_factor: float = 1.0
    precision_at_k: float = 1.0
    recall_at_k: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_documents": self.num_documents,
            "num_hashes": self.num_hashes,
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
            "index_time_ms": round(self.index_time_ms, 3),
            "last_query_lsh_time_ms": round(self.last_query_lsh_time_ms, 3),
            "last_query_linear_time_ms": round(self.last_query_linear_time_ms, 3),
            "last_candidate_count": self.last_candidate_count,
            "speedup_factor": round(self.speedup_factor, 2),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
        }


class LSHEngine:
    """Real-time LSH query engine for indexing and querying text documents."""

    def __init__(
        self,
        num_hashes: int = 100,
        num_bands: int = 20,
        shingle_size: int = 2,
        shingle_mode: str = "word",
        seed: int = 42,
    ) -> None:
        if num_hashes <= 0 or num_bands <= 0 or num_hashes % num_bands != 0:
            raise ValueError("num_hashes must be positive and divisible by num_bands.")
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size
        self.shingle_mode = shingle_mode
        self.seed = seed

        # Pre-cache Carter-Wegman hash parameters for fast repeated querying
        self.hash_params = make_hash_params(self.num_hashes, self.seed)

        self.documents: Dict[str, str] = {}
        self.shingle_sets: Dict[str, Set[str]] = {}
        self.signatures: Dict[str, List[int]] = {}
        self.buckets: List[Dict[Tuple[int, ...], Set[str]]] = [
            {} for _ in range(self.num_bands)
        ]
        self.metrics = PerformanceMetrics(
            num_hashes=self.num_hashes,
            num_bands=self.num_bands,
            rows_per_band=self.rows_per_band,
        )

    def add_document(self, doc_id: str, text: str) -> None:
        """Indexes a single document into the LSH bucket structure."""
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string.")
        t0 = time.perf_counter()
        shingles = make_shingles(text, self.shingle_mode, self.shingle_size)
        ids = shingle_ids(shingles, seed=self.seed)
        sig = (
            minhash_signatures(
                [ids],
                num_hashes=self.num_hashes,
                method="fast",
                seed=self.seed,
                hash_params=self.hash_params,
            )[0]
            if len(ids) > 0
            else [MERSENNE_PRIME] * self.num_hashes
        )
        self.documents[doc_id] = text
        self.shingle_sets[doc_id] = shingles
        self.signatures[doc_id] = sig

        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            key = tuple(sig[start : start + self.rows_per_band])
            self.buckets[b_idx].setdefault(key, set()).add(doc_id)

        self.metrics.num_documents = len(self.documents)
        self.metrics.index_time_ms += (time.perf_counter() - t0) * 1000.0

    def query(
        self, query_text: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Executes an LSH query with strict candidate collision filtering.
        Computes real Precision@K and Recall@K against the ground-truth linear baseline.
        """
        q_shingles = make_shingles(query_text, self.shingle_mode, self.shingle_size)
        if not q_shingles or top_k <= 0 or not self.documents:
            self.metrics.last_candidate_count = 0
            self.metrics.last_query_lsh_time_ms = 0.0
            self.metrics.last_query_linear_time_ms = 0.0
            self.metrics.speedup_factor = 1.0
            self.metrics.precision_at_k = 1.0
            self.metrics.recall_at_k = 1.0
            return []

        t0 = time.perf_counter()
        q_ids = shingle_ids(q_shingles, seed=self.seed)
        q_sig = minhash_signatures(
            [q_ids],
            num_hashes=self.num_hashes,
            method="fast",
            seed=self.seed,
            hash_params=self.hash_params,
        )[0]

        candidates: Set[str] = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            key = tuple(q_sig[start : start + self.rows_per_band])
            candidates.update(self.buckets[b_idx].get(key, set()))

        self.metrics.last_candidate_count = len(candidates)
        pool = candidates if candidates else set(self.documents.keys())

        results: List[SearchResult] = []
        for doc_id in pool:
            exact = jaccard(q_shingles, self.shingle_sets[doc_id])
            est = estimate_jaccard(q_sig, self.signatures[doc_id])
            if exact >= threshold:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        score=exact,
                        exact_jaccard=exact,
                        estimated_similarity=est,
                        snippet=self.documents[doc_id][:180].strip(),
                    )
                )
        results.sort(key=lambda r: (r.score, r.estimated_similarity), reverse=True)
        top_results = results[:top_k]

        lsh_ms = (time.perf_counter() - t0) * 1000.0

        # Ground-truth linear scan across all documents for baseline & accuracy benchmarking
        t1 = time.perf_counter()
        linear_all: List[Tuple[str, float]] = []
        for doc_id, s_set in self.shingle_sets.items():
            exact_val = jaccard(q_shingles, s_set)
            if exact_val >= threshold:
                linear_all.append((doc_id, exact_val))
        linear_all.sort(key=lambda item: item[1], reverse=True)
        linear_top = linear_all[:top_k]
        linear_ms = (time.perf_counter() - t1) * 1000.0

        # Precision@K & Recall@K computation
        lsh_top_ids = {r.doc_id for r in top_results}
        linear_top_ids = {item[0] for item in linear_top}

        if linear_top_ids:
            tp = len(lsh_top_ids & linear_top_ids)
            recall = round(tp / len(linear_top_ids), 4)
            precision = round(tp / len(lsh_top_ids), 4) if lsh_top_ids else 0.0
        else:
            recall = 1.0 if not lsh_top_ids else 0.0
            precision = 1.0 if not lsh_top_ids else 0.0

        self.metrics.last_query_lsh_time_ms = lsh_ms
        self.metrics.last_query_linear_time_ms = linear_ms
        self.metrics.speedup_factor = linear_ms / max(lsh_ms, 1e-6)
        self.metrics.precision_at_k = precision
        self.metrics.recall_at_k = recall

        return top_results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns performance metrics snapshot."""
        return self.metrics.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LSH + brute-force similarity analysis on a folder."
    )
    parser.add_argument("folder", type=str, help="Path to document folder")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Similarity threshold in 0..1"
    )
    parser.add_argument("--bands", type=int, default=20, choices=VALID_BANDS)
    parser.add_argument("--mode", type=str, default="word", choices=("word", "char"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--minhash", type=str, default="fast", choices=MINHASH_METHODS)
    args = parser.parse_args()

    docs = load_folder(args.folder)
    res = run_analysis(
        docs,
        threshold=args.threshold,
        bands=args.bands,
        shingle_mode=args.mode,
        shingle_size=args.k,
        minhash_method=args.minhash,
        source_label=str(args.folder),
    )
    m = res["metrics"]
    print(
        f"Documents: {m['document_count']} | Candidates: {m['candidate_count']} | Matches: {m['found_pairs']}"
    )
    print(
        f"Standard LSH: {m['lsh_ms']} ms (Recall: {m['recall']}, Speedup: {m['speedup']}x)"
    )
    print(
        f"Custom SHA-256: {m['custom_lsh_ms']} ms (Recall: {m['custom_recall']}, Speedup: {m['custom_speedup']}x)"
    )
    print(
        f"Brute Force: {m['brute_ms']} ms"
    )
    for pair in res["pairs"]:
        print(f"  {pair['doc_a']} <-> {pair['doc_b']}: {pair['similarity_text']}%")


if __name__ == "__main__":
    main()
