from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from Algorithm.preprocessing import clean_text, extract_term_frequencies, probability_jaccard

# Prime modulus for Carter-Wegman universal hashing: next prime strictly > 2^32
PRIME_32: int = 4294967311
_STR_HASH_CACHE: Dict[Tuple[str, int], int] = {}


def str_hash_32(token: str, seed: int = 42) -> int:
    key = (token, seed)
    if key in _STR_HASH_CACHE:
        return _STR_HASH_CACHE[key]

    h = (2166136261 ^ (seed * 0x9E3779B9)) & 0xFFFFFFFF
    for b in token.encode("utf-8", errors="ignore"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF

    if len(_STR_HASH_CACHE) < 100000:
        _STR_HASH_CACHE[key] = h
    return h


class ProbMinHasher:
    # ProbMinHash signature generator implementing the Weighted Exponential Race principle.
    # Transform word distributions into compact integer signature vectors of length `m`.
    # Theoretical Guarantee (Ertl, 2022): P(sig_A[k] == sig_B[k]) = Probability_Jaccard(A, B).

    def __init__(self, num_hashes: int = 100, seed: int = 42) -> None:
        if num_hashes <= 0:
            raise ValueError("num_hashes must be a positive integer.")
        self.num_hashes = num_hashes
        self.seed = seed

        # Generate universal hash parameters: h_k(x) = (a_k * x + b_k) % PRIME_32
        rng = np.random.RandomState(seed)
        self.a_coeffs = rng.randint(1, PRIME_32 - 1, size=self.num_hashes, dtype=np.uint64)
        self.b_coeffs = rng.randint(0, PRIME_32 - 1, size=self.num_hashes, dtype=np.uint64)

    def compute_signature(self, weights_dict: Dict[str, float]) -> np.ndarray:
        # Compute m-dimensional integer MinHash signature via weighted exponential race times.
        if not weights_dict:
            return np.zeros(self.num_hashes, dtype=np.uint32)

        items = list(weights_dict.keys())
        total = sum(weights_dict.values())
        if total <= 0.0:
            return np.zeros(self.num_hashes, dtype=np.uint32)

        # 1. Word probabilities p_x
        p = np.array([weights_dict[it] / total for it in items], dtype=np.float64)

        # 2. 32-bit integer hashes for words
        item_hashes = np.array([str_hash_32(it, seed=self.seed) for it in items], dtype=np.uint64)

        # 3. Universal hashing into uniform random variable u in (0, 1)
        a = self.a_coeffs[:, None]
        b = self.b_coeffs[:, None]
        x = item_hashes[None, :]
        hashed = (a * x + b) % np.uint64(PRIME_32)
        u = (hashed.astype(np.float64) + 1.0) / float(PRIME_32 + 1)

        # 4. Exponential arrival times: lambda = -ln(u) / p; winning items form signature
        exp_times = -np.log(u) / p[None, :]
        min_indices = np.argmin(exp_times, axis=1)
        return item_hashes[min_indices].astype(np.uint32)


def estimate_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    # Estimate Probability Jaccard similarity as the fraction of matching hash slots.
    if len(sig_a) == 0 or len(sig_b) == 0 or len(sig_a) != len(sig_b):
        return 0.0
    return float(np.mean(sig_a == sig_b))


preprocess_text = clean_text


def generalized_jaccard(weights_a: Dict[str, float], weights_b: Dict[str, float]) -> float:
    """Generalized Jaccard: sum(min(A,B)) / sum(max(A,B))."""
    all_k = set(weights_a.keys()) | set(weights_b.keys())
    inter = sum(min(weights_a.get(k, 0.0), weights_b.get(k, 0.0)) for k in all_k)
    union = sum(max(weights_a.get(k, 0.0), weights_b.get(k, 0.0)) for k in all_k)
    return float(inter / union) if union > 0 else 0.0
