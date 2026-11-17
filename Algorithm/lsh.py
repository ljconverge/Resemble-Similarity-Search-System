"""
Algorithm/lsh.py
================
Locality-Sensitive Hashing (LSH) using Banding on ProbMinHash Signatures.
Reuses ProbMinHasher and estimate_similarity directly from Algorithm.probminhash.
Zero external algorithm libraries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from Algorithm.preprocessing import extract_term_frequencies, probability_jaccard
from Algorithm.probminhash import ProbMinHasher, estimate_similarity


class ProbMinHashLSH:
    """Locality-Sensitive Hashing (LSH) Index using Banded ProbMinHash Signatures.

      1. Hashes documents into m-dimensional integer signatures using ProbMinHasher.
      2. Partitions the m hash values into b bands of r rows (m = b * r).
      3. Any documents sharing the exact same bucket in >= 1 band are candidate pairs!
      4. Verifies candidate pairs in O(1) time, avoiding exhaustive O(N^2) pairwise checks.
    """

    def __init__(
        self,
        num_hashes: int = 100,
        num_bands: int = 25,
        shingle_k: int = 1,
        shingle_mode: str = "word",
        seed: int = 42,
        hasher: Optional[ProbMinHasher] = None,
    ) -> None:
        if num_hashes <= 0 or num_bands <= 0 or num_hashes % num_bands != 0:
            raise ValueError(f"num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands}).")

        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_k = shingle_k
        self.shingle_mode = shingle_mode
        self.seed = seed

        # Reuse ProbMinHasher directly from probminhash.py
        self.hasher = hasher or ProbMinHasher(num_hashes=self.num_hashes, seed=self.seed)

        # LSH Hash Tables: one dictionary per band
        self.buckets: List[Dict[Tuple[int, ...], Set[str]]] = [{} for _ in range(self.num_bands)]
        # Track colliding buckets (>= 2 docs) for instant candidate pair retrieval
        self.colliding_buckets: Set[Tuple[int, Tuple[int, ...]]] = set()

        self.documents: Dict[str, str] = {}
        self.term_freqs: Dict[str, Dict[str, float]] = {}
        self.signatures: Dict[str, np.ndarray] = {}

    @property
    def approx_threshold(self) -> float:
        """S-curve inflection threshold: tau ≈ (1/b)^(1/r)."""
        return float((1.0 / self.num_bands) ** (1.0 / self.rows_per_band))

    def add_document(
        self,
        doc_id: str,
        text: str,
        freqs: Optional[Dict[str, float]] = None,
        signature: Optional[np.ndarray] = None,
    ) -> None:
        """Extracts word counts, computes signature via ProbMinHasher, and inserts into bands."""
        if freqs is None:
            freqs = extract_term_frequencies(text, shingle_k=self.shingle_k, shingle_mode=self.shingle_mode)
        if signature is None:
            signature = self.hasher.compute_signature(freqs)

        self.documents[doc_id] = text
        self.term_freqs[doc_id] = freqs
        self.signatures[doc_id] = signature

        # Partition signature into b bands of r rows and register into buckets
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            band_key = tuple(signature[start : start + self.rows_per_band].tolist())
            bucket = self.buckets[b_idx].setdefault(band_key, set())
            bucket.add(doc_id)
            if len(bucket) > 1:
                self.colliding_buckets.add((b_idx, band_key))

    def index_corpus(
        self,
        documents: Dict[str, str],
        cached_freqs: Optional[Dict[str, Dict[str, float]]] = None,
        cached_signatures: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Indexes all documents into LSH band hash tables."""
        t0 = time.perf_counter()
        self.buckets = [{} for _ in range(self.num_bands)]
        self.colliding_buckets.clear()
        self.documents.clear()
        self.term_freqs.clear()
        self.signatures.clear()

        for doc_id, text in documents.items():
            f = cached_freqs.get(doc_id) if cached_freqs else None
            sig = cached_signatures.get(doc_id) if cached_signatures else None
            self.add_document(doc_id, text, freqs=f, signature=sig)

        return (time.perf_counter() - t0) * 1000.0

    def index_signatures(self, signatures: Dict[str, np.ndarray]) -> None:
        """Indexes pre-computed signatures into LSH band hash tables."""
        for doc_id, sig in signatures.items():
            self.signatures[doc_id] = sig
            for b_idx in range(self.num_bands):
                start = b_idx * self.rows_per_band
                band_key = tuple(sig[start : start + self.rows_per_band].tolist())
                bucket = self.buckets[b_idx].setdefault(band_key, set())
                bucket.add(doc_id)
                if len(bucket) > 1:
                    self.colliding_buckets.add((b_idx, band_key))

    def get_candidate_pairs(self) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], int]]:
        """Instant O(1) candidate retrieval by checking only colliding buckets."""
        candidates: Set[Tuple[str, str]] = set()
        collision_counts: Dict[Tuple[str, str], int] = {}

        for b_idx, key in self.colliding_buckets:
            doc_list = sorted(self.buckets[b_idx][key])
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    pair = (doc_list[i], doc_list[j])
                    candidates.add(pair)
                    collision_counts[pair] = collision_counts.get(pair, 0) + 1

        return candidates, collision_counts

    def find_all_matches(self, threshold: float = 0.25) -> Dict[str, Any]:
        """Finds near-duplicate document pairs via LSH candidate filtering + verification."""
        t0 = time.perf_counter()
        candidates, collision_counts = self.get_candidate_pairs()
        matches = []
        verified_candidates = []

        for doc_a, doc_b in sorted(candidates):
            # Reuse estimate_similarity directly from probminhash.py
            est_sim = estimate_similarity(self.signatures[doc_a], self.signatures[doc_b]) * 100.0
            sim = probability_jaccard(self.term_freqs[doc_a], self.term_freqs[doc_b])
            record = {
                "doc_a": doc_a,
                "doc_b": doc_b,
                "similarity": round(sim * 100.0, 2),
                "similarity_raw": sim,
                "estimated_similarity": round(est_sim, 2),
                "shared_bands": collision_counts.get((doc_a, doc_b), 1),
                "is_match": sim >= threshold,
            }
            verified_candidates.append(record)
            if sim >= threshold:
                matches.append(record)

        total_ms = (time.perf_counter() - t0) * 1000.0
        matches.sort(key=lambda x: -x["similarity_raw"])
        possible = len(self.documents) * (len(self.documents) - 1) // 2
        pruned_pct = (1.0 - len(candidates) / possible) * 100.0 if possible > 0 else 0.0

        return {
            "matches": matches,
            "candidates": verified_candidates,
            "candidate_count": len(candidates),
            "possible_pairs": possible,
            "pruned_pct": round(pruned_pct, 2),
            "search_ms": round(max(total_ms, 0.001), 3),
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
            "approx_threshold": round(self.approx_threshold * 100.0, 1),
        }

    def query(
        self,
        query_text: str,
        threshold: float = 0.0,
        top_k: int = 5,
        query_sig: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Queries the indexed corpus with 1 document in sub-quadratic time."""
        t0 = time.perf_counter()
        q_freq = extract_term_frequencies(query_text, shingle_k=self.shingle_k, shingle_mode=self.shingle_mode)
        if query_sig is None:
            query_sig = self.hasher.compute_signature(q_freq)

        # Retrieve candidates colliding in >= 1 band
        candidate_ids = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            band_key = tuple(query_sig[start : start + self.rows_per_band].tolist())
            bucket = self.buckets[b_idx].get(band_key)
            if bucket:
                candidate_ids.update(bucket)

        hits = []
        for doc_id in candidate_ids:
            sim = probability_jaccard(q_freq, self.term_freqs[doc_id])
            est_sim = estimate_similarity(query_sig, self.signatures[doc_id]) * 100.0
            if sim >= threshold:
                hits.append({
                    "doc_id": doc_id,
                    "similarity": round(sim * 100.0, 2),
                    "similarity_raw": sim,
                    "estimated_similarity": round(est_sim, 2),
                    "snippet": self.documents[doc_id][:200].strip(),
                })

        hits.sort(key=lambda x: -x["similarity_raw"])
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "results": hits[:top_k],
            "execution_ms": round(max(total_ms, 0.001), 3),
            "candidate_count": len(candidate_ids),
            "candidates_checked": len(candidate_ids),
            "total_documents": len(self.documents),
        }


# Clean alias
LSH = ProbMinHashLSH
