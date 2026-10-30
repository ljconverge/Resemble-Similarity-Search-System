"""
Deterministic SHA-256 Modulo Bucketing Service
===============================================
Enriches records with SHA-256 digests, derives base-16 hash integers,
and deterministically partitions items into uniform modulo buckets.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

NUM_BUCKETS: int = 4
HASH_TRUNCATE_LEN: int = 12

MOCK_RECORDS: List[Dict[str, Any]] = [
    {"id": "REC-1001", "name": "CV_2014_Simonyan_VGG", "category": "Computer Vision", "role": "Computer Vision", "value": 94.2},
    {"id": "REC-1002", "name": "CV_2015_He_ResNet", "category": "Computer Vision", "role": "Computer Vision", "value": 98.91},
    {"id": "REC-1003", "name": "CV_2020_Dosovitskiy_ViT", "category": "Computer Vision", "role": "Computer Vision", "value": 97.4},
    {"id": "REC-1004", "name": "GRAPH_2016_Kipf_GCN", "category": "Graph Learning", "role": "Graph Learning", "value": 91.7},
    {"id": "REC-1005", "name": "GRAPH_2017_Hamilton_GraphSAGE", "category": "Graph Learning", "role": "Graph Learning", "value": 89.5},
    {"id": "REC-1006", "name": "Doc1_Sample_Corpus", "category": "Benchmark PDF", "role": "Benchmark PDF", "value": 100.0},
    {"id": "REC-1007", "name": "Doc2_Sample_Corpus", "category": "Benchmark PDF", "role": "Benchmark PDF", "value": 100.0},
    {"id": "REC-1008", "name": "Doc3_Sample_Corpus", "category": "Benchmark PDF", "role": "Benchmark PDF", "value": 98.91},
    {"id": "REC-1009", "name": "Doc4_Sample_Corpus", "category": "Benchmark PDF", "role": "Benchmark PDF", "value": 97.4},
    {"id": "REC-1010", "name": "NLP_Attention_Transformer", "category": "Natural Language", "role": "Natural Language", "value": 95.8},
    {"id": "REC-1011", "name": "LSH_MinHash_Indexing", "category": "Search Systems", "role": "Search Systems", "value": 92.3},
    {"id": "REC-1012", "name": "Shingle_Jaccard_Verifier", "category": "Search Systems", "role": "Search Systems", "value": 88.6},
]


def compute_sha256_bucket(
    key: str,
    num_buckets: int = NUM_BUCKETS,
    truncate_len: int = HASH_TRUNCATE_LEN,
) -> Dict[str, Any]:
    """
    Computes a SHA-256 digest for `key`, derives an integer via base-16 conversion,
    and deterministically assigns a bucket index in [0, num_buckets - 1].

    Parameters:
        key: The string key to hash.
        num_buckets: Positive integer modulus (default: 4).
        truncate_len: Number of hex characters for display digest (default: 12).

    Returns:
        Dictionary with full_hash, truncated_hash, hash_int, and bucket_id.
    """
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")
    if isinstance(truncate_len, bool) or not isinstance(truncate_len, int) or truncate_len <= 0:
        raise ValueError("truncate_len must be a positive integer.")

    normalized_key = str(key if key is not None else "").strip()
    full_hash = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()
    hash_int = int(full_hash, 16)
    bucket_id = hash_int % num_buckets

    return {
        "full_hash": full_hash,
        "truncated_hash": full_hash[:truncate_len],
        "hash_int": hash_int,
        "bucket_id": bucket_id,
    }


def build_bucketed_dataset(
    raw_records: Sequence[Dict[str, Any]] = MOCK_RECORDS,
    num_buckets: int = NUM_BUCKETS,
) -> Dict[str, Any]:
    """
    Enriches raw records with SHA-256 metadata, partitions them across
    `num_buckets` deterministic modulo buckets, and computes aggregate KPIs.

    Parameters:
        raw_records: Sequence of record dictionaries (must contain id, name, value).
        num_buckets: Number of modulo buckets (default: 4).

    Returns:
        Dictionary containing enriched records, bucket partitions, bucket summaries,
        and global aggregate metrics.
    """
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    buckets: Dict[int, List[Dict[str, Any]]] = {b: [] for b in range(num_buckets)}
    enriched_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_records, start=1):
        rec_id = str(item.get("id", f"REC-{idx:04d}"))
        rec_name = str(item.get("name", f"Record {idx}"))
        rec_category = str(item.get("category", "General"))
        rec_role = str(item.get("role", rec_category))
        try:
            rec_value = float(item.get("value", 0.0))
        except (ValueError, TypeError):
            rec_value = 0.0

        composite_key = f"{rec_id}:{rec_name}"
        hash_meta = compute_sha256_bucket(composite_key, num_buckets=num_buckets)

        enriched = {
            "id": rec_id,
            "name": rec_name,
            "category": rec_category,
            "role": rec_role,
            "value": round(rec_value, 2),
            "full_hash": hash_meta["full_hash"],
            "truncated_hash": hash_meta["truncated_hash"],
            "bucket_id": hash_meta["bucket_id"],
        }
        enriched_records.append(enriched)
        buckets[hash_meta["bucket_id"]].append(enriched)

    total_records = len(enriched_records)
    total_value = round(sum(r["value"] for r in enriched_records), 2)
    average_value = round(total_value / total_records, 2) if total_records > 0 else 0.0

    bucket_summaries: List[Dict[str, Any]] = []
    for b_id in range(num_buckets):
        b_records = buckets[b_id]
        b_count = len(b_records)
        b_avg = (
            round(sum(r["value"] for r in b_records) / b_count, 2)
            if b_count > 0
            else 0.0
        )
        b_pct = (
            round((b_count / total_records) * 100.0, 2)
            if total_records > 0
            else 0.0
        )
        bucket_summaries.append(
            {
                "bucket_id": b_id,
                "label": f"Bucket {b_id}",
                "count": b_count,
                "percentage": b_pct,
                "average_value": b_avg,
                "records": b_records,
            }
        )

    return {
        "records": enriched_records,
        "buckets": buckets,
        "bucket_summaries": bucket_summaries,
        "total_records": total_records,
        "total_value": total_value,
        "average_value": average_value,
        "num_buckets": num_buckets,
    }
