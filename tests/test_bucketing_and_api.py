"""
Integration tests for services/bucketing.py and app.py API routes.
Includes validation for Custom LSH (SHA-256 Modulo 4) vs. Standard LSH (MinHash) comparison.
"""
import pytest
from app import create_app
from Algorithm.lsh_engine import custom_sha256_candidates, run_analysis
from services.bucketing import (
    HASH_TRUNCATE_LEN,
    MOCK_RECORDS,
    NUM_BUCKETS,
    build_bucketed_dataset,
    compute_sha256_bucket,
)


# ---------------------------------------------------------------------------
# Bucketing Service Tests
# ---------------------------------------------------------------------------

def test_compute_sha256_bucket_deterministic():
    res1 = compute_sha256_bucket("test:key", num_buckets=4, truncate_len=12)
    res2 = compute_sha256_bucket("test:key", num_buckets=4, truncate_len=12)
    assert res1 == res2
    assert len(res1["truncated_hash"]) == 12
    assert 0 <= res1["bucket_id"] < 4
    assert res1["full_hash"].startswith(res1["truncated_hash"])


def test_compute_sha256_bucket_validation():
    with pytest.raises(ValueError):
        compute_sha256_bucket("key", num_buckets=0)
    with pytest.raises(ValueError):
        compute_sha256_bucket("key", num_buckets=-1)
    with pytest.raises(ValueError):
        compute_sha256_bucket("key", truncate_len=0)


def test_build_bucketed_dataset_aggregates():
    dataset = build_bucketed_dataset(MOCK_RECORDS, num_buckets=NUM_BUCKETS)
    assert dataset["total_records"] == len(MOCK_RECORDS)
    assert len(dataset["bucket_summaries"]) == NUM_BUCKETS

    # Total records match sum of bucket counts
    sum_counts = sum(b["count"] for b in dataset["bucket_summaries"])
    assert sum_counts == dataset["total_records"]

    # Percentages sum to ~100
    sum_pct = sum(b["percentage"] for b in dataset["bucket_summaries"])
    assert pytest.approx(sum_pct, abs=0.1) == 100.0

    # Ensure every record has truncated hash and role
    for rec in dataset["records"]:
        assert len(rec["truncated_hash"]) == HASH_TRUNCATE_LEN
        assert "role" in rec
        assert "category" in rec
        assert 0 <= rec["bucket_id"] < NUM_BUCKETS


# ---------------------------------------------------------------------------
# Custom LSH vs Standard LSH Algorithmic Comparison Tests
# ---------------------------------------------------------------------------

def test_custom_sha256_candidates_deterministic():
    docs = {
        "docA": "deep residual learning for image recognition",
        "docB": "deep residual learning for image recognition",
        "docC": "graph convolutional networks for semi supervised node classification",
    }
    cands, buckets = custom_sha256_candidates(docs, num_buckets=4)
    assert len(buckets) == 4
    # Identical texts must land in the identical bucket and collide as a candidate pair
    assert ("docA", "docB") in cands
    with pytest.raises(ValueError):
        custom_sha256_candidates(docs, num_buckets=0)


def test_run_analysis_contains_3way_comparison():
    docs = {
        "doc1.txt": "locality sensitive hashing groups similar items into buckets efficiently",
        "doc2.txt": "locality sensitive hashing groups similar items into buckets accurately",
        "doc3.txt": "solar system planetary bodies revolve around gravitationally massive stars",
    }
    res = run_analysis(docs, threshold=0.5, bands=20, shingle_mode="word", shingle_size=2)
    assert "comparison" in res
    assert "custom_lsh" in res

    comp = res["comparison"]
    assert "brute_force" in comp
    assert "standard_lsh" in comp
    assert "custom_lsh" in comp

    # Brute Force baseline checks
    assert comp["brute_force"]["recall"] == 1.0
    assert comp["brute_force"]["precision"] == 1.0
    assert comp["brute_force"]["comparisons"] == 3  # 3 * 2 // 2

    # Standard LSH checks
    assert comp["standard_lsh"]["recall"] == 1.0
    assert "execution_ms" in comp["standard_lsh"]

    # Custom LSH checks
    assert "execution_ms" in comp["custom_lsh"]
    assert 0.0 <= comp["custom_lsh"]["recall"] <= 1.0
    assert 0.0 <= comp["custom_lsh"]["precision"] <= 1.0


def test_identical_documents_both_algorithms_100_percent():
    docs = {
        "copy1.txt": "identical text across multiple files for rigorous similarity verification",
        "copy2.txt": "identical text across multiple files for rigorous similarity verification",
    }
    res = run_analysis(docs, threshold=0.8, bands=20)
    m = res["metrics"]
    assert m["recall"] == 1.0
    assert m["custom_recall"] == 1.0
    assert m["found_pairs"] == 1
    assert m["custom_found_pairs"] == 1


def test_zero_matches_no_division_by_zero():
    docs = {
        "a.txt": "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
        "b.txt": "mike november oscar papa quebec romeo sierra tango uniform victor whiskey",
    }
    res = run_analysis(docs, threshold=0.95, bands=20)
    m = res["metrics"]
    # 0 true pairs must yield 1.0 (100%) without division-by-zero or crashing into n/a
    assert m["true_pairs"] == 0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["custom_recall"] == 1.0
    assert m["custom_precision"] == 1.0
    assert len(res["closest_pairs"]) > 0  # Closest pairs fallback populated


# ---------------------------------------------------------------------------
# Flask API Route Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_get(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Resemble" in res.data
    assert b"SHA-256" in res.data


def test_api_buckets(client):
    res = client.get("/api/buckets")
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert "buckets" in data
    assert "bucket_summaries" in data
    assert data["total_records"] > 0


def test_api_performance(client):
    res = client.get("/api/performance")
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert "num_hashes" in data
    assert "num_bands" in data
    assert "speedup_factor" in data


def test_api_search_success(client):
    res = client.post(
        "/api/search",
        json={"query": "deep residual learning image recognition", "top_k": 3},
    )
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert data["query"] == "deep residual learning image recognition"
    assert isinstance(data["results"], list)
    assert "metrics" in data
    assert data["metrics"]["num_documents"] > 0


def test_api_search_empty_query(client):
    res = client.post("/api/search", json={"query": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_analyze_sample(client):
    res = client.post(
        "/api/analyze",
        data={
            "source": "sample",
            "threshold": "80",
            "bands": "20",
            "shingle_mode": "word",
            "shingle_size": "3",
            "minhash_method": "fast",
        },
    )
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert "analysis" in data
    analysis = data["analysis"]
    assert "pairs" in analysis
    assert "metrics" in analysis
    assert "comparison" in analysis
    assert analysis["metrics"]["document_count"] == 10
    assert analysis["metrics"]["recall"] == 1.0
    assert "custom_recall" in analysis["metrics"]
    assert "custom_lsh" in analysis["comparison"]


def test_api_analyze_invalid_params(client):
    res = client.post(
        "/api/analyze",
        data={"source": "sample", "threshold": "150", "bands": "20"},
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
