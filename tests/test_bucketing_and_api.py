"""
Integration tests for services/bucketing.py and app.py API routes.
"""
import pytest
from app import create_app
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
    assert analysis["metrics"]["document_count"] == 10
    assert analysis["metrics"]["recall"] == 1.0


def test_api_analyze_invalid_params(client):
    res = client.post(
        "/api/analyze",
        data={"source": "sample", "threshold": "150", "bands": "20"},
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
