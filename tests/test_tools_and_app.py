import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_spa_index_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Resemble Similarity Search" in html
    assert "id=\"search-form\"" in html
    assert "id=\"performance-section\"" in html


def test_api_search_json(client):
    res = client.post(
        "/api/search",
        json={"query": "deep residual learning for image recognition", "top_k": 3},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert "metrics" in data
    assert data["count"] > 0
    assert data["results"][0]["doc_id"] == "CV_2015_He_ResNet"


def test_api_search_empty_query(client):
    res = client.post("/api/search", json={"query": "   "})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_performance_json(client):
    res = client.get("/api/performance")
    assert res.status_code == 200
    data = res.get_json()
    assert "num_documents" in data
    assert "num_hashes" in data
    assert "precision_at_k" in data