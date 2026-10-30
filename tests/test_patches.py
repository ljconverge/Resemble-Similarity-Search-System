"""Tests for the bug-fix patches (numbers match the bug list)."""
import io
import re

import pytest

import lsh_engine as E

PAIR = {"x1.txt": "the quick brown fox jumps over the lazy dog near the river bank today",
        "x2.txt": "the quick brown fox jumps over the lazy dog near the river bank again",
        "y.txt": "completely different words about graph neural networks and attention"}


# 3 + 4: hashing is LSH time; median of repeats; results identical across repeats
def test_hashing_counted_in_lsh_time_and_repeats():
    m1 = E.run_analysis(PAIR, 0.5, 20, "word", 2, timing_repeats=1)
    m3 = E.run_analysis(PAIR, 0.5, 20, "word", 2, timing_repeats=3)
    for r in (m1, m3):
        m = r["metrics"]
        parts = m["hashing_ms"] + m["minhash_ms"] + m["banding_ms"] + m["verify_ms"]
        assert m["hashing_ms"] > 0 and m["lsh_ms"] == pytest.approx(parts, rel=0.25, abs=0.05)
    assert m3["metrics"]["timing_repeats"] == 3
    assert m1["pairs"] == m3["pairs"] and m1["metrics"]["recall"] == m3["metrics"]["recall"]
    with pytest.raises(ValueError):
        E.run_analysis(PAIR, 0.5, timing_repeats=0)


# 5: nothing to find -> n/a (None), not 100%
def test_metrics_none_when_nothing_to_find():
    m = E.run_analysis(PAIR, 0.99, 20, "word", 3)["metrics"]
    assert m["true_pairs"] == 0 and m["recall"] == 1.0 and m["f1"] == 1.0
    ev = E.evaluate([], [("a", "b")])
    assert ev["recall"] == 0.0 and ev["precision"] == 0.0 and ev["f1"] == 0.0
    ev = E.evaluate([("a", "b")], [("a", "b"), ("c", "d")])
    assert ev["recall"] == 0.5 and ev["precision"] == 1.0 and ev["f1"] == pytest.approx(0.6667)


# 6: candidate precision
def test_candidate_precision():
    m = E.run_analysis(PAIR, 0.5, 100, "word", 1)["metrics"]
    assert m["candidate_precision"] == pytest.approx(m["near_duplicate_count"] / m["candidate_count"], abs=1e-4)
    m = E.run_analysis({"a.txt": "one two three", "b.txt": "four five six"}, 0.5, 20, "word", 1)["metrics"]
    assert m["candidate_count"] == 0 and m["candidate_precision"] == 1.0


# 7: rounding never crosses the threshold
@pytest.mark.parametrize("score,t,text", [(0.79996, 0.8, "79.9960"), (0.8, 0.8, "80.00"),
                                          (0.98909, 0.8, "98.91"), (0.799999999, 0.8, "79.99999990")])
def test_pct_text(score, t, text):
    assert E._pct_text(score, t) == text


def test_rounding_in_results():
    a = " ".join(f"w{i}" for i in range(25000))
    b = " ".join(f"w{i}" for i in range(19999))
    res = E.run_analysis({"A.txt": a, "B.txt": b}, 0.8, 100, "word", 1)
    row = res["rejected_pairs"][0]
    assert row["similarity_text"] == "79.9960" and row["status"] == "Below threshold"


# 8: Unicode punctuation removed like ASCII punctuation
def test_unicode_punctuation():
    assert E.preprocess_text("Don’t “quote” 16–19 a—b • item…") == E.preprocess_text("Don't \"quote\" 16-19 a-b  item...")
    assert E.preprocess_text("“network”") == "network"


# 10: notebook method reports no seed
def test_seed_reported_only_for_fast():
    assert E.run_analysis(PAIR, 0.5, 20, "word", 2, minhash_method="fast")["metrics"]["seed"] == 42
    assert E.run_analysis(PAIR, 0.5, 20, "word", 2, minhash_method="notebook")["metrics"]["seed"] is None


# 9: non-English file names kept
def test_display_names():
    import app
    used = set()
    names = [app._unique_display_name(n, used) for n in
             ["日本.txt", "中国.txt", "रिपोर्ट.pdf", "report.pdf", "report.pdf", "..\\..\\evil<>.pdf", "C:/x/y/z.txt"]]
    assert names == ["日本.txt", "中国.txt", "रिपोर्ट.pdf", "report.pdf", "report_2.pdf", "evil__.pdf", "z.txt"]


# 1, 2, 11: the page contains the label-sync, keep-k and single-submit code
def test_index_page_script():
    import app
    html = app.app.test_client().get("/").get_data(as_text=True)
    assert "function syncLabels()" in html and 'addEventListener("pageshow"' in html
    assert "k < lo || k > hi" in html and "btn.disabled = true" in html


# web app end to end with the patches
def test_app_upload_and_pages():
    import app
    c = app.app.test_client()
    files = [(io.BytesIO(PAIR["x1.txt"].encode()), "x1.txt"), (io.BytesIO(PAIR["x2.txt"].encode()), "x2.txt")]
    c.post("/", data={"source": "upload", "threshold": "99", "bands": "20", "shingle_mode": "word",
                      "shingle_size": "3", "documents": files}, content_type="multipart/form-data")
    perf = c.get("/performance").get_data(as_text=True)
    assert "n/a" in perf and "median of 3 runs" in perf and "shingle hashing" in perf
    assert c.get("/results").status_code == 200
