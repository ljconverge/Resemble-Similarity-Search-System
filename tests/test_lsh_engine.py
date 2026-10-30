"""Tests for Algorithm/lsh_engine.py. Expected values are hand-computed."""
import math
import random

import numpy as np
import pytest

import lsh_engine as E


# ---------------------------------------------------------------- preprocessing
def test_preprocess_matches_notebook_steps():
    assert E.preprocess_text("Machine Learning is Useful!\r\nYes.") == "machine learning is useful yes"
    assert E.preprocess_text(None) == ""


def test_preprocess_pdf_fixes():
    assert E.preprocess_text("ﬁle") == "file"  # ligature
    assert E.preprocess_text("exam-\nple text") == "example text"


# ---------------------------------------------------------------- shingles
def test_word_and_char_shingles():
    assert E.make_shingles("machine learning is useful", "word", 2) == {
        "machine learning", "learning is", "is useful"
    }
    assert E.make_shingles("abcde", "char", 3) == {"abc", "bcd", "cde"}


def test_short_and_empty_text():
    assert E.make_shingles("", "word", 3) == set()
    assert E.make_shingles("two words", "word", 3) == {"two words"}
    assert E.make_shingles("abc", "char", 9) == {"abc"}


@pytest.mark.parametrize("mode,k", [("word", 0), ("word", 21), ("char", 31), ("xyz", 3), ("word", 2.5)])
def test_invalid_shingle_settings(mode, k):
    with pytest.raises(ValueError):
        E.make_shingles("some text here", mode, k)


def test_shingle_ids_are_notebook_mmh3():
    import mmh3
    ids = E.shingle_ids({"hello wor"})
    assert ids.dtype == np.uint32 and int(ids[0]) == mmh3.hash("hello wor", seed=42, signed=False)


# ---------------------------------------------------------------- Jaccard
A = "a rose is a rose is a rose"
B = "a rose is a flower which is a rose"


@pytest.mark.parametrize("k,expected", [(1, 3 / 5), (2, 3 / 6), (3, 3 / 7)])
def test_broder_1997_example(k, expected):
    assert E.jaccard(E.make_shingles(A, "word", k), E.make_shingles(B, "word", k)) == pytest.approx(expected)


def test_readme_example():
    d1 = E.preprocess_text("Machine learning is useful for healthcare.")
    d2 = E.preprocess_text("Machine learning is very useful for healthcare.")
    assert E.jaccard(E.make_shingles(d1, "word", 2), E.make_shingles(d2, "word", 2)) == pytest.approx(4 / 7)


def test_jaccard_properties():
    assert E.jaccard({1, 2}, {1, 2}) == 1.0
    assert E.jaccard({1}, {2}) == 0.0
    assert E.jaccard(set(), set()) == 1.0
    assert E.jaccard(set(), {1}) == 0.0


# ---------------------------------------------------------------- MinHash
def _ids(n, seed):
    rng = random.Random(seed)
    return np.array(rng.sample(range(2 ** 32), n), dtype=np.uint32)


def test_mersenne_prime_strictly_exceeds_uint32():
    # Mathematical correctness guarantee: p must strictly exceed the universe size 2^32
    assert E.MERSENNE_PRIME == (1 << 61) - 1
    assert E.MERSENNE_PRIME > 2 ** 32


@pytest.mark.parametrize("method", E.MINHASH_METHODS)
def test_minhash_reproducible_and_identical_sets(method):
    x = _ids(200, 1)
    s1 = E.minhash_signatures([x, x[::-1].copy()], 20, method)
    s2 = E.minhash_signatures([x], 20, method)
    assert s1[0] == s1[1] == s2[0] and len(s1[0]) == 20


def test_notebook_method_equals_original_notebook_code():
    import mmh3
    docs = [_ids(50, 2), _ids(60, 3)]
    sig = E.minhash_signatures(docs, 10, "notebook")
    for d, ids in enumerate(docs):
        for i in range(10):
            assert sig[d][i] == np.min([mmh3.hash(x.tobytes(), seed=i, signed=False) for x in ids])


def test_fast_method_matches_formula():
    ids = _ids(30, 4)
    a, b = E.make_hash_params(5, 42)
    sig = E.minhash_signatures([ids], 5, "fast", 42)[0]
    for i in range(5):
        assert sig[i] == min((int(a[i]) * int(x) + int(b[i])) % E.MERSENNE_PRIME for x in ids)


def test_fast_method_chunking_gives_same_result(monkeypatch):
    ids = _ids(1000, 5)
    full = E.minhash_signatures([ids], 16, "fast")
    monkeypatch.setattr(E, "_FAST_CHUNK", 7)
    assert E.minhash_signatures([ids], 16, "fast") == full


@pytest.mark.parametrize("method", E.MINHASH_METHODS)
@pytest.mark.parametrize("target", [0.2, 0.5, 0.8])
def test_minhash_estimates_jaccard(method, target):
    common = list(range(int(400 * target)))
    a = set(common) | set(range(10_000, 10_000 + (400 - len(common)) // 2))
    b = set(common) | set(range(20_000, 20_000 + (400 - len(common)) // 2))
    true_j = E.jaccard(a, b)
    to_ids = lambda s: E.shingle_ids({str(v) for v in s})
    m = 200
    sa, sb = E.minhash_signatures([to_ids(a), to_ids(b)], m, method)
    assert abs(E.estimate_jaccard(sa, sb) - true_j) < 4 * math.sqrt(true_j * (1 - true_j) / m) + 0.05


def test_minhash_rejects_bad_input():
    with pytest.raises(ValueError):
        E.minhash_signatures([np.array([], dtype=np.uint32)])
    with pytest.raises(ValueError):
        E.minhash_signatures([_ids(5, 1)], 0)
    with pytest.raises(ValueError):
        E.minhash_signatures([_ids(5, 1)], 10, "magic")


# ---------------------------------------------------------------- LSH
def test_bucketing_and_candidates():
    sigs = {"A": [1, 2, 3, 4], "B": [1, 2, 9, 9], "C": [7, 7, 3, 4]}
    cands, buckets = E.generate_candidates(sigs, 2)
    assert buckets[(0, (1, 2))] == ["A", "B"]
    assert cands == {("A", "B"), ("A", "C")}


def test_band_index_in_key_and_pair_once():
    cands, _ = E.generate_candidates({"A": [5, 5, 1, 1], "B": [2, 2, 5, 5]}, 2)
    assert cands == set()
    cands, buckets = E.generate_candidates({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}, 2)
    assert cands == {("A", "B")} and E.shared_band_counts(buckets)[("A", "B")] == 2


@pytest.mark.parametrize("bands", [3, 0, 5, True])
def test_bad_band_settings(bands):
    with pytest.raises(ValueError):
        E.generate_candidates({"A": [1, 2, 3, 4]}, bands)


def test_s_curve():
    assert E.candidate_probability(0.5, 20, 5) == pytest.approx(0.470, abs=1e-3)
    assert E.approx_threshold(20, 5) == pytest.approx(0.549, abs=1e-3)


# ---------------------------------------------------------------- brute force + evaluation
def test_brute_force_counts_every_pair():
    sets = {f"d{i}": {i, i + 1} for i in range(6)}
    _, comparisons = E.brute_force_all_pairs(sets, 0.0)
    assert comparisons == 15


def test_evaluate():
    ev = E.evaluate([("a", "b", 0.9)], [("b", "a"), ("c", "d")])
    assert ev["recall"] == 0.5 and ev["precision"] == 1.0 and ev["false_negatives"] == 1


# ---------------------------------------------------------------- LSHEngine & Query Search
def test_lsh_engine_query_strict_semantics_and_accuracy():
    engine = E.LSHEngine(num_hashes=100, num_bands=20, shingle_size=2)
    engine.add_document("doc1", "convolutional neural networks for image recognition")
    engine.add_document("doc2", "residual deep learning image recognition networks")
    engine.add_document("doc3", "astronomical planetary telescope observation methods")

    # High similarity query matching doc1 and doc2
    results = engine.query("convolutional neural networks image recognition", top_k=2)
    assert len(results) > 0
    top_ids = [r.doc_id for r in results]
    assert "doc1" in top_ids

    metrics = engine.get_performance_metrics()
    assert metrics["num_documents"] == 3
    assert metrics["last_candidate_count"] > 0
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0

    # Low similarity / disjoint query with 0 candidate collisions under high band rows
    zero_results = engine.query("quantum mechanics wave particle duality entanglement", top_k=2, threshold=0.9)
    assert zero_results == []


# ---------------------------------------------------------------- full run
DOCS = {
    "x1.txt": "locality sensitive hashing groups similar items into the same buckets so only "
              "items sharing a bucket are compared with one another in the search",
    "x2.txt": "locality sensitive hashing groups similar items into the same buckets so only "
              "items sharing a bucket are compared with each other in the search",
    "y.txt": "the mitochondrion is an organelle that generates most of the chemical energy "
             "needed to power the biochemical reactions of a cell",
    "empty.txt": "!!! ???",
}


@pytest.mark.parametrize("mode,k", [("word", 3), ("char", 9)])
@pytest.mark.parametrize("method", E.MINHASH_METHODS)
def test_run_analysis_end_to_end(mode, k, method):
    res = E.run_analysis(DOCS, 0.6, 20, mode, k, 100, method)
    m = res["metrics"]
    assert [(p["doc_a"], p["doc_b"]) for p in res["pairs"]] == [("x1.txt", "x2.txt")]
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["document_count"] == 3 and m["brute_comparisons"] == 3
    assert any("empty.txt" in w for w in res["warnings"])


def test_lsh_pairs_always_subset_of_brute_force():
    res = E.run_analysis(DOCS, 0.0, 100, "word", 1)
    lsh = {frozenset((p["doc_a"], p["doc_b"])) for p in res["pairs"]}
    brute = {frozenset((p["doc_a"], p["doc_b"])) for p in res["brute_force"]["pairs"]}
    assert lsh <= brute


def test_run_analysis_validation():
    with pytest.raises(ValueError):
        E.run_analysis(DOCS, 1.5)
    with pytest.raises(ValueError):
        E.run_analysis(DOCS, 0.8, bands=7)
    with pytest.raises(ValueError):
        E.run_analysis({"e": "  "}, 0.8)


def test_sample_documents_folder():
    folder = E.Path(__file__).resolve().parent.parent / "Document"
    res = E.run_analysis(E.load_folder(folder), 0.8, 20, "word", 3)
    found = {(p["doc_a"], p["doc_b"]) for p in res["pairs"]}
    assert found == {
        ("Doc1.pdf", "Doc2.pdf"),
        ("Doc1.pdf", "Doc3.pdf"),
        ("Doc2.pdf", "Doc3.pdf"),
        ("Doc1.pdf", "Doc4.pdf"),
        ("Doc2.pdf", "Doc4.pdf"),
        ("Doc3.pdf", "Doc4.pdf"),
    }
    s = {(p["doc_a"], p["doc_b"]): p["similarity"] for p in res["pairs"]}
    assert s[("Doc1.pdf", "Doc3.pdf")] == round(907 / 917 * 100, 2)
