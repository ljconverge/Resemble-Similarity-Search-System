import unittest
from Algorithm.brute_force import brute_force_all_pairs, brute_force_query
from Algorithm.lsh import ProbMinHashLSH
from Algorithm.evaluation import evaluate_accuracy


class TestLSH(unittest.TestCase):
    def setUp(self):
        self.mini_corpus = {
            "doc_0": "deep learning neural networks backpropagation gradient descent loss",
            "doc_1": "deep learning neural networks backpropagation gradient optimization loss function",
            "doc_2": "astronomy galaxies redshift cosmic microwave background radiation stars telescope",
            "doc_3": "organic chemistry synthesis catalysts chemical reaction aromatic compounds",
        }

    def test_brute_force_all_pairs(self):
        result = brute_force_all_pairs(self.mini_corpus, threshold=0.30)
        self.assertEqual(result["comparisons"], 6)  # 4 * 3 / 2
        self.assertIn("matches", result)
        match_pairs = [(m["doc_a"], m["doc_b"]) for m in result["matches"]]
        self.assertTrue(
            ("doc_0", "doc_1") in match_pairs or ("doc_1", "doc_0") in match_pairs
        )

    def test_lsh_indexing_and_query(self):
        lsh = ProbMinHashLSH(num_hashes=80, num_bands=16)
        idx_time = lsh.index_corpus(self.mini_corpus)
        self.assertGreaterEqual(idx_time, 0)

        query_res = lsh.query("deep learning neural networks optimization", threshold=0.20, top_k=3)
        self.assertLessEqual(query_res["candidates_checked"], len(self.mini_corpus))
        self.assertGreater(len(query_res["results"]), 0)
        top_doc = query_res["results"][0]["doc_id"]
        self.assertIn(top_doc, ["doc_0", "doc_1"])

    def test_lsh_find_all_matches_pruning(self):
        lsh = ProbMinHashLSH(num_hashes=80, num_bands=16)
        lsh.index_corpus(self.mini_corpus)
        lsh_res = lsh.find_all_matches(threshold=0.30)

        self.assertGreaterEqual(lsh_res["pruned_pct"], 0.0)
        self.assertLessEqual(lsh_res["candidate_count"], 6)

    def test_evaluate_accuracy(self):
        ground_truth = [
            {"doc_a": "doc_1", "doc_b": "doc_2", "similarity": 0.8},
            {"doc_a": "doc_3", "doc_b": "doc_4", "similarity": 0.7},
        ]
        predicted = [
            {"doc_a": "doc_1", "doc_b": "doc_2", "similarity": 0.8},
        ]
        metrics = evaluate_accuracy(predicted, ground_truth)
        self.assertEqual(metrics["recall"], 50.0)
        self.assertEqual(metrics["precision"], 100.0)


if __name__ == "__main__":
    unittest.main()
