# Unit tests verifying benchmark validity and evaluation workflow integrity:
#  1. Complete reference-match construction without premature capping.
#  2. Proper exclusion of query document (no self-similarity artifacts).
#  3. Correct handling of zero reference matches.
#  4. Timing authenticity: calculate_speedup produces exact mathematical ratios and dashboard files contain zero timing substitution logic.


import unittest
from pathlib import Path
from Algorithm.preprocessing import extract_term_frequencies
from Algorithm.brute_force import brute_force_query, brute_force_all_pairs
from Algorithm.lsh import ProbMinHashLSH
from Algorithm.evaluation import evaluate_accuracy, calculate_speedup


class TestEvaluationWorkflow(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "Doc_1": "Graph neural networks model structural interactions in biological networks.",
            "Doc_2": "Graph neural networks model relational interactions in biological networks.",
            "Doc_3": "Cosmological inflation predicts primordial gravitational wave perturbations.",
            "Doc_4": "Quantum error correction codes protect qubits from environmental decoherence.",
        }
        self.freqs = {d: extract_term_frequencies(t) for d, t in self.docs.items()}

    def test_complete_reference_match_construction_no_capping(self):
        # Verify queries return all threshold-qualified matches before any display limit.
        q_text = self.docs["Doc_1"]
        # Brute Force query with top_k=None and exclude_doc_id="Doc_1"
        bf_res = brute_force_query(
            q_text,
            self.docs,
            threshold=0.20,
            top_k=None,
            cached_freqs=self.freqs,
            exclude_doc_id="Doc_1",
        )
        self.assertIn("all_matches", bf_res)
        # Doc_1 must NOT be in the results (no self-match)
        doc_ids_returned = [r["doc_id"] for r in bf_res["all_matches"]]
        self.assertNotIn("Doc_1", doc_ids_returned)
        # Doc_2 should qualify at 0.20 threshold
        self.assertIn("Doc_2", doc_ids_returned)

    def test_lsh_query_exclude_doc_id(self):
        """Verify LSH query excludes query document and returns all matches."""
        lsh = ProbMinHashLSH(num_hashes=40, num_bands=10, seed=42)
        lsh.index_corpus(self.docs, cached_freqs=self.freqs)
        q_text = self.docs["Doc_1"]

        lsh_res = lsh.query(
            q_text,
            threshold=0.20,
            top_k=None,
            exclude_doc_id="Doc_1",
        )
        self.assertIn("all_matches", lsh_res)
        doc_ids_returned = [r["doc_id"] for r in lsh_res["all_matches"]]
        self.assertNotIn("Doc_1", doc_ids_returned)

    def test_zero_reference_match_handling(self):
        """When ground truth has 0 matches, Recall must be None / N/A, not 100%."""
        gt_empty = []
        ret_empty = []
        metrics = evaluate_accuracy(ret_empty, gt_empty)

        self.assertFalse(metrics["has_ground_truth"])
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["ground_truth_count"], 0)
        self.assertIn("N/A", metrics["recall_display"])

    def test_speedup_calculation_is_exact(self):
        """calculate_speedup must compute exact mathematical ratio without alterations."""
        ratio, label = calculate_speedup(100.0, 25.0)
        self.assertEqual(ratio, 4.0)
        self.assertEqual(label, "4.0x")

        # When LSH is slower (ratio < 1.0), it should honestly report it
        ratio_slow, label_slow = calculate_speedup(10.0, 20.0)
        self.assertEqual(ratio_slow, 0.5)
        self.assertEqual(label_slow, "0.5x")

    def test_no_timing_substitutions_in_codebase(self):
        """Verify no hardcoded timing overrides exist in streamlit_app.py."""
        app_path = Path("streamlit_app.py")
        content = app_path.read_text(encoding="utf-8")

        self.assertNotIn("round(bf_time * 0.35", content)
        self.assertNotIn("round(bf_time * 0.45", content)
        self.assertNotIn("if lsh_time >= bf_time:\n                    lsh_time =", content)


if __name__ == "__main__":
    unittest.main()
