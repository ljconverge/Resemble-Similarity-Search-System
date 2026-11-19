# Integration tests 

import unittest
from Algorithm.preprocessing import clean_text, extract_term_frequencies, probability_jaccard
from Algorithm.brute_force import brute_force_all_pairs, brute_force_query
from Algorithm.lsh import LSH, ProbMinHashLSH
from Algorithm.probminhash import ProbMinHasher, estimate_similarity
from Algorithm.evaluation import evaluate_accuracy, calculate_speedup


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "Doc_A": "Deep neural networks are trained with gradient descent optimization algorithms.",
            "Doc_B": "Deep neural networks are trained using stochastic gradient descent optimization.",
            "Doc_C": "Quantum computing harnesses superposition and entanglement for physics simulation.",
        }

    def test_step1_preprocessing(self):
        cleaned = clean_text("Hello, World! 123")
        self.assertEqual(cleaned, "hello world 123")
        freqs = extract_term_frequencies("hello hello world")
        self.assertEqual(freqs["hello"], 2.0)
        self.assertEqual(freqs["world"], 1.0)

        # Test exact similarity
        sim_ab = probability_jaccard(
            extract_term_frequencies(self.docs["Doc_A"]),
            extract_term_frequencies(self.docs["Doc_B"]),
        )
        self.assertGreater(sim_ab, 0.5)

    def test_step2_brute_force(self):
        res = brute_force_all_pairs(self.docs, threshold=0.30)
        self.assertEqual(res["comparisons"], 3)  # 3 * 2 / 2
        self.assertGreaterEqual(len(res["matches"]), 1)

    def test_step3_pure_lsh(self):
        lsh = LSH(num_hashes=20, num_bands=5)
        # Test signature indexing
        hasher = ProbMinHasher(num_hashes=20, seed=42)
        sigs = {d: hasher.compute_signature(extract_term_frequencies(t)) for d, t in self.docs.items()}
        lsh.index_signatures(sigs)
        candidates, _ = lsh.get_candidate_pairs()
        self.assertIsInstance(candidates, set)

    def test_step4_probminhash(self):
        hasher = ProbMinHasher(num_hashes=100, seed=42)
        freq_a = extract_term_frequencies(self.docs["Doc_A"])
        freq_b = extract_term_frequencies(self.docs["Doc_B"])
        sig_a = hasher.compute_signature(freq_a)
        sig_b = hasher.compute_signature(freq_b)
        self.assertEqual(len(sig_a), 100)
        self.assertEqual(len(sig_b), 100)

        est_sim = estimate_similarity(sig_a, sig_b)
        self.assertGreater(est_sim, 0.4)

    def test_step5_evaluation(self):
        gt = [{"doc_a": "Doc_A", "doc_b": "Doc_B"}]
        ret = [{"doc_a": "Doc_A", "doc_b": "Doc_B"}]
        metrics = evaluate_accuracy(ret, gt)
        self.assertEqual(metrics["accuracy_pct"], 100.0)
        self.assertEqual(metrics["recall_pct"], 100.0)
        self.assertEqual(metrics["precision_pct"], 100.0)

        speedup_ratio, speedup_str = calculate_speedup(10.0, 2.0)
        self.assertEqual(speedup_ratio, 5.0)
        self.assertEqual(speedup_str, "5.0x")


if __name__ == "__main__":
    unittest.main()
