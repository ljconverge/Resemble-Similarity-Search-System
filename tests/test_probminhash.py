# Unit tests for core Probability Jaccard and ProbMinHash mathematical properties

import unittest
import math
import numpy as np
from Algorithm.probminhash import (
    preprocess_text,
    extract_term_frequencies,
    probability_jaccard,
    generalized_jaccard,
    ProbMinHasher,
)


class TestProbMinHash(unittest.TestCase):
    def test_preprocess_and_shingles(self):
        text = "ProbMinHash is a Locality-Sensitive Hashing algorithm!"
        tokens = preprocess_text(text)
        self.assertIn("probminhash", tokens)

        weights = extract_term_frequencies(text, shingle_k=2, shingle_mode="word")
        self.assertTrue(len(weights) > 0)
        # Term frequencies are positive integers (raw counts)
        self.assertTrue(all(v > 0 for v in weights.values()))

    def test_probability_jaccard_identical(self):
        p = {"token_a": 0.6, "token_b": 0.4}
        sim = probability_jaccard(p, p)
        self.assertTrue(math.isclose(sim, 1.0, rel_tol=1e-6))

    def test_probability_jaccard_disjoint(self):
        p = {"apple": 0.7, "banana": 0.3}
        q = {"cherry": 0.5, "date": 0.5}
        sim = probability_jaccard(p, q)
        self.assertEqual(sim, 0.0)

    def test_probability_jaccard_symmetry(self):
        p = {"x": 0.5, "y": 0.3, "z": 0.2}
        q = {"y": 0.4, "z": 0.1, "w": 0.5}
        sim_pq = probability_jaccard(p, q)
        sim_qp = probability_jaccard(q, p)
        self.assertTrue(math.isclose(sim_pq, sim_qp, rel_tol=1e-6))
        self.assertTrue(0.0 <= sim_pq <= 1.0)

    def test_probminhash_signature_shape(self):
        hasher = ProbMinHasher(num_hashes=64, seed=42)
        text = "Machine learning algorithms analyze heavy-tailed data distributions."
        weights = extract_term_frequencies(text, shingle_k=2)
        sig = hasher.compute_signature(weights)
        self.assertIsInstance(sig, np.ndarray)
        self.assertEqual(sig.shape, (64,))
        self.assertEqual(sig.dtype, np.uint32)

    def test_probminhash_collision_approximation(self):
        """
        Otmar Ertl (2022) Theorem: Pr(h(p) == h(q)) = J_P(p, q).
        Collision rate must approximate J_P.
        """
        doc_a = "quantum computing algorithms solve discrete optimization problems using superposition and entanglement in physics"
        doc_b = "quantum computing systems solve continuous optimization problems using superposition and decoherence in physics"

        p = extract_term_frequencies(doc_a, shingle_k=2)
        q = extract_term_frequencies(doc_b, shingle_k=2)
        exact_jp = probability_jaccard(p, q)

        hasher = ProbMinHasher(num_hashes=600, seed=123)
        sig_a = hasher.compute_signature(p)
        sig_b = hasher.compute_signature(q)

        collision_rate = np.mean(sig_a == sig_b)
        self.assertLess(abs(collision_rate - exact_jp), 0.08)


if __name__ == "__main__":
    unittest.main()
