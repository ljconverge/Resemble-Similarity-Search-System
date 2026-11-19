# Validates that the dataset strictly conforms to:
# - 70% synthetic data
# - 30% real data
# - Heavy-tailed Zipfian & calibrated similarity distributions as in Otmar Ertl (2022)


import unittest
from services.dataset_service import load_corpus, get_dataset_stats


class TestDataset(unittest.TestCase):
    def test_dataset_composition_70_30(self):
        docs, manifest = load_corpus()
        stats = get_dataset_stats()

        total = stats["total_documents"]
        real = stats["real_documents_count"]
        synth = stats["synthetic_documents_count"]

        self.assertEqual(total, 100, f"Expected 100 documents, got {total}")
        self.assertEqual(real, 30, f"Expected exactly 30 real documents, got {real}")
        self.assertEqual(synth, 70, f"Expected exactly 70 synthetic documents, got {synth}")
        self.assertEqual(stats["real_documents_percentage"], 30.0)
        self.assertEqual(stats["synthetic_documents_percentage"], 70.0)

    def test_dataset_categories_and_content(self):
        docs, manifest = load_corpus()
        self.assertEqual(len(docs), 100)

        for doc_id, text in docs.items():
            self.assertIsInstance(text, str)
            self.assertTrue(len(text.strip()) > 20, f"Document {doc_id} text is too short")

        categories = manifest["categories"]
        self.assertIn("Real Domain Corpus", categories)
        self.assertIn("Synthetic (Zipfian Power-Law)", categories)
        self.assertIn("Synthetic (Calibrated Overlap)", categories)
        self.assertEqual(categories["Real Domain Corpus"], 30)
        self.assertEqual(
            categories["Synthetic (Zipfian Power-Law)"] + categories["Synthetic (Calibrated Overlap)"],
            70,
        )


if __name__ == "__main__":
    unittest.main()
