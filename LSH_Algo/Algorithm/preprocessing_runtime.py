"""
Runtime-accessible copy of the preprocessing and MinHash functions from
preprocessing.ipynb.

The notebook is intentionally preserved unchanged. The text cleaning,
character-shingle size, MurmurHash3 seed, and MinHash matrix orientation match
the notebook's existing implementation.
"""

import re
import string
import mmh3
import numpy as np


def preprocess_text(text: str) -> str:
    """Apply the same text cleaning steps as the existing notebook."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_hashed_shingles(text: str, k: int = 9, shingle_seed: int = 42):
    """Create hashed character shingles using the notebook's existing logic."""
    shingles = {text[i:i + k] for i in range(len(text) - k + 1)}
    hashed_list = [
        mmh3.hash(shingle, seed=shingle_seed, signed=False)
        for shingle in shingles
    ]
    return np.array(hashed_list, dtype=np.uint32)


def generate_signature_matrix(
    documents_shingles: list[np.ndarray], num_hashes: int = 100
):
    """
    Return a MinHash signature matrix shaped (num_hashes, num_documents).

    Empty shingle collections are rejected here with a clear message rather than
    allowing np.min([]) to fail later. The web app filters such documents first.
    """
    if not isinstance(num_hashes, int) or isinstance(num_hashes, bool) or num_hashes < 1:
        raise ValueError("num_hashes must be a positive integer.")

    num_docs = len(documents_shingles)
    signature_matrix = np.empty((num_hashes, num_docs), dtype=np.uint32)

    for hash_idx in range(num_hashes):
        for doc_idx, shingle_array in enumerate(documents_shingles):
            if len(shingle_array) == 0:
                raise ValueError(
                    "Cannot generate a MinHash signature for an empty shingle set."
                )
            scrambled_hashes = [
                mmh3.hash(x.tobytes(), seed=hash_idx, signed=False)
                for x in shingle_array
            ]
            signature_matrix[hash_idx, doc_idx] = np.min(scrambled_hashes)

    return signature_matrix


def character_shingles(text: str, k: int = 9) -> set[str]:
    """Return the original (unhashed) character shingles for exact Jaccard scoring."""
    if k < 1:
        raise ValueError("Shingle size k must be at least 1.")
    if len(text) < k:
        return set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def exact_jaccard(set_a, set_b) -> float:
    """Calculate exact Jaccard similarity on the original character-shingle sets."""
    if not set_a and not set_b:
        return 1.0
    union_size = len(set_a | set_b)
    return len(set_a & set_b) / union_size if union_size else 0.0
