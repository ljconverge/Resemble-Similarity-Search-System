"""
brute_force.py
==============
Brute-force near-duplicate / similarity search (the exact baseline that the
LSH module is benchmarked against).

Standard library only.

Places marked  # >>> MERGE POINT <<<  are the ones to touch when the
extraction and LSH parts are joined in.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path


# ======================================================================
# 1. LOADING THE CORPUS
# ======================================================================
# >>> MERGE POINT 1 <<<
# The extraction team's output format is not fixed yet. Everything below
# ends up as one dict:  {doc_id: extracted_text}.
# When their format is final, edit ONLY load_corpus() / the loaders.

def load_from_json(path: str | Path) -> dict[str, str]:
    """JSON file: [{"doc_id": ..., "text": ...}, ...]  or  {doc_id: text}.
    A record may also carry "text_path" (path to a .txt) instead of "text"."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "documents" in data:   # manifest wrapper
        data = data["documents"]
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    corpus = {}
    for rec in data:
        if "text" in rec:
            text = rec["text"]
        else:
            tp = Path(rec["text_path"])
            tp = tp if tp.is_absolute() else path.parent / tp
            text = tp.read_text(encoding="utf-8", errors="ignore")
        corpus[str(rec["doc_id"])] = str(text)
    return corpus


def load_from_folder(directory: str | Path) -> dict[str, str]:
    """Folder tree of .txt files, e.g. maths/book1.txt, physics/book2.txt.
    doc_id keeps the category folder: 'maths/book1'."""
    directory = Path(directory)
    corpus = {}
    for p in sorted(directory.rglob("*.txt")):
        doc_id = p.relative_to(directory).with_suffix("").as_posix()
        corpus[doc_id] = p.read_text(encoding="utf-8", errors="ignore")
    return corpus


def load_corpus(source) -> dict[str, str]:
    """Accept a dict (in-memory), a folder of .txt files, or a .json file."""
    if isinstance(source, dict):
        return {str(k): str(v) for k, v in source.items()}
    path = Path(source)
    if path.is_dir():
        return load_from_folder(path)
    if path.suffix.lower() == ".json":
        return load_from_json(path)
    raise ValueError(f"Unsupported corpus source: {source!r}")


# ======================================================================
# 2. PREPROCESSING + SHINGLING
# ======================================================================
# >>> MERGE POINT 2 <<<
# LSH MUST use exactly these functions (import them) so that brute force and
# LSH see identical shingle sets. Otherwise the accuracy comparison is unfair.

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)   # fixes PDF ligatures (ﬁ -> fi)
    text = text.lower()
    text = text.replace("-\n", "")               # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalise(text))


def _hash64(s: str) -> int:
    """Stable 64-bit hash. (Python's built-in hash() changes every run, so
    it must NOT be used for shingles.)"""
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def word_shingles(text: str, k: int = 5) -> set[int]:
    """Set of k-word shingles, each stored as a 64-bit integer.
    Integers keep memory small for book-sized documents and are exactly what
    MinHash needs as input (apply the h_i hash functions on these ints)."""
    tokens = tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < k:
        return {_hash64(" ".join(tokens))}
    return {_hash64(" ".join(tokens[i:i + k]))
            for i in range(len(tokens) - k + 1)}


def term_frequencies(text: str) -> Counter:
    return Counter(tokenize(text))


# ======================================================================
# 3. SIMILARITY MEASURES
# ======================================================================

def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|  (primary measure)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: set, b: set) -> float:
    """|A ∩ B| / |A|  -- how much of A appears inside B (excerpts)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity over term-frequency vectors."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(c * large[t] for t, c in small.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


MEASURES = ("jaccard", "cosine", "containment")


# ======================================================================
# 4. BRUTE-FORCE ENGINE
# ======================================================================

class BruteForceIndex:
    """Exhaustive search: every comparison is performed, nothing is pruned."""

    def __init__(self, corpus, shingle_k: int = 5):
        corpus = load_corpus(corpus)
        self.shingle_k = shingle_k
        self.doc_ids: list[str] = list(corpus.keys())
        self.shingles = {d: word_shingles(t, shingle_k) for d, t in corpus.items()}
        self.tf = {d: term_frequencies(t) for d, t in corpus.items()}
        self.last_stats = {"comparisons": 0, "elapsed_seconds": 0.0}

    # ---- internal helpers ------------------------------------------

    def _score(self, measure, q_sh, q_tf, doc_id):
        if measure == "jaccard":
            return jaccard(q_sh, self.shingles[doc_id])
        if measure == "cosine":
            return cosine(q_tf, self.tf[doc_id])
        if measure == "containment":
            return containment(q_sh, self.shingles[doc_id])
        raise ValueError(f"Unknown measure {measure!r}; use one of {MEASURES}")

    def _record(self, comparisons, start):
        self.last_stats = {
            "comparisons": comparisons,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
        }

    # ---- query: one uploaded document vs the whole corpus -----------
    # >>> MERGE POINT 3 <<<  (Flask calls this; LSH exposes the same
    # signature so the UI can switch engines.)

    def query(self, query_text: str, measure: str = "jaccard",
              top_n: int | None = 5, min_score: float = 0.0,
              exclude_id: str | None = None):
        """Return [(doc_id, score), ...] sorted best-first.
        exclude_id: skip a corpus document (used when benchmarking with a
        document that is itself in the corpus)."""
        q_sh = word_shingles(query_text, self.shingle_k)
        q_tf = term_frequencies(query_text)

        start = time.perf_counter()
        scored, comparisons = [], 0
        for doc_id in self.doc_ids:
            if doc_id == exclude_id:
                continue
            comparisons += 1
            s = self._score(measure, q_sh, q_tf, doc_id)
            if s >= min_score:
                scored.append((doc_id, s))
        self._record(comparisons, start)

        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_n] if top_n else scored

    def near_duplicates_of(self, query_text: str, threshold: float = 0.8,
                           measure: str = "jaccard", **kw):
        return self.query(query_text, measure=measure, top_n=None,
                          min_score=threshold, **kw)

    # ---- all pairs inside the corpus (ground truth for LSH) ---------
    # >>> MERGE POINT 4 <<<  ground truth for recall/precision.

    def all_pairs(self, threshold: float = 0.8, measure: str = "jaccard"):
        """Full O(N^2) scan. Exactly N(N-1)/2 comparisons.
        Returns [(doc_a, doc_b, score), ...] sorted by score."""
        start = time.perf_counter()
        comparisons, found = 0, []
        ids, n = self.doc_ids, len(self.doc_ids)
        for i in range(n):
            a = ids[i]
            for j in range(i + 1, n):
                b = ids[j]
                comparisons += 1
                if measure == "jaccard":
                    s = jaccard(self.shingles[a], self.shingles[b])
                elif measure == "cosine":
                    s = cosine(self.tf[a], self.tf[b])
                elif measure == "containment":
                    s = max(containment(self.shingles[a], self.shingles[b]),
                            containment(self.shingles[b], self.shingles[a]))
                else:
                    raise ValueError(f"Unknown measure {measure!r}")
                if s >= threshold:
                    found.append((a, b, s))
        self._record(comparisons, start)
        found.sort(key=lambda t: -t[2])
        return found

    def ground_truth_pairs(self, threshold: float = 0.8,
                           measure: str = "jaccard") -> set[frozenset]:
        return {frozenset((a, b))
                for a, b, _ in self.all_pairs(threshold, measure)}

    # ---- stats -------------------------------------------------------

    def theoretical_comparisons(self) -> int:
        n = len(self.doc_ids)
        return n * (n - 1) // 2

    def stats(self) -> dict:
        s = dict(self.last_stats)
        s["num_documents"] = len(self.doc_ids)
        s["comparisons_per_second"] = (
            round(s["comparisons"] / s["elapsed_seconds"], 2)
            if s["elapsed_seconds"] > 0 else None)
        return s


# ======================================================================
# 5. EVALUATION HELPER (used once LSH output is available)
# ======================================================================
# >>> MERGE POINT 5 <<<  pass LSH's found pairs here.

def evaluate(found_pairs, truth_pairs) -> dict:
    """Compare LSH output against brute-force ground truth.
    Both args: iterables of (a, b) or (a, b, score) tuples, or frozensets."""
    def norm(pairs):
        out = set()
        for p in pairs:
            if isinstance(p, frozenset):
                out.add(p)
            else:
                out.add(frozenset((p[0], p[1])))
        return out

    found, truth = norm(found_pairs), norm(truth_pairs)
    tp = len(found & truth)
    precision = tp / len(found) if found else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"true_pairs": len(truth), "found_pairs": len(found),
            "true_positives": tp, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


# ======================================================================
# 6. DEMO
# ======================================================================

if __name__ == "__main__":
    demo = {
        "maths/book_a": "Locality sensitive hashing groups similar items into the "
                        "same buckets so that only items sharing a bucket need to "
                        "be compared against one another in the search.",
        "maths/book_a_edited": "Locality sensitive hashing groups similar items into "
                               "the same buckets so that only items sharing a bucket "
                               "must be compared with one another in the search.",
        "physics/book_b": "The mitochondrion is an organelle that generates most of "
                          "the chemical energy needed to power a cell's biochemical "
                          "reactions in living organisms.",
        "english/book_c": "Brute force comparison evaluates every possible pair of "
                          "documents, which becomes impractical as the collection "
                          "grows very large.",
    }
    idx = BruteForceIndex(demo, shingle_k=3)

    query = ("Locality sensitive hashing groups similar items into the same "
             "buckets so that only items sharing a bucket need to be compared "
             "against each other in the search.")

    print("Top matches (jaccard):")
    for d, s in idx.query(query, top_n=3):
        print(f"  {d:<22} {s:.4f}")
    print("Query stats:", idx.stats())

    truth = idx.all_pairs(threshold=0.3)
    print("\nAll-pairs (jaccard >= 0.3):")
    for a, b, s in truth:
        print(f"  {a} <-> {b}  {s:.4f}")
    print("All-pairs stats:", idx.stats(),
          "| theoretical:", idx.theoretical_comparisons())

    print("\nEvaluate a pretend LSH result:",
          evaluate([("maths/book_a", "maths/book_a_edited")], truth))
