"""
brute_force_k1.py
=====================
Brute-force similarity search using 1-word shingles (every single word is a shingle).

NO HASHING. Every shingle is kept as plain text, so the sets that are compared
contain the real single words (for example "smart").

What the code does, step by step
  1. normalise the text    lowercase, join hyphenated line breaks, collapse spaces
  2. tokenize              words = runs of letters, digits and apostrophes
  3. build the shingle set each document becomes a SET of single words (text strings)
  4. compare EVERY pair    Jaccard = |A intersect B| / |A union B|
                           n documents -> n(n-1)/2 comparisons, nothing skipped

With k = 1 a document is simply its SET of distinct words, so word order
and phrasing are ignored: two documents match wherever they use the same words.

Usage (from the project folder):
    python brute_force_k1.py Document
    python brute_force_k1.py Document --out results_k1

Needs in the same folder: document_loader.py, extract_books.py, text_quality.py
(only to READ the .pdf / .docx / .txt files; the similarity code is below).

Files written to --out (default results_k1):
    report_k1.txt               everything printed on screen
    pairs_k1.csv                every pair: shingles, shared, union, Jaccard
    similarity_matrix_k1.csv    the full document-by-document matrix
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import time
import unicodedata
from pathlib import Path

from document_loader import load_documents

K = 1                       # shingle size in words

# ----------------------------------------------------------------------
# 1. Text cleaning
# ----------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> str:
    """Lowercase, fix PDF ligatures, join 'exam-<newline>ple', collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalise(text))


# ----------------------------------------------------------------------
# 2. Shingles: plain text, no hashing
# ----------------------------------------------------------------------

def word_shingles(text: str, k: int = K) -> set[str]:
    """Set of k-word shingles, each stored as the actual text, e.g. "smart"."""
    tokens = tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


# ----------------------------------------------------------------------
# 3. Similarity
# ----------------------------------------------------------------------

def overlap(a: set, b: set):
    """(shared, union) sizes of two sets."""
    shared = len(a & b)
    return shared, len(a) + len(b) - shared


def jaccard(a: set, b: set) -> float:
    """|A intersect B| / |A union B|."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    shared, union = overlap(a, b)
    return shared / union


# ----------------------------------------------------------------------
# 4. The brute-force engine
# ----------------------------------------------------------------------

class BruteForceIndex:
    """Exhaustive search. No hashing, no buckets, no pruning."""

    def __init__(self, corpus: dict[str, str], k: int = K):
        self.k = k
        self.doc_ids = list(corpus)
        t0 = time.perf_counter()
        self.shingles = {d: word_shingles(t, k) for d, t in corpus.items()}
        self.build_seconds = time.perf_counter() - t0
        self.last_stats = {"comparisons": 0, "elapsed_seconds": 0.0}

    def query(self, text: str, top_n: int | None = 5, exclude_id: str | None = None):
        """Compare one document against EVERY document. Best match first."""
        q = word_shingles(text, self.k)
        start = time.perf_counter()
        scored, comparisons = [], 0
        for doc_id in self.doc_ids:
            if doc_id == exclude_id:
                continue
            comparisons += 1
            scored.append((doc_id, jaccard(q, self.shingles[doc_id])))
        self.last_stats = {"comparisons": comparisons,
                           "elapsed_seconds": time.perf_counter() - start}
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_n] if top_n else scored

    def all_pairs(self):
        """Every pair once: exactly n(n-1)/2 comparisons."""
        rows, ids = [], self.doc_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                shared, union = overlap(self.shingles[a], self.shingles[b])
                rows.append({"a": a, "b": b,
                             "size_a": len(self.shingles[a]),
                             "size_b": len(self.shingles[b]),
                             "shared": shared, "union": union,
                             "jaccard": jaccard(self.shingles[a], self.shingles[b])})
        return rows

    def timing(self):
        """Clean timing of all comparisons, plus the median of one comparison."""
        ids, sets = self.doc_ids, self.shingles
        ns = time.perf_counter_ns
        t0 = ns()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                jaccard(sets[ids[i]], sets[ids[j]])
        total = (ns() - t0) / 1e9
        one = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                t = ns()
                jaccard(sets[ids[i]], sets[ids[j]])
                one.append((ns() - t) / 1000.0)
        return {"comparisons": len(ids) * (len(ids) - 1) // 2,
                "all_pairs_seconds": total,
                "median_comparison_us": statistics.median(one) if one else 0.0,
                "build_seconds": self.build_seconds}


# ----------------------------------------------------------------------
# 5. Run on a folder, print and save
# ----------------------------------------------------------------------

LINES: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    LINES.append(line)


def analyse(folder: str) -> dict:
    """Load the folder, run the brute force, return all results (used by compare_k1_k3.py)."""
    corpus, info = load_documents(Path(folder))
    if len(corpus) < 2:
        raise SystemExit("Need at least 2 readable documents (.pdf, .docx or .txt).")
    index = BruteForceIndex(corpus)
    names = index.doc_ids
    pairs = index.all_pairs()

    pos = {n: i for i, n in enumerate(names)}
    matrix = [[1.0] * len(names) for _ in names]
    for p in pairs:
        matrix[pos[p["a"]]][pos[p["b"]]] = matrix[pos[p["b"]]][pos[p["a"]]] = p["jaccard"]

    queries = {n: index.query(corpus[n], top_n=3, exclude_id=n) for n in names}
    return {"k": K, "names": names,
            "words": {d: w for d, w, _ in info},
            "shingle_counts": {d: len(s) for d, s in index.shingles.items()},
            "pairs": pairs, "matrix": matrix, "queries": queries,
            "timing": index.timing()}


def print_report(res: dict, folder: str) -> None:
    names, pairs = res["names"], res["pairs"]
    n = len(names)
    emit(f"BRUTE FORCE, k = {K} (single words), no hashing")
    emit(f"Folder: {Path(folder).resolve()}\n")

    emit(f"{'#':>3} {'document':<18}{'words':>8}{'distinct shingles':>20}")
    for i, name in enumerate(names, 1):
        emit(f"{i:>3} {name[:18]:<18}{res['words'][name]:>8,}{res['shingle_counts'][name]:>20,}")
    emit("(distinct shingles = size of the set; repeated shingles count once)\n")

    emit(f"Jaccard similarity matrix (k = {K}):")
    emit(" " * 19 + "".join(f"{i + 1:>7}" for i in range(n)))
    for i in range(n):
        row = "".join("      -" if i == j else f"{res['matrix'][i][j]:7.3f}" for j in range(n))
        emit(f"{i + 1:>3} {names[i][:14]:<14}  {row}")
    emit()

    ranked = sorted(pairs, key=lambda r: -r["jaccard"])
    emit(f"All {len(ranked)} pairs, most similar first:")
    emit(f"{'pair':<22}{'|A|':>7}{'|B|':>7}{'shared':>8}{'union':>8}{'Jaccard':>9}")
    for r in ranked:
        emit(f"{r['a'][:10] + ' / ' + r['b'][:10]:<22}{r['size_a']:>7}{r['size_b']:>7}"
             f"{r['shared']:>8}{r['union']:>8}{r['jaccard']:>9.4f}")
    emit("Jaccard = shared / union, where union = |A| + |B| - shared.\n")

    vals = [r["jaccard"] for r in pairs]
    emit("Summary over all pairs:")
    emit(f"  mean {statistics.mean(vals):.4f}   median {statistics.median(vals):.4f}   "
         f"max {max(vals):.4f}   min {min(vals):.4f}")
    for t in (0.8, 0.5, 0.2, 0.05):
        emit(f"  pairs with Jaccard >= {t:<4}: {sum(v >= t for v in vals)} of {len(vals)}")
    emit()

    emit("Each document used as a query against all the others (top 3):")
    emit(f"{'query':<14}{'comparisons':>12}  top matches")
    for name in names:
        top = ", ".join(f"{d} ({s:.3f})" for d, s in res["queries"][name])
        emit(f"{name[:14]:<14}{n - 1:>12}  {top}")
    emit()

    t = res["timing"]
    emit("Timing:")
    emit(f"  building the shingle sets   : {t['build_seconds'] * 1000:.2f} ms")
    emit(f"  one comparison (median)     : {t['median_comparison_us']:.1f} microseconds")
    emit(f"  all {t['comparisons']} comparisons         : {t['all_pairs_seconds'] * 1000:.2f} ms")


def save_files(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = res["names"]
    with open(out / f"pairs_k{K}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_a", "doc_b", "shingles_a", "shingles_b", "shared", "union", "jaccard"])
        for r in sorted(res["pairs"], key=lambda r: -r["jaccard"]):
            w.writerow([r["a"], r["b"], r["size_a"], r["size_b"], r["shared"],
                        r["union"], f"{r['jaccard']:.6f}"])
    with open(out / f"similarity_matrix_k{K}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for name, row in zip(names, res["matrix"]):
            w.writerow([name] + [f"{v:.6f}" for v in row])
    (out / f"report_k{K}.txt").write_text("\n".join(LINES), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Brute-force similarity, k=1, no hashing.")
    ap.add_argument("folder", nargs="?", default="synthetic_datasets",
                    help="folder with .pdf / .docx / .txt files (default: Document)")
    ap.add_argument("--out", default=f"results_k{K}")
    args = ap.parse_args()
    if not Path(args.folder).is_dir():
        raise SystemExit(f"Folder not found: {Path(args.folder).resolve()}")

    res = analyse(args.folder)
    print_report(res, args.folder)
    save_files(res, Path(args.out))
    print(f"\nFiles written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
