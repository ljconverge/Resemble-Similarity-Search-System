"""
lsh_engine.py  --  the ONE algorithm file of the project
=========================================================

Everything that decides a similarity result lives here, so LSH and brute force
are guaranteed to see the SAME cleaned text and the SAME shingle sets:

    1. Preprocessing        preprocess_text()            (notebook steps)
    2. Shingling            make_shingles()              word OR character, size k
    3. Exact similarity     jaccard()
    4. MinHash              minhash_signatures()         "notebook" (mmh3) or "fast"
    5. LSH banding          generate_candidates()        from scratch, no LSH library
    6. Verification         verify_candidates()          exact Jaccard on candidates
    7. Brute force          brute_force_all_pairs()      all n(n-1)/2 pairs
    8. Evaluation           evaluate()                   recall / precision / F1
    9. Full run             run_analysis()               used by app.py
   10. Command line         python Algorithm/lsh_engine.py Document --mode word --k 3

Replaces the old Algorithm/lsh.py and Algorithm/preprocessing_runtime.py.
Dependencies: numpy, mmh3 (pypdf only for the command-line PDF reader).
"""

from __future__ import annotations

import random
import re
import statistics
import string
import time
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import mmh3
import numpy as np

# ----------------------------------------------------------------------------
# Settings shared by the app, the tests and the command line
# ----------------------------------------------------------------------------
SHINGLE_MODES = ("word", "char")
MINHASH_METHODS = ("fast", "notebook")
DEFAULT_MODE = "word"
DEFAULT_K = {"word": 3, "char": 9}
K_LIMITS = {"word": (1, 20), "char": (1, 30)}
NUM_HASHES = 100
SHINGLE_SEED = 42           # notebook: mmh3 seed for turning a shingle into an int
MERSENNE_PRIME = (1 << 31) - 1
_FAST_CHUNK = 20_000        # shingles processed at once (keeps memory small for books)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ============================================================================
# 1. PREPROCESSING
# ============================================================================

def preprocess_text(text: str) -> str:
    """Clean raw text. Same steps as preprocessing.ipynb, plus two PDF fixes.

    0a. Unicode NFKC        PDF ligatures like 'ﬁ' become 'fi'          (added)
    0b. join 'exam-\\nple'   words split across a line break are re-joined (added)
    1.  newlines -> spaces
    2.  lowercase
    3.  remove punctuation (string.punctuation)
    4.  collapse repeated whitespace
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\r?\n", "", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    # PATCH 8: string.punctuation is ASCII only. Also delete Unicode punctuation
    # (curly quotes, en/em dashes, bullets...) the same way, so "don’t" == "don't".
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# 2. SHINGLING
# ============================================================================

def validate_shingle_settings(mode: str, k: int) -> None:
    if mode not in SHINGLE_MODES:
        raise ValueError(f"Shingle mode must be one of {SHINGLE_MODES}, got {mode!r}.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("Shingle size k must be an integer.")
    lo, hi = K_LIMITS[mode]
    if not lo <= k <= hi:
        unit = "words" if mode == "word" else "characters"
        raise ValueError(f"Shingle size for {mode} shingles must be {lo}-{hi} {unit}.")


def make_shingles(clean_text: str, mode: str = DEFAULT_MODE, k: int | None = None) -> set[str]:
    """Set of shingles (plain strings) of an ALREADY CLEANED text.

    mode="word": k consecutive words,      "machine learning is"
    mode="char": k consecutive characters, "machine l"

    Text shorter than k (but not empty) becomes ONE shingle, so short documents
    are still comparable instead of silently disappearing.
    """
    k = DEFAULT_K[mode] if k is None else k
    validate_shingle_settings(mode, k)
    units = clean_text.split() if mode == "word" else clean_text
    if len(units) == 0:
        return set()
    joiner = " " if mode == "word" else ""
    if len(units) < k:
        return {joiner.join(units)}
    return {joiner.join(units[i:i + k]) for i in range(len(units) - k + 1)}


def shingle_ids(shingles: set[str]) -> np.ndarray:
    """Each shingle string -> unsigned 32-bit MurmurHash3 (seed 42), as in the notebook."""
    return np.array([mmh3.hash(s, seed=SHINGLE_SEED, signed=False) for s in shingles],
                    dtype=np.uint32)


# ============================================================================
# 3. EXACT SIMILARITY
# ============================================================================

def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|. The union is never built: |A ∪ B| = |A| + |B| - |A ∩ B|.
    Convention: two empty sets are identical (1.0)."""
    if not a and not b:
        return 1.0
    if len(a) > len(b):
        a, b = b, a
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ============================================================================
# 4. MINHASH  (two interchangeable methods)
# ============================================================================

def _minhash_notebook(id_arrays: list[np.ndarray], num_hashes: int) -> list[list[int]]:
    """The notebook's method, unchanged: hash function i = mmh3 with seed i,
    applied to the 4-byte form of every shingle ID; keep the minimum.
    Correct but slow: num_hashes * total_shingles Python-level mmh3 calls."""
    n = len(id_arrays)
    matrix = np.empty((num_hashes, n), dtype=np.uint32)
    for hash_idx in range(num_hashes):
        for doc_idx, ids in enumerate(id_arrays):
            matrix[hash_idx, doc_idx] = min(
                mmh3.hash(x.tobytes(), seed=hash_idx, signed=False) for x in ids)
    return [matrix[:, j].tolist() for j in range(n)]


def make_hash_params(num_hashes: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """a_i in [1, p-1], b_i in [0, p-1] for h_i(x) = (a_i * x + b_i) mod p. Seeded."""
    rng = random.Random(seed)
    a = [rng.randint(1, MERSENNE_PRIME - 1) for _ in range(num_hashes)]
    b = [rng.randint(0, MERSENNE_PRIME - 1) for _ in range(num_hashes)]
    return np.array(a, dtype=np.uint64), np.array(b, dtype=np.uint64)


def _minhash_fast(id_arrays: list[np.ndarray], num_hashes: int, seed: int) -> list[list[int]]:
    """Same MinHash idea with universal hashing h_i(x) = (a_i x + b_i) mod (2^31 - 1).

    NumPy evaluates all num_hashes functions on a block of shingles at once.
    Overflow check: a < 2^31 and x < 2^32, so a*x + b < 2^63 fits in uint64.
    """
    a, b = make_hash_params(num_hashes, seed)
    a, b = a[:, None], b[:, None]
    p = np.uint64(MERSENNE_PRIME)
    signatures = []
    for ids in id_arrays:
        x_all = ids.astype(np.uint64)
        best = np.full(num_hashes, MERSENNE_PRIME, dtype=np.uint64)
        for start in range(0, len(x_all), _FAST_CHUNK):
            x = x_all[start:start + _FAST_CHUNK][None, :]
            best = np.minimum(best, ((a * x + b) % p).min(axis=1))
        signatures.append(best.astype(np.int64).tolist())
    return signatures


def minhash_signatures(id_arrays: list[np.ndarray], num_hashes: int = NUM_HASHES,
                       method: str = "fast", seed: int = 42) -> list[list[int]]:
    """One signature (list of num_hashes ints) per document, in input order."""
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes < 1:
        raise ValueError("num_hashes must be a positive integer.")
    if method not in MINHASH_METHODS:
        raise ValueError(f"MinHash method must be one of {MINHASH_METHODS}, got {method!r}.")
    if any(len(ids) == 0 for ids in id_arrays):
        raise ValueError("Cannot build a MinHash signature for an empty shingle set.")
    if method == "notebook":
        return _minhash_notebook(id_arrays, num_hashes)
    return _minhash_fast(id_arrays, num_hashes, seed)


def estimate_jaccard(sig_a, sig_b) -> float:
    """Fraction of equal signature positions (the MinHash estimate of Jaccard)."""
    if len(sig_a) != len(sig_b) or not sig_a:
        raise ValueError("Signatures must be non-empty and of equal length.")
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


# ============================================================================
# 5. LSH BANDING  (from Algorithm/lsh.py, unchanged logic)
# ============================================================================

def generate_candidates(signatures: dict, bands: int = 20):
    """Band every signature; documents with an identical band share a bucket.

    Returns (candidates, buckets):
      candidates: set of sorted (doc_a, doc_b) tuples, each pair once
      buckets:    {(band_index, band_tuple): [doc_id, ...]}
    The band index is part of the key so equal values in different bands
    never collide by accident.
    """
    if not signatures:
        return set(), {}
    if isinstance(bands, bool) or not isinstance(bands, int):
        raise ValueError("The number of bands must be an integer.")
    length = len(next(iter(signatures.values())))
    if length == 0:
        raise ValueError("Signatures cannot be empty.")
    if not 1 <= bands <= length:
        raise ValueError(f"bands must be between 1 and {length}.")
    if length % bands != 0:
        raise ValueError(f"Signature length ({length}) must be divisible by bands ({bands}).")

    rows = length // bands
    buckets: dict = {}
    for doc_id, sig in signatures.items():
        if len(sig) != length:
            raise ValueError("All signatures must have equal length.")
        for band in range(bands):
            key = (band, tuple(int(v) for v in sig[band * rows:(band + 1) * rows]))
            buckets.setdefault(key, []).append(doc_id)

    candidates = set()
    for docs in buckets.values():
        if len(docs) > 1:
            candidates.update(combinations(sorted(docs), 2))
    return candidates, buckets


def shared_band_counts(buckets: dict) -> Counter:
    """How many bands each candidate pair shared (for explainable results)."""
    counts = Counter()
    for docs in buckets.values():
        if len(docs) > 1:
            counts.update(combinations(sorted(docs), 2))
    return counts


def candidate_probability(s: float, bands: int, rows: int) -> float:
    """Theory: P(pair with Jaccard s becomes a candidate) = 1 - (1 - s^r)^b."""
    return 1 - (1 - s ** rows) ** bands


def approx_threshold(bands: int, rows: int) -> float:
    """Similarity where the S-curve is steepest, about (1/b)^(1/r)."""
    return (1 / bands) ** (1 / rows)


# ============================================================================
# 6. VERIFICATION  and  7. BRUTE FORCE   (identical scoring rule)
# ============================================================================

def verify_candidates(candidates, shingle_sets: dict, threshold: float):
    """Exact Jaccard on LSH candidates only. Returns ([(a, b, score)], comparisons)."""
    scored = [(a, b, jaccard(shingle_sets[a], shingle_sets[b])) for a, b in candidates]
    scored.sort(key=lambda r: (-r[2], r[0], r[1]))
    return scored, len(scored)


def brute_force_all_pairs(shingle_sets: dict, threshold: float = 0.0):
    """Exhaustive baseline: every unordered pair, nothing skipped.
    Returns ([(a, b, score)] with score >= threshold, comparisons)."""
    ids = sorted(shingle_sets)
    found, comparisons = [], 0
    for i in range(len(ids)):
        a = shingle_sets[ids[i]]
        for j in range(i + 1, len(ids)):
            comparisons += 1
            s = jaccard(a, shingle_sets[ids[j]])
            if s >= threshold:
                found.append((ids[i], ids[j], s))
    found.sort(key=lambda r: (-r[2], r[0], r[1]))
    return found, comparisons


# ============================================================================
# 8. EVALUATION
# ============================================================================

def evaluate(found_pairs, truth_pairs) -> dict:
    """LSH output vs brute-force ground truth. Accepts (a, b) or (a, b, score)."""
    norm = lambda pairs: {frozenset(p[:2]) if not isinstance(p, frozenset) else p for p in pairs}
    found, truth = norm(found_pairs), norm(truth_pairs)
    tp = len(found & truth)
    # PATCH 5: with nothing to find (or nothing found) a percentage is meaningless,
    # so it is None (shown as "n/a") instead of a misleading 100%.
    precision = round(tp / len(found), 4) if found else None
    recall = round(tp / len(truth), 4) if truth else None
    if recall is None:
        f1 = None
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {"true_pairs": len(truth), "found_pairs": len(found), "true_positives": tp,
            "false_positives": len(found - truth), "false_negatives": len(truth - found),
            "precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# 9. FULL RUN  (called by app.py and the command line)
# ============================================================================

def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _pct_text(score: float, threshold: float) -> str:
    """PATCH 7: score as a percentage with 2 decimals, but add decimals when
    rounding would put it on the wrong side of the threshold (0.79996 -> "79.9960")."""
    limit = round(threshold * 100, 6)
    for decimals in (2, 4, 6, 8):
        text = f"{score * 100:.{decimals}f}"
        if (float(text) >= limit) == (score >= threshold):
            return text
    return text


def run_analysis(raw_documents: dict, threshold: float = 0.8, bands: int = 20,
                 shingle_mode: str = DEFAULT_MODE, k: int | None = None,
                 num_hashes: int = NUM_HASHES, minhash_method: str = "fast",
                 seed: int = 42, run_brute_force: bool = True,
                 timing_repeats: int = 1) -> dict:
    """raw_documents: {name: raw_text}. threshold in [0, 1].

    Timing is fair: cleaning + shingling is done ONCE, timed separately, and
    shared by both engines. LSH time = shingle hashing + MinHash + banding +
    verification. Brute-force time = all-pairs Jaccard. PDF reading is never
    included. Each engine is timed `timing_repeats` times; the median is reported.
    """
    k = DEFAULT_K[shingle_mode] if k is None else k
    validate_shingle_settings(shingle_mode, k)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    if num_hashes % bands != 0:
        raise ValueError(f"num_hashes ({num_hashes}) must be divisible by bands ({bands}).")

    warnings: list[str] = []
    t0 = time.perf_counter()
    shingle_sets, documents = {}, []
    for name, raw in raw_documents.items():
        clean = preprocess_text(raw)
        sh = make_shingles(clean, shingle_mode, k)
        if not sh:
            warnings.append(f"{name}: skipped because it has no text after cleaning.")
            continue
        shingle_sets[name] = sh
        documents.append({"name": name, "characters": len(clean),
                          "words": len(clean.split()), "shingles": len(sh)})
    names = list(shingle_sets)
    t_prep = time.perf_counter() - t0
    if not names:
        raise ValueError("No usable text documents were found.")
    if isinstance(timing_repeats, bool) or not isinstance(timing_repeats, int) or timing_repeats < 1:
        raise ValueError("timing_repeats must be a positive integer.")

    # ---- LSH --------------------------------------------------------------
    # PATCH 3: turning shingles into numbers is needed only by LSH, so it is
    #          now timed as LSH work (it used to sit in the shared time).
    # PATCH 4: repeat and take the median; every repeat gives identical results.
    runs = []
    for _ in range(timing_repeats):
        t1 = time.perf_counter()
        id_arrays = [shingle_ids(shingle_sets[n]) for n in names]
        t1b = time.perf_counter()
        sigs = minhash_signatures(id_arrays, num_hashes, minhash_method, seed)
        t2 = time.perf_counter()
        candidates, buckets = generate_candidates(dict(zip(names, sigs)), bands)
        t3 = time.perf_counter()
        scored, lsh_comparisons = verify_candidates(candidates, shingle_sets, threshold)
        t4 = time.perf_counter()
        runs.append((t1b - t1, t2 - t1b, t3 - t2, t4 - t3, t4 - t1))
    t_hash, t_minhash, t_band, t_verify, t_lsh = (statistics.median(c) for c in zip(*runs))
    band_counts = shared_band_counts(buckets)
    lsh_found = [r for r in scored if r[2] >= threshold]

    # ---- Brute force --------------------------------------------------------
    brute, evaluation, t_brute = None, None, None
    if run_brute_force:
        bf_times = []
        for _ in range(timing_repeats):
            t5 = time.perf_counter()
            truth, bf_comparisons = brute_force_all_pairs(shingle_sets, threshold)
            bf_times.append(time.perf_counter() - t5)
        t_brute = statistics.median(bf_times)
        evaluation = evaluate(lsh_found, truth)
        found_keys = {frozenset(r[:2]) for r in lsh_found}
        brute = {"pairs": [{"doc_a": a, "doc_b": b, "similarity": round(s * 100, 2),
                            "similarity_text": _pct_text(s, threshold),
                            "found_by_lsh": frozenset((a, b)) in found_keys}
                           for a, b, s in truth],
                 "comparisons": bf_comparisons}

    n = len(names)
    possible = n * (n - 1) // 2
    rows = num_hashes // bands
    rows_out = [{"doc_a": a, "doc_b": b, "similarity": round(s * 100, 2),
                 "similarity_text": _pct_text(s, threshold),
                 "shared_buckets": band_counts[(a, b)], "is_match": s >= threshold,
                 "status": "Near duplicate" if s >= threshold else "Below threshold"}
                for a, b, s in scored]

    metrics = {
        "document_count": n, "possible_pairs": possible,
        "candidate_count": len(candidates), "near_duplicate_count": len(lsh_found),
        "lsh_comparisons": lsh_comparisons,
        "candidate_reduction": round((1 - len(candidates) / possible) * 100, 2) if possible else 100.0,
        "threshold": round(threshold * 100, 2), "bands": bands, "rows_per_band": rows,
        "num_hashes": num_hashes, "shingle_mode": shingle_mode, "shingle_size": k,
        "shingle_unit": "words" if shingle_mode == "word" else "characters",
        "minhash_method": minhash_method,
        # PATCH 10: the notebook method always uses seeds 0..m-1, so no seed applies
        "seed": seed if minhash_method == "fast" else None,
        "approx_lsh_threshold": round(approx_threshold(bands, rows) * 100, 1),
        "bucket_count": len(buckets),
        "collision_bucket_count": sum(1 for d in buckets.values() if len(d) > 1),
        "max_bucket_size": max((len(d) for d in buckets.values()), default=0),
        "preprocessing_ms": _ms(t_prep), "hashing_ms": _ms(t_hash), "minhash_ms": _ms(t_minhash),
        "banding_ms": _ms(t_band), "verify_ms": _ms(t_verify), "lsh_ms": _ms(t_lsh),
        "elapsed_ms": _ms(t_prep + t_lsh), "timing_repeats": timing_repeats,
        # PATCH 6: share of LSH candidates that really were near-duplicates
        "candidate_precision": round(len(lsh_found) / len(candidates), 4) if candidates else None,
    }
    if run_brute_force:
        metrics.update({
            "brute_comparisons": brute["comparisons"], "brute_ms": _ms(t_brute),
            "speedup": round(t_brute / t_lsh, 3) if t_lsh > 0 else None,
            "comparison_reduction": round((1 - lsh_comparisons / brute["comparisons"]) * 100, 2)
                                    if brute["comparisons"] else 100.0,
            **evaluation,
        })
    return {"documents": documents, "warnings": warnings,
            "pairs": [r for r in rows_out if r["is_match"]],
            "rejected_pairs": [r for r in rows_out if not r["is_match"]],
            "candidate_pairs": rows_out, "brute_force": brute, "metrics": metrics}


# ============================================================================
# 10. COMMAND LINE
# ============================================================================

def load_folder(folder) -> dict[str, str]:
    """Read every .pdf / .txt in a folder (pypdf for PDFs). {file name: text}."""
    docs = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() == ".txt":
            docs[p.name] = p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            docs[p.name] = "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
    return docs


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LSH vs brute force on a folder of PDF/TXT files.")
    ap.add_argument("folder", nargs="?", default="Document")
    ap.add_argument("--mode", choices=SHINGLE_MODES, default=DEFAULT_MODE)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.8, help="0-1 (default 0.8)")
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--hashes", type=int, default=NUM_HASHES)
    ap.add_argument("--minhash", choices=MINHASH_METHODS, default="fast")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    res = run_analysis(load_folder(args.folder), args.threshold, args.bands, args.mode,
                       args.k, args.hashes, args.minhash, args.seed)
    m = res["metrics"]
    print(f"{m['document_count']} documents | {m['shingle_size']}-{m['shingle_unit'][:-1]} shingles | "
          f"{m['num_hashes']} hashes ({m['minhash_method']}) | b={m['bands']} r={m['rows_per_band']} "
          f"| threshold {m['threshold']}%")
    for w in res["warnings"]:
        print("  note:", w)
    print("\nNear duplicates found by LSH:")
    for r in res["pairs"]:
        print(f"  {r['doc_a']:<16} {r['doc_b']:<16} {r['similarity']:6.2f}%  bands={r['shared_buckets']}")
    print("\nBrute-force pairs missed by LSH:",
          [(p["doc_a"], p["doc_b"]) for p in res["brute_force"]["pairs"] if not p["found_by_lsh"]] or "none")
    print(f"\n{'':<26}{'LSH':>12}{'Brute force':>14}")
    print(f"{'exact comparisons':<26}{m['lsh_comparisons']:>12}{m['brute_comparisons']:>14}")
    print(f"{'time (ms)':<26}{m['lsh_ms']:>12.3f}{m['brute_ms']:>14.3f}")
    print(f"  hashing {m['hashing_ms']} ms, MinHash {m['minhash_ms']} ms, banding {m['banding_ms']} ms, "
          f"verify {m['verify_ms']} ms; shared cleaning + shingling {m['preprocessing_ms']} ms")
    na = lambda v: "n/a" if v is None else v
    print(f"recall {na(m['recall'])}  precision {na(m['precision'])}  F1 {na(m['f1'])}  "
          f"candidate precision {na(m['candidate_precision'])}  "
          f"speedup {m['speedup']}x  candidate reduction {m['candidate_reduction']}%")


if __name__ == "__main__":
    main()
