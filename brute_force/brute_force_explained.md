# `brute_force.py`: Line-by-Line Explanation

Project: Near-Duplicate and Similarity Search (LSH vs Brute Force)
File explained: `brute_force.py` (341 lines, standard library only)

Line numbers below refer to `brute_force.py`.

---

## 0. The big picture

Brute force means **no shortcuts**: every document is compared with every other document it could match. The module does five jobs:

| Job | Where | Purpose |
|---|---|---|
| Load the corpus | lines 25-75 | Turn the extraction team's output into `{doc_id: text}` |
| Preprocess and shingle | lines 78-121 | Turn text into sets of k-word shingles |
| Measure similarity | lines 124-158 | Jaccard (primary), containment, cosine |
| Search | lines 161-271 | Compare a query against all documents, or all pairs against each other |
| Evaluate | lines 274-299 | Score LSH's output against brute force's exact answer |

The core idea in one line:

> similarity(A, B) = |shingles(A) ∩ shingles(B)| / |shingles(A) ∪ shingles(B)|

### Tiny worked example (3-word shingles)

Document: `"I love studying data science"` gives tokens `[i, love, studying, data, science]` and shingles:

1. `i love studying`
2. `love studying data`
3. `studying data science`

If a second document is `"I love studying data analysis"`, its shingles are `i love studying`, `love studying data`, `studying data analysis`. The two sets share 2 shingles, and their union has 4. So Jaccard = 2 / 4 = 0.5.

---

## 1. Header and imports (lines 1-23)

| Lines | Code | Explanation |
|---|---|---|
| 1-11 | Docstring | Module description. It says this is the exact baseline for LSH, and that places marked `MERGE POINT` are where to touch the code when extraction and LSH are joined in. |
| 13 | `from __future__ import annotations` | Lets type hints such as `str \| Path` and `dict[str, str]` work on older Python versions (3.8 and 3.9) by not evaluating them at run time. |
| 15 | `import hashlib` | Provides `blake2b`, the stable hash used for shingles. |
| 16 | `import json` | Reads JSON corpus files and manifests. |
| 17 | `import math` | Provides `sqrt` for cosine similarity. |
| 18 | `import re` | Regular expressions for tokenising and whitespace cleanup. |
| 19 | `import time` | `perf_counter` for timing searches. |
| 20 | `import unicodedata` | Unicode normalisation (fixes PDF ligatures). |
| 21 | `from collections import Counter` | A dict subclass that counts items; used for word frequencies. |
| 22 | `from pathlib import Path` | Safe file path handling across operating systems. |

---

## 2. Loading the corpus (lines 25-75)

The goal of this section: whatever format the extraction team delivers, end up with **one dict**, `{doc_id: extracted_text}`.

### `load_from_json` (lines 33-52)

| Line | Explanation |
|---|---|
| 33 | Takes a path and returns a dict from document id to text. |
| 34-35 | Docstring: accepts a list of records, a plain dict, or records with a `text_path` instead of inline `text`. |
| 36 | Converts the path string into a `Path` object. |
| 37-38 | Opens the file as UTF-8 and parses the JSON into `data`. |
| 39-40 | If the JSON is a dict with a `"documents"` key (the manifest format used by `extract_books.py`), unwrap it so `data` becomes the list of records. |
| 41-42 | If `data` is still a dict (the format `{doc_id: text}`), return it with all keys and values converted to strings. |
| 43 | Starts an empty result dict for the list-of-records format. |
| 44 | Loops over each record. |
| 45-46 | If the record has inline `"text"`, use it. |
| 47-50 | Otherwise the text lives in a file: build the path from `text_path` (49: if the path is relative, resolve it relative to the JSON file's own folder), then read the file as UTF-8, ignoring undecodable bytes. |
| 51 | Stores the text under the record's `doc_id`. |
| 52 | Returns the finished corpus dict. |

### `load_from_folder` (lines 55-63)

| Line | Explanation |
|---|---|
| 55-57 | Reads a folder tree such as `maths/book1.txt`. The `doc_id` keeps the category folder. |
| 58 | Converts to a `Path`. |
| 59 | Empty result dict. |
| 60 | `rglob("*.txt")` finds every `.txt` file in all subfolders. `sorted` makes the order deterministic, so results are repeatable. |
| 61 | Builds the id: the path relative to the root, with `.txt` removed, using `/` separators. So `corpus/maths/book1.txt` becomes `maths/book1`. |
| 62 | Reads the file as UTF-8, ignoring bad bytes, and stores it. |
| 63 | Returns the corpus. |

### `load_corpus` (lines 66-75)

| Line | Explanation |
|---|---|
| 66-67 | Single entry point that accepts several input kinds. |
| 68-69 | If the input is already a dict (in-memory, e.g. from Flask), just clean the types and return it. |
| 70 | Otherwise treat the input as a path. |
| 71-72 | If it is a folder, use `load_from_folder`. |
| 73-74 | If it ends in `.json`, use `load_from_json`. |
| 75 | Anything else raises a clear error. |

**Merge note:** if the extraction team's format changes, only these three functions change.

---

## 3. Preprocessing and shingling (lines 78-121)

This is the most important section for fairness. The LSH module must import these exact functions, so both methods see identical shingles.

| Line | Code | Explanation |
|---|---|---|
| 85 | `_TOKEN_RE = re.compile(r"[a-z0-9']+")` | A token is a run of lowercase letters, digits, or apostrophes. Everything else (spaces, punctuation, hyphens) separates tokens. The pattern is compiled once for speed. |

### `normalise` (lines 88-93)

| Line | Explanation |
|---|---|
| 89 | `unicodedata.normalize("NFKC", text)` converts look-alike characters to standard ones. For example the PDF ligature "ﬁ" becomes "fi", so "ﬁnite" and "finite" match. |
| 90 | Lowercases, so "The" and "the" are the same. |
| 91 | Removes a hyphen followed by a newline, which rejoins words that were split across lines ("exam-\nple" becomes "example"). |
| 92 | Collapses any run of whitespace (spaces, tabs, newlines) into one space. |
| 93 | Trims spaces at both ends and returns the result. |

### `tokenize` (lines 96-97)

Normalises the text, then `findall` returns the list of tokens. For example `"Hello, World!"` becomes `["hello", "world"]`.

### `_hash64` (lines 100-104)

| Line | Explanation |
|---|---|
| 100-102 | The docstring explains why Python's built-in `hash()` must not be used: it gives different values on each program run, so results would not be reproducible and LSH could not match them. |
| 103-104 | Encodes the string as UTF-8 bytes, hashes with BLAKE2b configured for an 8-byte (64-bit) digest, and converts the 8 bytes to an integer. Same input always gives the same number. |

Why integers: a set of 64-bit numbers uses far less memory than a set of long strings, and MinHash needs numbers as input anyway.

### `word_shingles` (lines 107-117)

| Line | Explanation |
|---|---|
| 107 | `k` is the shingle size in words (default 5). |
| 111 | Gets the token list. |
| 112-113 | If the document has no tokens, return an empty set. |
| 114-115 | If the document is shorter than `k` words, use the whole document as one shingle, so short documents are still comparable. |
| 116-117 | Otherwise slide a window of `k` tokens: for `i` from 0 to `len(tokens) - k`, join `tokens[i:i+k]` with spaces, hash it, and add it to a set. Using a set removes repeated shingles, so each distinct shingle counts once. This matches the mathematical definition of Jaccard on sets. |

The number of windows is `len(tokens) - k + 1`.

### `term_frequencies` (lines 120-121)

Counts how many times each word occurs, for example `{"the": 12, "matrix": 3, ...}`. Used only by the cosine measure.

---

## 4. Similarity measures (lines 124-158)

### `jaccard` (lines 128-135)

| Line | Explanation |
|---|---|
| 130-131 | Both sets empty: define similarity as 1.0 (two empty documents are identical). This also avoids dividing by zero. |
| 132-133 | Only one is empty: similarity is 0.0. |
| 134 | `a & b` is the set intersection. Its size is the count of shared shingles. |
| 135 | Union size by inclusion-exclusion: `|A| + |B| - |A ∩ B|`. This avoids building the union set in memory. Returns intersection over union. |

Range: 0.0 (no shared shingles) to 1.0 (identical shingle sets).

### `containment` (lines 138-142)

| Line | Explanation |
|---|---|
| 140-141 | If `a` is empty, return 0.0. |
| 142 | Fraction of A's shingles that also appear in B: `|A ∩ B| / |A|`. It is **not symmetric**. It is 1.0 when A is a full excerpt of B, even though Jaccard would be low because B is much longer. |

### `cosine` (lines 145-155)

| Line | Explanation |
|---|---|
| 147-148 | If either word-count dict is empty, return 0.0. |
| 149 | Picks the smaller vocabulary as `small` and the other as `large`, so the loop below runs over fewer words. |
| 150 | Dot product: for every word in `small`, multiply its count by the same word's count in `large` (a `Counter` returns 0 for missing words), and add up. |
| 151-152 | No shared words means the dot product is 0, so return 0.0 immediately. |
| 153-154 | Vector lengths: square root of the sum of squared counts, for each document. |
| 155 | Cosine = dot product divided by the product of the lengths. Range 0.0 to 1.0. |

### `MEASURES` (line 158)

A tuple of valid measure names, used in error messages.

---

## 5. The brute-force engine (lines 161-271)

### `__init__` (lines 165-174)

| Line | Explanation |
|---|---|
| 165-166 | Class docstring: exhaustive search, nothing is pruned. |
| 168 | Takes a corpus (dict, folder, or JSON path) and the shingle size `k`. |
| 169 | Loads the corpus into a dict using `load_corpus`. |
| 170 | Remembers `k` so queries use the same shingle size as the corpus. This matters: mixing different `k` values would make results meaningless. |
| 171 | Keeps the list of document ids in a fixed order. |
| 172 | **Precomputes** the shingle set for every document once, at build time. Searches reuse these sets, so no shingling happens during a search. |
| 173 | Precomputes word counts for cosine. |
| 174 | Initial statistics (zero comparisons, zero time). |

### `_score` (lines 178-185)

Picks the right measure by name. Jaccard and containment use shingle sets (`q_sh`), cosine uses word counts (`q_tf`). An unknown name raises an error listing valid options (line 185).

### `_record` (lines 187-191)

Saves the statistics of the last search: how many comparisons were made and how many seconds elapsed (`perf_counter() - start`, rounded to 6 decimals).

### `query` (lines 197-218): one uploaded document vs the whole corpus

This is what the Flask app calls.

| Line | Explanation |
|---|---|
| 197-199 | Parameters: the query text, the measure, how many results to return (`top_n`), a minimum score, and an optional document to skip. |
| 203-204 | Builds the query's shingle set and word counts, using the same functions and `k` as the corpus. |
| 206 | Starts the timer **after** the query is preprocessed, so the measured time covers only the comparison loop. That is the part that differs between brute force and LSH. |
| 207 | `scored` collects results, `comparisons` counts them. |
| 208 | Loops over **every** document: this is the brute-force step. |
| 209-210 | Skips `exclude_id`. This is used when benchmarking with a document that is itself in the corpus, so it does not match itself. |
| 211 | Counts this comparison. |
| 212 | Computes the similarity. |
| 213-214 | Keeps the result only if it reaches `min_score`. |
| 215 | Saves comparison count and elapsed time. |
| 217 | Sorts by score descending (`-p[1]`). Ties are broken alphabetically by id (`p[0]`), so the order is the same every run. |
| 218 | Returns the top `top_n` results, or all results if `top_n` is `None` or 0. |

### `near_duplicates_of` (lines 220-223)

A convenience wrapper: returns **every** document at or above the threshold (`top_n=None`, `min_score=threshold`). `**kw` passes extra arguments such as `exclude_id` through to `query`.

### `all_pairs` (lines 228-252): every pair inside the corpus

This produces the **ground truth** that LSH is measured against.

| Line | Explanation |
|---|---|
| 231 | Starts the timer. |
| 232 | Counter for comparisons and a list for pairs that pass the threshold. |
| 233 | Shortcuts for the id list and its length `n`. |
| 234-237 | The outer loop picks document `a` (index `i`). The inner loop picks document `b` from index `i + 1` onwards. Starting at `i + 1` means each unordered pair is visited exactly once and a document is never compared with itself. |
| 238 | Counts the comparison. In total this makes `n(n-1)/2` comparisons. |
| 239-247 | Computes the score for the chosen measure. For containment (244-245) it uses the larger of the two directions, so the score does not depend on the order of `a` and `b`. An unknown measure raises an error. |
| 248-249 | If the score reaches the threshold, store `(a, b, score)`. |
| 250 | Saves statistics. |
| 251 | Sorts pairs from most to least similar. |
| 252 | Returns the list. |

### `ground_truth_pairs` (lines 254-257)

Runs `all_pairs` and returns the pairs as a **set of frozensets** (`frozenset((a, b))`). A frozenset ignores order, so `(A, B)` and `(B, A)` count as the same pair, and it can be placed inside a set for fast comparison. Note: calling this overwrites `last_stats` with the all-pairs statistics.

### `theoretical_comparisons` (lines 261-263)

Returns `n(n-1)/2` using integer division. This is the number of comparisons brute force must make, and the number to plot against LSH's candidate count.

### `stats` (lines 265-271)

| Line | Explanation |
|---|---|
| 266 | Copies the last-search stats so the original is not modified. |
| 267 | Adds the number of documents. |
| 268-270 | Adds comparisons per second, but only if elapsed time is above 0, to avoid dividing by zero. Otherwise `None`. |
| 271 | Returns the dict for the evaluation report. |

---

## 6. Evaluation helper (lines 274-299)

Used once LSH produces its own list of similar pairs.

| Line | Explanation |
|---|---|
| 279-281 | `evaluate(found_pairs, truth_pairs)`: `found_pairs` is what LSH reported, `truth_pairs` is what brute force found. |
| 282-289 | Inner helper `norm` converts any pair format (`(a, b)`, `(a, b, score)`, or frozenset) into a set of frozensets, so formats and ordering do not matter. |
| 291 | Normalises both inputs. |
| 292 | `tp` (true positives): pairs that appear in both. |
| 293 | **Precision** = correct pairs / pairs LSH reported. Defaults to 1.0 if LSH reported nothing (it made no wrong claims). |
| 294 | **Recall** = correct pairs / pairs that truly exist. Defaults to 1.0 if no true pairs exist. This is the key measure for LSH: how many real near-duplicates did it find? |
| 295-296 | **F1** = harmonic mean of precision and recall, or 0.0 if both are 0. |
| 297-299 | Returns all counts and the three scores, rounded to 4 decimals. |

---

## 7. Demo (lines 302-340)

Runs only when you execute `python brute_force.py` directly (`if __name__ == "__main__":`).

| Lines | Explanation |
|---|---|
| 307-320 | Four small documents: `book_a`, a lightly edited copy `book_a_edited`, and two unrelated texts. |
| 321 | Builds the index with 3-word shingles, because the demo texts are short. |
| 323-325 | The query: a rewording of `book_a`. |
| 327-330 | Prints the top 3 matches with their Jaccard scores, then the statistics of that query. `book_a` should rank first. |
| 332 | Runs `all_pairs` with threshold 0.3 to get the ground truth. |
| 333-337 | Prints each similar pair, then the all-pairs statistics next to the theoretical count (4 documents give 6 comparisons). |
| 339-340 | Calls `evaluate` with a pretend LSH result containing the one correct pair, which gives precision, recall, and F1 of 1.0. |

---

## 8. Complexity summary

Let N be the number of documents, s the average number of shingles per document, and T the total number of tokens in the corpus.

| Operation | Time | Space |
|---|---|---|
| Build the index (shingling and counting) | O(T) | O(N · s) for the shingle sets, plus the word counts |
| `jaccard(a, b)` | O(min(\|A\|, \|B\|)) on average (Python's `a & b` loops over the smaller set) | O(1) extra |
| `query` | O(N · s) | O(N) for the scored list |
| `all_pairs` | O(N² · s), exactly N(N-1)/2 comparisons | O(number of pairs above the threshold) |

Memory note: each shingle is a Python integer inside a set, which costs tens of bytes, so a 100,000-word book takes a few megabytes. The shingle sets of the whole corpus are kept in memory, which is fine for dozens or low hundreds of books.

---

## 9. Known limitations

1. **Accented and non-English letters.** The token pattern only accepts `a-z`, digits, and apostrophes. A word like "café" is split at the accent. This is fine for English books.
2. **Hyphenated words** such as "state-of-the-art" are split into separate words.
3. **Windows line endings.** The de-hyphenation step matches `-\n` only, not `-\r\n`. Text from the extraction script uses `\n`, so this normally does not matter.
4. **Cosine norms are recomputed** on every comparison. Caching them would speed up cosine, but the primary measure (Jaccard) is unaffected.
5. **Hash collisions.** Different shingles could share a 64-bit hash, but the probability is negligible at this corpus size.

---

## 10. How to run

```bash
python brute_force.py            # runs the demo
```

```python
from brute_force import BruteForceIndex, evaluate

idx = BruteForceIndex("extracted/manifest.json", shingle_k=5)

text = open("uploaded_book.txt", encoding="utf-8").read()
print(idx.query(text, measure="jaccard", top_n=5))
print(idx.stats())

truth = idx.ground_truth_pairs(threshold=0.8)       # for LSH recall
# later: evaluate(lsh_pairs, truth)
```
