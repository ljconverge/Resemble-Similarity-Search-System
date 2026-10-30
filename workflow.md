# Resemble Similarity Search: System Architecture & Workflow Pipeline

This document details the end-to-end thinking workflow, architectural design decisions, and exact mathematical formulations underpinning the Resemble Similarity Search System.

---

## 1. High-Level System Workflow

```
                                      [ Document Ingestion ]
                                  (PDF / TXT via PyMuPDF / UTF-8)
                                                │
                                                ▼
                                    [ Unicode Normalization ]
                              (NFKC, Ligature Fix, Punctuation Strip)
                                                │
                                                ▼
                                    [ k-Shingle Generation ]
                                 (Word k=1..20, Char k=1..30)
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │                                               │
                        ▼                                               ▼
         [ Standard LSH Pipeline ]                        [ Custom LSH Pipeline ]
     Carter-Wegman Universal Hashing                     Deterministic SHA-256 Digest
      p = 2^61 - 1, 100 Permutations                   int(SHA-256(content), 16) % 4
                        │                                               │
                        ▼                                               ▼
            [ LSH Banding Matrix ]                         [ 4 Modulo Buckets ]
           b bands × r rows = 100                        Within-Bucket Candidate Pairs
                        │                                               │
                        ▼                                               ▼
             [ Candidate Pairs ]                             [ Candidate Pairs ]
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
                                   [ Exact Jaccard Verification ]
                                     J(A, B) = |A ∩ B| / |A ∪ B|
                                                │
                                                ▼
                                   [ Ground-Truth Brute Force ]
                                    Exhaustive N(N-1)/2 Checks
                                                │
                                                ▼
                                  [ 3-Way Comparative Metrics ]
                         Recall, Precision, F1, Pruning %, Latency, Speedup
```

---

## 2. Step-by-Step Pipeline Execution

### Stage 1: Document Ingestion & Text Extraction
1. **Multi-Format Input:** The engine ingests `.pdf` and `.txt` files.
2. **Robust Multi-Backend Fallback:** 
   * PDF text extraction utilizes high-performance C-bindings via PyMuPDF (`fitz`).
   * If PyMuPDF is unavailable in the execution environment, the pipeline falls back to pure-Python `pypdf`.
   * Text files are decoded via UTF-8 with automatic character replacement on corrupt bytes.
3. **Display Name Sanitization:** Filenames undergo path-traversal sanitation (`re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', ...)`) while preserving Unicode scripts (e.g., Japanese, Chinese, Devanagari) and enforcing uniqueness.

---

### Stage 2: Text Preprocessing & Unicode Normalization
Before shingling, text is normalized to eliminate formatting artifacts that would distort token equality:
1. **NFKC Normalization:** Converts typographic ligatures (e.g., `ﬁ` $\to$ `fi`, `ﬂ` $\to$ `fl`) into standard component characters.
2. **Hyphenated Line Breaks:** Strips hyphenated line breaks (`exam-\nple` $\to$ `example`) commonly introduced by PDF column justification.
3. **Punctuation & Symbol Removal:** Removes both ASCII punctuation and Unicode punctuation (categories `P*` and `S*`, such as smart quotes `“` `”` and em-dashes `—`).
4. **Whitespace Collapsing:** Converts all runs of whitespace into a single space and casts characters to lowercase.

---

### Stage 3: Shingle Generation
A document is converted from an ordered sequence of words into an unordered set of $k$-shingles (token $n$-grams):
* **Word Shingles ($k \in [1, 20]$):**
  $$\text{Shingles}_{\text{word}}(D) = \{ (w_i, w_{i+1}, \dots, w_{i+k-1}) \mid 1 \le i \le |D| - k + 1 \}$$
* **Character Shingles ($k \in [1, 30]$):**
  $$\text{Shingles}_{\text{char}}(D) = \{ c[i : i+k] \mid 0 \le i \le |c| - k \}$$
* **Short-Text Safeguard:** Documents with fewer than $k$ tokens return a single shingle containing all tokens, preventing empty shingle sets.

---

### Stage 4: Carter-Wegman Universal MinHash Hashing
To approximate the Jaccard similarity without storing large shingle sets, each document is compressed into a signature vector of length $n = 100$:

1. **Integer Hashing:** String shingles are hashed into 32-bit unsigned integers via MurmurHash3 (or SHA-256 fallback):
   $$x \in [0, 2^{32} - 1]$$
2. **Carter-Wegman Universal Family:** To eliminate collision aliasing, the engine applies universal linear congruential hash functions over the 9th Mersenne prime $M_{61} = 2^{61} - 1$:
   $$h_i(x) = (a_i \cdot x + b_i) \bmod (2^{61} - 1)$$
   where $a_i \in [1, M_{61} - 1]$ and $b_i \in [0, M_{61} - 1]$ are randomly chosen constants seeded deterministically.
3. **MinHash Computation:**
   $$s(D)_i = \min_{x \in \text{shingle\_ids}(D)} h_i(x), \quad i \in \{1, 2, \dots, n\}$$
4. **Vectorized Chunking:** Implemented in NumPy using broadcasting across chunks of size 512 (`_FAST_CHUNK`) to avoid large intermediate matrix allocations:
   $$\text{hashed} = (A_{\text{col}} \cdot X_{\text{chunk}} + B_{\text{col}}) \bmod M_{61}$$

**Mathematical Property:**
The probability that two documents share the same minimum hash value is exactly equal to their Jaccard similarity:
$$P(h(D_1) = h(D_2)) = J(D_1, D_2)$$

---

### Stage 5: LSH Banding Matrix Partitioning
To find candidate pairs in sub-quadratic time without pairwise comparisons, signatures are banded:
1. The $n = 100$ dimensional signature is divided into $b$ bands of $r$ rows each ($b \times r = n$).
2. For each band $j \in \{0, \dots, b-1\}$, the sub-vector of length $r$ is mapped to a hash bucket:
   $$\text{key}_j = (j, s_j[0], s_j[1], \dots, s_j[r-1])$$
3. **Candidate Generation:** If any two documents collide in at least one band bucket, they become a **candidate pair**.
4. **S-Curve Collision Probability:**
   $$P(\text{collision} \mid s) = 1 - (1 - s^r)^b$$
   where $s = J(D_1, D_2)$. The inflection threshold is approximately:
   $$\tau \approx \left(\frac{1}{b}\right)^{1/r}$$

---

### Stage 6: Custom LSH Pipeline (Deterministic SHA-256 Modulo 4)
As an explicit algorithmic comparison, the engine also implements deterministic SHA-256 modulo partitioning:
1. For each document $D_i$ with preprocessed text $T_i$:
   $$\text{digest} = \text{SHA-256}(T_i)$$
   $$\text{hash\_int} = \text{int}(\text{digest}, 16)$$
   $$\text{bucket\_id} = \text{hash\_int} \bmod 4$$
2. **Candidate Rule:** Documents assigned to the exact same bucket ($0, 1, 2, \text{ or } 3$) form candidate pairs.
3. **Theoretical Contrast:** Because SHA-256 is cryptographic, it exhibits the **avalanche effect**. Changing a single word completely scrambles the hash integer, dispersing similar documents uniformly across the 4 buckets ($\sim 25\%$ collision probability). Identical documents always collide ($100\%$), but non-identical near-duplicates suffer high false negative rates.

---

### Stage 7: Exact Jaccard Verification & Brute-Force Baseline
1. **Candidate Verification:**
   For each candidate pair $(D_a, D_b)$ produced by either Standard LSH or Custom LSH, exact Jaccard similarity is verified:
   $$J(D_a, D_b) = \frac{|\text{Shingles}(D_a) \cap \text{Shingles}(D_b)|}{|\text{Shingles}(D_a) \cup \text{Shingles}(D_b)|}$$
   Pairs with $J(D_a, D_b) \ge \text{threshold}$ are classified as **Near Duplicates**.
2. **Brute-Force Baseline ($N^2$):**
   Computes exact Jaccard across all $\frac{N(N-1)}{2}$ combinations without filtering.

---

### Stage 8: Comparative Multi-Metric Evaluation
Both algorithms are evaluated against the Brute-Force ground truth:

* **Empirical Recall:**
  $$\text{Recall} = \frac{|\text{Verified Candidates} \cap \text{True Pairs}|}{|\text{True Pairs}|}$$
  *(Safely defined as $1.0$ when $|\text{True Pairs}| = 0$ and $|\text{Verified Candidates}| = 0$).*

* **Empirical Precision:**
  $$\text{Precision} = \frac{|\text{Verified Candidates} \cap \text{True Pairs}|}{|\text{Verified Candidates}|}$$

* **F1 Score:**
  $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

* **Candidate Reduction (Pruning %):**
  $$\text{Pruned} = \left(1 - \frac{|\text{Candidates}|}{N(N-1)/2}\right) \times 100\%$$

* **Speedup Factor:**
  $$\text{Speedup} = \frac{\text{Brute Force Time (ms)}}{\text{Algorithm Time (ms)}}$$

---

## 3. Architectural Design Decisions

1. **Separation of Concerns (Two-Page Architecture):**
   * `index.html`: Optimized for fast upload, configuration, and top-level KPIs. Contains only one primary button (`"Analyze uploaded files"`).
   * `detailed.html`: Dedicated deep-dive console for the 3-Way Comparison Table, Pairwise Jaccard scores, and the 4-Bucket Grid.
2. **Numerical Resilience:**
   * No `NaN`, `inf`, or `0 / 0` division errors.
   * If zero pairs meet the cutoff threshold, the engine automatically extracts the top 5 closest pairs to ensure similarity percentages never disappear.
3. **Stateless Scalability:**
   * Analysis results are passed through request contexts and stored in application state, allowing interactive AJAX querying against the live indexed corpus.
