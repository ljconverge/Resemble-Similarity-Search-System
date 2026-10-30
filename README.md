# Resemble: Similarity Search & Bucketing System

An end-to-end near-duplicate text detection and document similarity search system combining:
1. **Locality-Sensitive Hashing (LSH) & MinHash** with Carter-Wegman universal hashing for fast sub-quadratic candidate pruning.
2. **Deterministic SHA-256 Modulo Bucketing (4 Buckets)** for balanced record partitioning and aggregate analytics.
3. **Single-Page Application (SPA) & REST API** built with Flask and vanilla JavaScript for real-time, reload-free query retrieval and pairwise verification.

---

## Folder Layout

```text
Resemble-Similarity-Search-System/
├── app.py                      Flask application factory & REST API routes (/api/*)
├── requirements.txt            Project dependencies
├── .gitignore                  Git exclusions for Python caches, venv, and generated reports
├── Algorithm/
│   ├── lsh_engine.py           Core algorithm: text cleaning, shingling, MinHash (Carter-Wegman & notebook),
│   │                           LSH banding, verification, brute force, evaluation, and interactive LSHEngine
│   └── preprocessing.ipynb     Original reference notebook
├── services/
│   ├── __init__.py
│   └── bucketing.py            Deterministic SHA-256 4-bucket modulo partitioning & metric aggregation
├── Document/                   Bundled sample PDFs (Doc1.pdf to Doc10.pdf)
├── templates/
│   └── index.html              Unified Single-Page Application (SPA) console (tabs: Unified, 4-Bucket, Search, Performance)
├── static/
│   └── style.css               Optional external stylesheet
├── tools/                      Stand-alone brute-force analysis & extraction utilities
│   ├── brute_force_k1.py       Brute-force report for 1-word shingles
│   ├── brute_force_k3.py       Brute-force report for 3-word shingles
│   ├── brute_force_k9.py       Brute-force report for 9-word shingles
│   ├── compare_k1_k3.py        k=1 vs k=3 comparative report
│   ├── brute_force.py          General brute-force index
│   ├── extract_books.py        Quality-checked PDF extraction with multi-backend & OCR fallback
│   ├── document_loader.py      Multi-format loader (.pdf, .docx, .txt) preserving extensions
│   └── text_quality.py         Quality classifier for extracted text
└── tests/                      Automated test suites
    ├── conftest.py             Test path configurations
    ├── test_lsh_engine.py      MinHash, Carter-Wegman prime bound, LSH banding, and Jaccard tests
    ├── test_bucketing_and_api.py SHA-256 bucketing service & Flask endpoint tests
    ├── test_patches.py         Regression and edge-case boundary tests
    └── test_tools_and_app.py   SPA and tool verification tests
```

---

## Quickstart & Commands

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# macOS / Linux:
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser. The SPA console allows:
- Running pairwise collection analysis on bundled PDFs or custom uploads.
- Querying the indexed corpus asynchronously via `/api/search`.
- Inspecting uniform record distribution across 4 modulo buckets with dynamic role/text filtering.

### 3. Run Automated Tests
```bash
python -m pytest -q
```

### 4. CLI Execution
```bash
# Pairwise analysis on Document folder (3-word shingles, threshold 0.8, 20 bands):
python Algorithm/lsh_engine.py Document --mode word --k 3 --threshold 0.8 --bands 20

# Run character shingles with notebook MinHash method:
python Algorithm/lsh_engine.py Document --mode char --k 9 --minhash notebook

# Run standalone brute-force report:
cd tools && python brute_force_k3.py ../Document
```

---

## System Architecture & Hashing Mechanics

### 1. Carter-Wegman MinHash Universal Hashing
For string shingle IDs $x \in [0, 2^{32}-1]$:
$$h_i(x) = (a_i \cdot x + b_i) \pmod p$$
- **Prime modulus:** $p = 2^{61} - 1$ (Mersenne prime $M_{61}$). Because $p \gg 2^{32}$, collisions prior to hash evaluation are mathematically eliminated.
- **MinHash theorem:** $\Pr[\min h(A) = \min h(B)] = J(A, B)$.
- Vectorized over shingle chunks with NumPy object arrays to avoid 64-bit integer overflow.

### 2. LSH Banded Bucketing
- The $m=100$ MinHash signature is split into $b$ bands of $r$ rows ($b \times r = 100$, e.g. $20 \times 5$).
- Each band forms a bucket key `(band_index, tuple_of_rows)`.
- Colliding documents become candidate pairs with collision probability:
  $$P(\text{collision}) = 1 - (1 - s^r)^b$$
- Exact Jaccard is only calculated for candidate pairs, eliminating up to $95\%+$ of non-matching comparisons.

### 3. SHA-256 Modulo 4-Bucket Partitioning
- Computes SHA-256 on composite key `"{id}:{name}"`.
- Assigns bucket via `int(full_hash, 16) % 4` for uniform distribution.
- Tracks truncated digests (`#hash[:12]`) and aggregate value summaries across roles and categories.

---

## Settings Reference

| Setting | Options | Description |
|---|---|---|
| **Threshold** | 0–100% | Applied with exact Jaccard verification after candidate generation |
| **Bands** | 10, 20, 25, 50, 100 | Partitions the 100 hashes ($r = 100 / \text{bands}$) |
| **Shingles** | `word` ($k=1\text{--}20$) or `char` ($k=1\text{--}30$) | Default is 3 words (phrases); character shingles capture sub-word overlap |
| **MinHash** | `fast` or `notebook` | `fast` uses vectorized Carter-Wegman universal hashing; `notebook` uses seeded `mmh3` |

---

## Performance Metrics Explained

- **LSH time:** Shingle hashing + MinHash generation + banding + verification of candidate pairs.
- **Brute-force time:** Exact Jaccard calculated across all $\frac{n(n-1)}{2}$ document pairs.
- **Recall:** $\frac{\text{Pairs found by LSH}}{\text{True pairs found by Brute Force}}$.
- **Speedup:** $\frac{\text{Brute-Force Time}}{\text{LSH Time}}$. As the document collection grows beyond small datasets, LSH scales near linearly ($O(n)$) compared to brute force's quadratic growth ($O(n^2)$).
