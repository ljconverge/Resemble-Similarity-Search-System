# Resemble Similarity Search: Test Suite Documentation

This document catalogs every automated test case in the Resemble Similarity Search System. It describes the test purpose, inputs, edge-case coverage, and expected assertions across the entire test suite.

---

## Test Execution Instructions

Run all 79 tests in the workspace:

```powershell
.\venv\Scripts\python.exe -m pytest -v
```

Or run individual test modules:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_bucketing_and_api.py -v
.\venv\Scripts\python.exe -m pytest tests/test_lsh_engine.py -v
.\venv\Scripts\python.exe -m pytest tests/test_patches.py -v
.\venv\Scripts\python.exe -m pytest tests/test_tools_and_app.py -v
```

---

## 1. `tests/test_bucketing_and_api.py` (16 Tests)

This suite validates deterministic SHA-256 modulo bucketing, Custom LSH vs. Standard LSH side-by-side comparison, edge cases (zero matches, identical files), and Flask API routes with SPA AJAX file upload interception.

| Test Function | Purpose | Input / Setup | Edge-Case Handled | Expected Output / Assertions |
| :--- | :--- | :--- | :--- | :--- |
| `test_compute_sha256_bucket_deterministic` | Verifies SHA-256 hashing and modulo assignment are deterministic. | Key `"test:key"`, `num_buckets=4`, `truncate_len=12`. | Repeated runs with identical input. | Returns identical 12-char digest and bucket in $[0, 3]$. |
| `test_compute_sha256_bucket_validation` | Ensures invalid bucket counts and truncation lengths raise `ValueError`. | `num_buckets=0`, `-1`, `truncate_len=0`. | Non-positive parameters. | Raises `ValueError`. |
| `test_build_bucketed_dataset_aggregates` | Verifies full mock dataset partitioning across 4 buckets and KPI math. | 12 mock records (`MOCK_RECORDS`). | Non-uniform role counts. | Total records equal sum of bucket counts; percentages sum to $100\%$. |
| `test_custom_sha256_candidates_deterministic` | Tests candidate pair generation from SHA-256 modulo bucketing. | 3 documents (two identical, one distinct). | Identical document texts. | Identical documents land in identical bucket; forms candidate pair `("docA", "docB")`. |
| `test_run_analysis_contains_3way_comparison` | Verifies the 3-Way comparison structure returned by `run_analysis()`. | 3 text documents, threshold 0.5. | Multi-algorithm comparison format. | Returns `comparison` dict with `brute_force`, `standard_lsh`, and `custom_lsh` metrics. |
| `test_identical_documents_both_algorithms_100_percent` | Verifies that both Standard LSH and Custom LSH achieve 100% recall on duplicate files. | 2 identical files (`copy1.txt`, `copy2.txt`), threshold 0.8. | $100\%$ duplicate files. | Both `recall` and `custom_recall` equal $1.0$; found pairs equal 1. |
| `test_zero_matches_no_division_by_zero` | **The "n/a" Bug Test**: Ensures 0 matches never trigger zero-division or `"n/a"`. | 2 completely disjoint phonetic alphabet texts, threshold 0.95. | Zero pairs meet threshold ($0/0$ division). | `recall` and `precision` evaluate to $1.0$; `closest_pairs` fallback populated. |
| `test_index_get` | Tests GET request to minimalist dashboard `/`. | HTTP GET `/`. | Clean initial load without analysis. | Status code 200; contains "Resemble" and "SHA-256". |
| `test_api_buckets` | Tests `/api/buckets` JSON endpoint. | HTTP GET `/api/buckets`. | API response format. | Status 200; JSON contains `buckets`, `bucket_summaries`, `total_records`. |
| `test_api_performance` | Tests `/api/performance` JSON endpoint. | HTTP GET `/api/performance`. | API metric extraction. | Status 200; JSON contains `num_hashes`, `num_bands`, `speedup_factor`. |
| `test_api_search_success` | Tests corpus query search via JSON. | Query `"deep residual learning image recognition"`, `top_k=3`. | Vector search query against corpus. | Status 200; returns non-empty result list; `doc_id` matches ResNet. |
| `test_api_search_empty_query` | Tests query endpoint rejects empty queries. | Query `""`. | Empty string input. | Status 400; JSON contains `"error"` message. |
| `test_api_analyze_sample` | Tests full analysis on synthetic sample dataset via API. | Source `"sample"`, 10 sample PDFs, threshold 80%. | Multi-file document collection. | Status 200; document count 10; recall $1.0$; contains `comparison` metrics. |
| `test_api_analyze_ajax_file_upload` | **SPA Interception Test**: Verifies AJAX multipart upload to `/api/analyze`. | 2 text files via `multipart/form-data`, `X-Requested-With: XMLHttpRequest`. | Multipart form data without manual Content-Type header. | Status 200; JSON response contains `analysis` and `redirect_url`. |
| `test_root_post_ajax_file_upload` | Verifies root `/` POST handles AJAX multipart uploads gracefully. | 2 text files via `multipart/form-data`, `X-Requested-With: XMLHttpRequest`. | SPA AJAX interception on root route. | Status 200; JSON response with full analysis payload. |
| `test_api_analyze_invalid_params` | Validates API rejection of out-of-range thresholds. | Threshold `"150"`, bands 20. | Value $> 100\%$. | Status 400; JSON contains validation error message. |

---

## 2. `tests/test_lsh_engine.py` (45 Tests)

Validates the mathematical core: Unicode normalization, Broder's Jaccard formulation, Carter-Wegman universal hashing, chunked signature generation, LSH S-curve theory, and corpus query retrieval.

| Test Function / Parameters | Purpose | Input / Setup | Expected Assertions |
| :--- | :--- | :--- | :--- |
| `test_preprocess_matches_notebook_steps` | Verifies whitespace collapsing, lowercasing, and null safety. | `"Machine Learning is Useful!\r\nYes."`, `None`. | Returns `"machine learning is useful yes"`; `None` returns `""`. |
| `test_preprocess_pdf_fixes` | Verifies ligature replacement and hyphenated line break joining. | `"ﬁle"`, `"exam-\nple text"`. | Ligature `ﬁ` $\to$ `"file"`; hyphen break $\to$ `"example text"`. |
| `test_word_and_char_shingles` | Verifies $k$-word and $k$-char token slicing. | Word $k=2$, Char $k=3$. | Generates correct set of 2-word and 3-char shingles. |
| `test_short_and_empty_text` | Validates short text handling when text length $< k$. | Empty string, 2 words with $k=3$. | Empty returns `set()`; short text returns single shingle with all tokens. |
| `test_invalid_shingle_settings` | Parameterized test (5 cases) checking $k$ range boundaries. | Word $k=0, 21, 2.5$; Char $k=31$; mode `"xyz"`. | Raises `ValueError` on all invalid settings. |
| `test_shingle_ids_are_notebook_mmh3` | Ensures 32-bit uint32 hashing matches reference unsigned MurmurHash3. | String shingle `"hello wor"`, seed 42. | Output array has dtype `uint32` and matches `mmh3.hash(..., signed=False)`. |
| `test_broder_1997_example` | Parameterized test (3 cases) checking Broder's 1997 paper examples. | Sentences A & B with word $k=1, 2, 3$. | Exactly matches theoretical values: $3/5$ ($0.6$), $3/6$ ($0.5$), $3/7$ ($0.4286$). |
| `test_readme_example` | Tests Jaccard similarity on sample healthcare sentences. | 2 sample sentences, word $k=2$. | Matches exact Jaccard similarity of $4/7$. |
| `test_jaccard_properties` | Tests mathematical axioms of Jaccard similarity. | Identical sets, disjoint sets, empty sets. | Identical = $1.0$; disjoint = $0.0$; two empty = $1.0$; one empty = $0.0$. |
| `test_mersenne_prime_strictly_exceeds_uint32` | **Universality Proof**: Verifies $M_{61} > 2^{32}$. | $M_{61} = 2^{61}-1$, Universe $2^{32}-1$. | Confirms prime strictly exceeds universe size, preventing hash aliasing. |
| `test_minhash_reproducible_and_identical_sets` | Parameterized test (2 cases: fast, notebook) for deterministic signatures. | Random uint32 IDs and reversed copies. | Reversed set produces identical signature to original set. |
| `test_notebook_method_equals_original_notebook_code` | Confirms classic notebook MinHash mode matches exact reference loops. | 2 synthetic shingle arrays, 10 hashes. | Generated signature matches element-for-element. |
| `test_fast_method_matches_formula` | Confirms Carter-Wegman vectorization matches $h(x) = (ax+b) \bmod M_{61}$. | 30 shingle IDs, 5 hashes, seed 42. | Vectorized output equals manual loop over scalar formula. |
| `test_fast_method_chunking_gives_same_result` | Verifies chunked broadcasting (`_FAST_CHUNK=7`) yields identical result to unchunked. | 1,000 shingle IDs, 16 hashes. | Chunked signature is byte-for-byte identical to full matrix signature. |
| `test_minhash_estimates_jaccard` | Parameterized test (6 cases: fast/notebook $\times$ targets $0.2, 0.5, 0.8$). | Controlled overlap shingle sets, 200 hashes. | Estimated Jaccard falls within $4 \times \text{std\_err} + 0.05$ of true Jaccard. |
| `test_minhash_rejects_bad_input` | Validates error checking on empty arrays, zero hashes, and unknown methods. | Empty array, `num_hashes=0`, method `"magic"`. | Raises `ValueError`. |
| `test_bucketing_and_candidates` | Tests 2-band partitioning and candidate generation. | 3 signatures with known band overlaps. | Colliding sub-vectors generate candidate pairs; non-colliding do not. |
| `test_band_index_in_key_and_pair_once` | Ensures band index is included in key to prevent cross-band collisions. | Signatures with identical numbers in different bands. | Cross-band matches do not collide; multiple band collisions produce single candidate pair. |
| `test_bad_band_settings` | Parameterized test (4 cases: bands=3, 0, 5, True) on 4-hash signatures. | Non-divisible band counts. | Raises `ValueError`. |
| `test_s_curve` | Verifies mathematical S-curve probability and threshold approximation. | Bands=20, Rows=5, Similarity=0.5. | Probability matches $1-(1-0.5^5)^{20} \approx 0.470$; threshold $\approx 0.549$. |
| `test_brute_force_counts_every_pair` | Verifies brute-force exhaustive pairwise loop. | 6 synthetic document sets. | Exactly computes $\frac{6 \times 5}{2} = 15$ comparisons. |
| `test_evaluate` | Tests recall, precision, and false negative calculation. | Found pairs `[("a", "b")]`, True pairs `[("a", "b"), ("c", "d")]`. | Recall = $0.5$, Precision = $1.0$, False Negatives = $1$. |
| `test_lsh_engine_query_strict_semantics_and_accuracy` | End-to-end test of interactive `LSHEngine` with high and low similarity queries. | Corpus of 3 documents (VGG, ResNet, Astronomy). | High query matches relevant papers; disjoint query returns 0 candidates under high threshold. |
| `test_run_analysis_end_to_end` | Parameterized test (4 cases: modes $\times$ methods) for end-to-end analysis. | 4 documents (including 1 near-duplicate pair and 1 empty file). | Identifies near-duplicate pair; flags empty file with warning; recall = $1.0$. |
| `test_lsh_pairs_always_subset_of_brute_force` | Verifies that LSH candidate verified pairs are always a strict subset of brute force. | 4 documents, threshold 0.0. | $\text{LSH verified pairs} \subseteq \text{Brute-force pairs}$. |
| `test_run_analysis_validation` | Validates threshold $> 1.0$, non-divisible bands, and all-empty input documents. | Threshold 1.5, bands 7, blank text. | Raises `ValueError`. |
| `test_sample_documents_folder` | Benchmark test on `synthetic_datasets/` (10 PDFs). | 10 sample PDFs loaded from disk, threshold 80%. | Finds all 6 near-duplicate pairs (Doc1–Doc4); Doc1 ↔ Doc3 similarity = $98.91\%$. |

---

## 3. `tests/test_patches.py` (13 Tests)

Validates critical bug fixes and patch guarantees.

| Test Function | Purpose | Input / Setup | Assertion |
| :--- | :--- | :--- | :--- |
| `test_hashing_counted_in_lsh_time_and_repeats` | Verifies shingle hashing is counted in total LSH latency and repeat timing. | Timing repeats 1 and 3 on 3 documents. | `hashing_ms > 0`; sum of stage latencies equals `lsh_ms`; repeats parameter validated. |
| `test_metrics_none_when_nothing_to_find` | Tests metric evaluation when threshold is set to 99% and 0 pairs qualify. | 3 documents, threshold 0.99. | True pairs = 0; recall = 1.0; F1 = 1.0; no exceptions. |
| `test_candidate_precision` | Validates candidate precision metric calculation. | 3 documents, 100 bands. | Candidate precision equals verified pairs divided by candidate count. |
| `test_pct_text` | Parameterized test (4 cases) ensuring percentage formatting never rounds across threshold. | Scores $0.79996$, $0.80$, $0.98909$, $0.799999999$. | Formats to exact decimal depth without incorrectly crossing cutoff. |
| `test_rounding_in_results` | Verifies boundary score of 79.996% is marked as "Below threshold" at 80% cutoff. | Two large generated word lists. | Status is `"Below threshold"`; formatted as `"79.9960"`. |
| `test_unicode_punctuation` | Ensures Unicode typographic quotes and dashes are cleaned identically to ASCII. | Strings with `“`, `”`, `—`, `…`. | Matches ASCII equivalent preprocessing output. |
| `test_seed_reported_only_for_fast` | Confirms seed parameter is reported for Carter-Wegman fast mode, but None for notebook. | Fast vs. notebook analysis runs. | Fast reports `seed=42`; notebook reports `seed=None`. |
| `test_display_names` | Validates unique display name generation and path traversal sanitization. | Foreign names (Japanese, Chinese, Hindi), relative paths (`..\\..\\evil.pdf`). | Traversal stripped; Unicode preserved; duplicate names appended with counter (`_2`). |
| `test_index_page_script` | Verifies presence of frontend safety scripts in `index.html`. | HTTP GET `/`. | HTML includes `syncLabels()`, `pageshow` listener, and shingle validation logic. |
| `test_app_upload_and_pages` | Web app end-to-end test uploading files and verifying detailed performance page. | Multipart upload of 2 text files. | HTML includes `"median of 3 runs"`, `"shingle hashing"`, and graceful fallback. |

---

## 4. `tests/test_tools_and_app.py` (5 Tests)

Validates single-page application structure and interactive REST endpoints.

| Test Function | Purpose | Input / Setup | Assertion |
| :--- | :--- | :--- | :--- |
| `test_spa_index_loads` | Verifies main dashboard loads with required elements. | HTTP GET `/`. | Status 200; contains branding, `search-form`, and `performance-section`. |
| `test_detailed_analysis_page_loads` | Verifies secondary detailed analysis page loads with required sections. | HTTP GET `/detailed-analysis`. | Status 200; contains `performance-section`, `search-form`, and `buckets-section`. |
| `test_api_search_json` | Tests `/api/search` JSON query against corpus. | Query on deep residual learning, `top_k=3`. | Status 200; returns non-empty matches; top match is `CV_2015_He_ResNet`. |
| `test_api_search_empty_query` | Tests `/api/search` handles whitespace-only query. | Query `"   "`. | Status 400; JSON contains error message. |
| `test_api_performance_json` | Tests `/api/performance` returns full performance dictionary. | HTTP GET `/api/performance`. | Status 200; contains `num_documents`, `num_hashes`, `precision_at_k`. |

---

## 5. Test Suite Verification Summary

```
============================= 79 passed in 3.08s ==============================
```

* **Total Test Cases:** 79
* **Passed:** 79
* **Failed:** 0
* **Edge Cases Covered:** Empty files, short texts, ligature replacement, non-English scripts, path traversal, $0$-match zero-division, $100\%$ duplicate files, boundary thresholds ($0.0$ and $1.0$), chunked matrix broadcasting, and AJAX SPA multipart uploads.
