# Debug log: merging brute_force.zip and LSH_Algo.zip

| # | Problem found | Where | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: document_loader` (crash) | brute_force_k1/k3/k9.py, compare_k1_k3.py | Wrote `tools/document_loader.py`. Its k=3 report matches the original `report_k3.txt` in every shingle count and Jaccard score |
| 2 | `ModuleNotFoundError: text_quality` (crash) | extract_books.py | Wrote `tools/text_quality.py` (EMPTY / TOO_SHORT / GARBLED / UNRECOGNISED / WRONG_LANGUAGE / OK) |
| 3 | LSH used 9-character shingles; brute force used 1/3/9-word or 5-word hashed shingles, with different cleaning, so recall was not comparable | whole project | One `lsh_engine.py`: both engines use the same `preprocess_text` + `make_shingles` (word or char, chosen in the UI) |
| 4 | Notebook MinHash took 2.6 s for 10 PDFs (about 5.4 million mmh3 calls in Python) | preprocessing_runtime.py | Kept it as `minhash_method="notebook"`; added `"fast"` (universal hashing + NumPy), about 25× faster. A test proves the notebook path gives the original output |
| 5 | Run time included PDF reading, so the LSH time was not comparable | app.py | Times stages separately: extraction, shared preprocessing, LSH (MinHash + banding + verify), brute force |
| 6 | No brute force, recall or speedup in the app | app.py, performance.html | Every run now does brute force on the same shingles; the performance page shows recall, precision, F1, speedup and missed pairs |
| 7 | `import fitz` deprecation warning (it will break in a future PyMuPDF) | extract_books.py | Imports `pymupdf` first, falls back to `fitz` |
| 8 | "45-pair rankings" was fixed text | compare_k1_k3.py | Uses the real number of pairs |
| 9 | Uploading with no file said "No readable PDF text" | app.py | Now says "Please choose at least one PDF or TXT file" |
| 10 | Non-numeric threshold, band or k crashed into a generic error | app.py | Clear validation messages |
| 11 | Only PDFs could be uploaded | app.py | .txt uploads accepted too |

**Known difference:** your original report has slightly lower word counts for Doc5–Doc10 (for example 881 vs 889). That depends on which PDF reader was used. Shingle sets and all Jaccard scores are identical.
