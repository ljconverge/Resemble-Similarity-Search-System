# LSH Similarity Search: merged project

LSH and brute force now share one algorithm file, `Algorithm/lsh_engine.py`. Both use the same cleaning and the same shingles, so the recall and speedup figures are a fair comparison.

## Folder layout

```text
LSH_Project/
├── app.py                    Flask web app (calls lsh_engine)
├── requirements.txt
├── Algorithm/
│   ├── lsh_engine.py         THE algorithm file: preprocessing, shingles, MinHash (fast + notebook),
│   │                         LSH banding, verification, brute force, evaluation, command line
│   └── preprocessing.ipynb   original notebook (unchanged, for reference)
├── Document/                 10 sample PDFs
├── templates/  static/       web pages and CSS
├── tools/                    stand-alone brute-force analysis scripts
│   ├── brute_force_k1.py / _k3.py / _k9.py   reports for 1-, 3-, 9-word shingles
│   ├── compare_k1_k3.py      k=1 vs k=3 comparison report
│   ├── brute_force.py        general brute-force index (query, cosine, containment)
│   ├── extract_books.py      quality-checked PDF extraction with OCR fallback
│   ├── document_loader.py    NEW (was missing): reads .pdf/.docx/.txt for the scripts above
│   └── text_quality.py       NEW (was missing): detects empty/garbled/wrong-language text
└── tests/                    65 automated tests
```

`Algorithm/lsh.py` and `Algorithm/preprocessing_runtime.py` were merged into `lsh_engine.py` and removed.

## Run

```bash
python -m pip install -r requirements.txt
python app.py                                     # web app at http://127.0.0.1:5000
python -m pytest -q                               # all tests
python Algorithm/lsh_engine.py Document --mode word --k 3           # command line
python Algorithm/lsh_engine.py Document --mode char --k 9 --minhash notebook
cd tools && python brute_force_k3.py ../Document  # brute-force report
```

## Settings in the web app

| Setting | Options | Notes |
|---|---|---|
| Threshold | 0–100% | Applied with exact Jaccard after LSH |
| Bands | 10, 20, 25, 50, 100 | 100 MinHash values, so rows = 100 / bands |
| Shingles | word (k = 1–20) or char (k = 1–30) | Default is 3 words; 9 characters was the old notebook setting |
| MinHash | fast or notebook | Same idea. Fast uses (a·x + b) mod (2³¹−1) with NumPy; notebook uses mmh3 with seed i and is about 25× slower |

## What the performance page measures

- **LSH time** = MinHash + banding + verification.
- **Brute-force time** = Jaccard on all n(n−1)/2 pairs.
- Reading the files and cleaning/shingling are shared by both, so they're shown separately and not counted in either time.
- **Recall** = pairs LSH found ÷ pairs brute force found.
- With only 10 documents, brute force is faster (45 comparisons). LSH pays off as the collection grows, because brute-force work grows with n².
