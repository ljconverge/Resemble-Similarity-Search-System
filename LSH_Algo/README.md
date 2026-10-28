# LSH Similarity Search — dynamic PDF UI

This version connects the Flask UI to the project's from-scratch LSH implementation. It removes the hard-coded demo result values and the standalone `test_lsh.py` file.

## Run locally

1. Extract this folder.
2. Open a terminal in the `LSH_UI_updated` folder.
3. (Recommended) Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the web app:

   ```bash
   python app.py
   ```

6. Open the local URL printed by Flask (usually `http://127.0.0.1:5000`).
7. Upload PDFs or click **Analyze bundled sample PDFs** to process the PDFs in `Document/`.

## What is connected

- PDF text extraction using `pypdf`.
- Existing notebook text cleaning logic.
- Existing 9-character shingling and MurmurHash3 hashing.
- Existing 100-value MinHash signature approach (matrix columns are documents).
- LSH banding and candidate generation from `Algorithm/lsh.py`.
- Exact Jaccard scoring of LSH candidates only.
- Dynamic results and per-run performance metrics in the UI.

## Important scope notes

- `Algorithm/preprocessing.ipynb` is preserved unchanged.
- `Algorithm/preprocessing_runtime.py` makes the notebook's preprocessing/MinHash functions importable by Flask. The text cleaning, shingle size, seed, and signature-generation approach are kept consistent with the notebook. It also provides original character shingles for exact Jaccard verification.
- `Algorithm/lsh.py` is implemented from scratch; no ready-made LSH library is used.
- `test_lsh.py` and hard-coded demo results are not included in this folder.
- Brute-force benchmarking is intentionally not implemented yet. The performance page clearly distinguishes the number of possible pairs from pairwise comparisons actually performed.
- The app uses an in-memory cache for the latest analyses. Restarting Flask clears results. Uploaded files are processed in memory and are not saved to disk by the app.
- Only PDFs with selectable/extractable text work directly. Scanned/image-only PDFs need OCR first.
- The `Document/` folder contains the provided sample PDFs. The sample button processes those files; user-uploaded PDFs are processed separately.

## LSH controls

The current MinHash signature length remains 100. The UI allows 10, 20, 25, 50, or 100 bands so the signature can be split evenly. With 20 bands, each band contains 5 signature values. The similarity threshold (0–100%) is applied after candidate generation using exact Jaccard similarity. The main results table includes only candidate pairs meeting the threshold; rejected candidates appear in a separate, collapsed diagnostic section. A threshold of 0% still only includes pairs that LSH generated as candidates, not every possible pair in the collection.

LSH is probabilistic: a similar pair can fail to become a candidate. A later brute-force baseline can measure recall and compare runtime using the same shingle sets.
