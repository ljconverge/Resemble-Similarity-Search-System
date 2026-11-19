# Resemble Similarity Search System

**Live demo:** [Open the Streamlit dashboard](https://resemble-similarity-search-system.streamlit.app/)

Resemble is a Streamlit application for finding near-duplicate text documents. It compares an exhaustive brute-force baseline with a search pipeline that uses ProbMinHash and locality-sensitive hashing (LSH). The dashboard shows which matches LSH retrieved, which it missed, and how the two methods compare at a selected similarity threshold.

## Quick start on Windows

1. **Extract the project ZIP.** In PyCharm, open the extracted folder containing `streamlit_app.py` and `requirements.txt`. Open the PyCharm terminal or a PowerShell window in that folder.
2. **Check that Python 3.13 is installed.** The original project environment used this version.

   ```powershell
   py -3.13 --version
   ```

   If the Python launcher cannot find this version, install Python 3.13 and reopen the terminal. If your installation exposes Python 3.13 as `python`, you can use that command in the next step.

3. **Create a virtual environment.**

   ```powershell
   py -3.13 -m venv .venv
   ```

4. **Install the project dependencies into that environment.**

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   Streamlit provides the dashboard, pandas formats result tables, NumPy supports signature computation, and pypdf and python-docx extract text from PDF and Word files.

5. **Start Streamlit.**

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
   ```

6. **Open the Local URL** printed in the terminal, usually `http://localhost:8501`. Keep the terminal open while using the dashboard. Press **Ctrl+C** to stop it. For later launches, repeat step 5 from the project folder.

## Using the dashboard

| Mode | Workflow |
| --- | --- |
| Benchmark corpus | Choose an individual document to search the rest of the corpus, or choose **ALL** to evaluate all 4,950 unique pairs. Set the threshold and click **Compare documents**. Use **Show matches** to inspect all reference matches or only the matches LSH missed. |
| Upload documents | Select at least two readable `.txt`, `.pdf`, or `.docx` files. Set the threshold and click **Compare uploads** to evaluate pairs among those files. PDF extraction reads up to the first 50 pages. Image-only scans may have no extractable text. |

The bundled corpus contains 100 documents. Its dataset builder defines 30 source texts, 35 synthetic documents with Zipfian term frequencies, and 35 perturbed documents designed to introduce controlled overlap. The corpus and its manifest are stored in `data/`.

## How a comparison runs

1. **Load the text.** The app reads the bundled corpus or extracts text from the uploaded files. It builds and caches the benchmark indexes for reuse across comparisons.
2. **Build term frequencies.** Preprocessing cleans and tokenizes the text, then converts each document into a weighted term distribution. Probability Jaccard similarity measures overlap between these distributions.
3. **Compute the reference matches.** Brute force scores each eligible pair directly, or compares one selected document against the rest of the corpus in query mode. Pairs meeting the selected threshold form the reference match set.
4. **Index and retrieve candidates.** ProbMinHash creates a 100-value signature for each document. LSH splits signatures into bands and groups matching bands into buckets. A shared bucket in at least one band makes a document or pair eligible for verification. The app selects 50 bands below 35%, 25 bands from 35% up to 60%, and 20 bands at 60% or above.
5. **Verify the candidates.** The LSH pipeline computes exact Probability Jaccard similarity for the candidates it retrieves and applies the same threshold. It also reports the signature-based estimate. A qualifying pair can be missed if its signatures share no band.
6. **Evaluate retrieval.** The app compares the LSH matches against the brute-force reference and displays timing, recall, and match status.

## Reading the results

- **Exact Similarity (BF)** is the Probability Jaccard score computed from term frequencies. **LSH Estimated Similarity** is based on the fraction of matching signature entries.
- **Accuracy** in the dashboard is match recall. It measures the share of brute-force reference matches found by LSH.
- **Missed by LSH** identifies reference matches absent from the LSH result. A pruned row has no displayed LSH estimate.
- **Timing and speedup** are dashboard summaries. The LSH timing covers search after indexing.

## Repository guide

| Path | Responsibility |
| --- | --- |
| `streamlit_app.py` | Dashboard, document extraction, and benchmark orchestration |
| `Algorithm/preprocessing.py` | Text cleaning, term frequencies, and Probability Jaccard similarity |
| `Algorithm/brute_force.py` | Exhaustive all-pairs search and single-document queries |
| `Algorithm/probminhash.py` | Weighted signatures and similarity estimation |
| `Algorithm/lsh.py` | Band indexing, candidate retrieval, and verification |
| `Algorithm/evaluation.py` | Match evaluation and speedup calculations |
| `services/dataset_service.py` | Corpus generation and loading |
| `data/` | Bundled documents and dataset manifest |
| `References/` | ProbMinHash research paper |
| `tests/` | Unit tests for the corpus and search pipeline |

## Run the tests

After installing dependencies, run the test suite from the same project folder.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

If the launch command cannot find `streamlit_app.py` or pip cannot find `requirements.txt`, check that your terminal is inside the extracted project folder. If uploads produce too little readable text, try plain-text files or PDFs with selectable text.

## Reference

The search pipeline is based on Otmar Ertl’s [*ProbMinHash: A Class of Locality-Sensitive Hash Algorithms for the Probability Jaccard Similarity*](References/ProbMinHash__A_Class_of_Locality-Sensitive_Hash_Algorithms_for_the_Probability_Jaccard_Similarity.pdf). The paper is included in the repository.
