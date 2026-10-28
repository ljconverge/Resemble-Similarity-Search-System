"""Flask UI for dynamic PDF similarity search using the project's LSH implementation."""

from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import os
import sys
from itertools import combinations

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
ALGORITHM_DIR = BASE_DIR / "Algorithm"
DOCUMENT_DIR = BASE_DIR / "Document"
sys.path.insert(0, str(ALGORITHM_DIR))

from lsh import generate_candidates  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB per request
app.config["MAX_PDF_COUNT"] = 30

# Local-development cache: only a short analysis ID is kept in the browser session.
# Results disappear if the Flask process restarts; this avoids putting large result
# objects into the session cookie or saving uploaded PDFs to disk.
ANALYSIS_CACHE = {}
MAX_CACHED_ANALYSES = 10
NUM_HASHES = 100
SHINGLE_SIZE = 9


def _unique_display_name(filename, used_names):
    """Keep uploaded file labels unique if two files share the same filename."""
    safe_name = secure_filename(filename) or "document.pdf"
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix or ".pdf"
    candidate = f"{stem}{suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _extract_pdf(file_stream):
    """Extract selectable text from a PDF stream."""
    from pypdf import PdfReader

    reader = PdfReader(file_stream)
    page_texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            page_texts.append(text)
    return "\n".join(page_texts)


def _read_uploaded_documents(files):
    documents = {}
    warnings = []
    used_names = set()

    if len(files) > app.config["MAX_PDF_COUNT"]:
        raise ValueError(
            f"Please upload no more than {app.config['MAX_PDF_COUNT']} PDF files at once."
        )

    for uploaded_file in files:
        if not uploaded_file or not uploaded_file.filename:
            continue
        if Path(uploaded_file.filename).suffix.lower() != ".pdf":
            warnings.append(f"{uploaded_file.filename}: skipped because it is not a PDF.")
            continue

        display_name = _unique_display_name(uploaded_file.filename, used_names)
        try:
            uploaded_file.stream.seek(0)
            raw_text = _extract_pdf(uploaded_file.stream)
            if raw_text.strip():
                documents[display_name] = raw_text
            else:
                warnings.append(
                    f"{display_name}: no selectable text found; scanned PDFs need OCR."
                )
        except Exception as exc:
            warnings.append(f"{display_name}: could not read this PDF ({exc}).")

    return documents, warnings


def _read_sample_documents():
    """Load PDFs included in the project's Document folder."""
    from pypdf import PdfReader

    documents = {}
    warnings = []
    pdf_paths = sorted(DOCUMENT_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(
            "No sample PDFs were found in the Document folder. Upload PDFs instead."
        )

    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            page_texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    page_texts.append(text)
            raw_text = "\n".join(page_texts)
            if raw_text.strip():
                documents[path.name] = raw_text
            else:
                warnings.append(
                    f"{path.name}: no selectable text found; scanned PDFs need OCR."
                )
        except Exception as exc:
            warnings.append(f"{path.name}: could not read this PDF ({exc}).")

    return documents, warnings


def _run_lsh_analysis(raw_documents, threshold_percent, bands, warnings, source_label, started_at=None):
    """Run the notebook-compatible preprocessing/MinHash pipeline and LSH."""
    from preprocessing_runtime import (
        character_shingles,
        exact_jaccard,
        generate_signature_matrix,
        preprocess_text,
        text_to_hashed_shingles,
    )

    started = started_at if started_at is not None else perf_counter()
    cleaned_documents = {}
    raw_shingle_sets = {}
    hashed_shingles_by_doc = {}
    document_details = []

    for name, raw_text in raw_documents.items():
        cleaned = preprocess_text(raw_text)
        shingles = character_shingles(cleaned, k=SHINGLE_SIZE)
        if not shingles:
            warnings.append(
                f"{name}: skipped because its cleaned text is shorter than "
                f"the {SHINGLE_SIZE}-character shingle size."
            )
            continue

        # Keep the original notebook's hashing and MinHash input representation.
        hashed_shingles = text_to_hashed_shingles(cleaned, k=SHINGLE_SIZE)
        if len(hashed_shingles) == 0:
            warnings.append(f"{name}: skipped because no shingles were generated.")
            continue

        cleaned_documents[name] = cleaned
        raw_shingle_sets[name] = shingles
        hashed_shingles_by_doc[name] = hashed_shingles
        document_details.append({
            "name": name,
            "characters": len(cleaned),
            "shingles": len(shingles),
        })

    if not cleaned_documents:
        raise ValueError(
            "No usable text documents were found. Try selectable-text PDFs "
            "or check that the PDFs contain enough text."
        )

    # Notebook signature matrix shape is (num_hashes, num_documents).
    names = list(cleaned_documents.keys())
    signature_matrix = generate_signature_matrix(
        [hashed_shingles_by_doc[name] for name in names],
        num_hashes=NUM_HASHES,
    )
    signatures = {
        name: signature_matrix[:, index].tolist()
        for index, name in enumerate(names)
    }

    candidates, buckets = generate_candidates(signatures, bands=bands)

    # Count how many distinct bands each candidate pair shares.
    pair_bucket_counts = Counter()
    for document_ids in buckets.values():
        if len(document_ids) > 1:
            for pair in combinations(sorted(document_ids), 2):
                pair_bucket_counts[pair] += 1

    threshold = threshold_percent / 100.0
    pair_rows = []
    for doc_a, doc_b in candidates:
        similarity = exact_jaccard(
            raw_shingle_sets[doc_a],
            raw_shingle_sets[doc_b],
        )
        pair_rows.append({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "similarity": round(similarity * 100, 2),
            "shared_buckets": pair_bucket_counts[(doc_a, doc_b)],
            "is_match": similarity >= threshold,
            "status": "Near duplicate" if similarity >= threshold else "Below threshold",
        })

    pair_rows.sort(key=lambda row: (-row["similarity"], row["doc_a"], row["doc_b"]))
    near_duplicate_count = sum(1 for row in pair_rows if row["is_match"])
    document_count = len(names)
    possible_pairs = document_count * (document_count - 1) // 2
    candidate_count = len(candidates)
    candidate_reduction = (
        (1 - candidate_count / possible_pairs) * 100
        if possible_pairs
        else 100.0
    )
    collision_bucket_count = sum(1 for ids in buckets.values() if len(ids) > 1)
    elapsed_ms = (perf_counter() - started) * 1000

    return {
        "analysis_id": uuid4().hex,
        "source_label": source_label,
        "documents": document_details,
        # The main results table contains only pairs that meet the exact
        # Jaccard threshold. Rejected LSH candidates remain available in a
        # separate diagnostic table so they cannot be mistaken for matches.
        "pairs": [row for row in pair_rows if row["is_match"]],
        "rejected_pairs": [row for row in pair_rows if not row["is_match"]],
        "candidate_pairs": pair_rows,
        "warnings": warnings,
        "metrics": {
            "document_count": document_count,
            "candidate_count": candidate_count,
            "near_duplicate_count": near_duplicate_count,
            "threshold": threshold_percent,
            "bands": bands,
            "rows_per_band": NUM_HASHES // bands,
            "num_hashes": NUM_HASHES,
            "shingle_size": SHINGLE_SIZE,
            "possible_pairs": possible_pairs,
            "candidate_reduction": round(candidate_reduction, 2),
            "bucket_count": len(buckets),
            "collision_bucket_count": collision_bucket_count,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    }


def _store_analysis(result):
    ANALYSIS_CACHE[result["analysis_id"]] = result
    while len(ANALYSIS_CACHE) > MAX_CACHED_ANALYSES:
        oldest_key = next(iter(ANALYSIS_CACHE))
        if oldest_key == result["analysis_id"] and len(ANALYSIS_CACHE) > 1:
            oldest_key = next(key for key in ANALYSIS_CACHE if key != result["analysis_id"])
        ANALYSIS_CACHE.pop(oldest_key, None)
    session["analysis_id"] = result["analysis_id"]


def _current_analysis():
    analysis_id = session.get("analysis_id")
    return ANALYSIS_CACHE.get(analysis_id)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        analysis_started = perf_counter()
        try:
            threshold = int(request.form.get("threshold", "80"))
            bands = int(request.form.get("bands", "20"))
            if not 0 <= threshold <= 100:
                raise ValueError("Similarity threshold must be between 0% and 100%.")
            if bands not in {10, 20, 25, 50, 100}:
                raise ValueError("Choose one of the available band settings.")
            if NUM_HASHES % bands != 0:
                raise ValueError("The number of MinHash values must divide evenly into bands.")

            source = request.form.get("source", "upload")
            if source == "samples":
                raw_documents, warnings = _read_sample_documents()
                source_label = "PDFs in the Document folder"
            else:
                files = request.files.getlist("documents")
                raw_documents, warnings = _read_uploaded_documents(files)
                source_label = "Uploaded PDFs"

            if not raw_documents:
                detail = " ".join(warnings[:3])
                raise ValueError(
                    "No readable PDF text was found. Upload selectable-text PDFs. "
                    + detail
                )

            result = _run_lsh_analysis(
                raw_documents=raw_documents,
                threshold_percent=threshold,
                bands=bands,
                warnings=warnings,
                source_label=source_label,
                started_at=analysis_started,
            )
            _store_analysis(result)
            return redirect(url_for("results"))

        except (ValueError, ModuleNotFoundError) as exc:
            if isinstance(exc, ModuleNotFoundError):
                flash(
                    "A required package is missing. Run: python -m pip install -r requirements.txt",
                    "error",
                )
            else:
                flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception as exc:
            app.logger.exception("Document analysis failed")
            flash(
                f"Analysis failed: {exc}. Check the terminal for the detailed error.",
                "error",
            )
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/results")
def results():
    analysis = _current_analysis()
    return render_template("results.html", analysis=analysis)


@app.route("/performance")
def performance():
    analysis = _current_analysis()
    return render_template("performance.html", analysis=analysis)


@app.errorhandler(413)
def request_too_large(_error):
    flash("The upload is too large. Please keep the total upload under 50 MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
