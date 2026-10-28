"""Flask UI: LSH vs brute-force near-duplicate search, using Algorithm/lsh_engine.py."""

from pathlib import Path
from time import perf_counter
from uuid import uuid4
import os
import re
import sys

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
ALGORITHM_DIR = BASE_DIR / "Algorithm"
DOCUMENT_DIR = BASE_DIR / "Document"
sys.path.insert(0, str(ALGORITHM_DIR))

import lsh_engine as engine  # noqa: E402  (the one algorithm file)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB per request
app.config["MAX_PDF_COUNT"] = 30

# Local-development cache: only a short analysis ID is kept in the browser session.
# Results disappear if the Flask process restarts; this avoids putting large result
# objects into the session cookie or saving uploaded PDFs to disk.
ANALYSIS_CACHE = {}
MAX_CACHED_ANALYSES = 10
NUM_HASHES = engine.NUM_HASHES
ALLOWED_BANDS = {10, 20, 25, 50, 100}
ALLOWED_SUFFIXES = {".pdf", ".txt"}


def _unique_display_name(filename, used_names):
    """Keep uploaded file labels unique if two files share the same filename."""
    # PATCH 9: secure_filename() deletes every non-English letter ("日本.txt" -> "txt").
    # Uploads are only read in memory, never saved, so only folder parts and
    # characters that are unsafe to display are removed.
    base = Path(filename.replace("\\", "/")).name
    safe_name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", base).strip(" .") or "document.pdf"
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
            f"Please upload no more than {app.config['MAX_PDF_COUNT']} files at once."
        )

    for uploaded_file in files:
        if not uploaded_file or not uploaded_file.filename:
            continue
        suffix = Path(uploaded_file.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            warnings.append(f"{uploaded_file.filename}: skipped (only .pdf and .txt are supported).")
            continue

        display_name = _unique_display_name(uploaded_file.filename, used_names)
        try:
            uploaded_file.stream.seek(0)
            if suffix == ".txt":
                raw_text = uploaded_file.stream.read().decode("utf-8", errors="ignore")
            else:
                raw_text = _extract_pdf(uploaded_file.stream)
            if raw_text.strip():
                documents[display_name] = raw_text
            else:
                warnings.append(
                    f"{display_name}: no text found (scanned PDFs need OCR)."
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
            "No sample PDFs were found in the Document folder. Upload files instead."
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


def _run_lsh_analysis(raw_documents, threshold_percent, bands, warnings, source_label,
                      shingle_mode=engine.DEFAULT_MODE, shingle_size=None,
                      minhash_method="fast", extraction_ms=0.0):
    """Run LSH and the brute-force baseline on the SAME shingle sets (lsh_engine)."""
    result = engine.run_analysis(
        raw_documents,
        threshold=threshold_percent / 100.0,
        bands=bands,
        shingle_mode=shingle_mode,
        k=shingle_size,
        num_hashes=NUM_HASHES,
        minhash_method=minhash_method,
        run_brute_force=True,
        timing_repeats=3 if minhash_method == "fast" else 1,   # PATCH 4
    )
    result["warnings"] = list(warnings) + result["warnings"]
    result["metrics"]["extraction_ms"] = round(extraction_ms, 2)
    result["analysis_id"] = uuid4().hex
    result["source_label"] = source_label
    return result


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
        try:
            try:
                threshold = int(request.form.get("threshold", "80"))
                bands = int(request.form.get("bands", "20"))
                shingle_mode = request.form.get("shingle_mode", engine.DEFAULT_MODE)
                shingle_size = int(request.form.get(
                    "shingle_size", engine.DEFAULT_K.get(shingle_mode, 3)))
            except ValueError:
                raise ValueError("Threshold, bands and shingle size must be whole numbers.")
            minhash_method = request.form.get("minhash_method", "fast")
            if not 0 <= threshold <= 100:
                raise ValueError("Similarity threshold must be between 0% and 100%.")
            if bands not in ALLOWED_BANDS:
                raise ValueError("Choose one of the available band settings.")
            if NUM_HASHES % bands != 0:
                raise ValueError("The number of MinHash values must divide evenly into bands.")
            if minhash_method not in engine.MINHASH_METHODS:
                raise ValueError("Choose a valid MinHash method.")
            engine.validate_shingle_settings(shingle_mode, shingle_size)

            extraction_started = perf_counter()

            source = request.form.get("source", "upload")
            if source == "samples":
                raw_documents, warnings = _read_sample_documents()
                source_label = "Files in the Document folder"
            else:
                files = [f for f in request.files.getlist("documents") if f and f.filename]
                if not files:
                    raise ValueError("Please choose at least one PDF or TXT file to upload.")
                raw_documents, warnings = _read_uploaded_documents(files)
                source_label = "Uploaded files"

            if not raw_documents:
                detail = " ".join(warnings[:3])
                raise ValueError(
                    "No readable text was found. Upload selectable-text PDFs or TXT files. "
                    + detail
                )

            extraction_ms = (perf_counter() - extraction_started) * 1000

            result = _run_lsh_analysis(
                raw_documents=raw_documents,
                threshold_percent=threshold,
                bands=bands,
                warnings=warnings,
                source_label=source_label,
                shingle_mode=shingle_mode,
                shingle_size=shingle_size,
                minhash_method=minhash_method,
                extraction_ms=extraction_ms,
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

    return render_template("index.html", default_k=engine.DEFAULT_K, k_limits=engine.K_LIMITS)


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
