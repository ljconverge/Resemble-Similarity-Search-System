"""
Flask Application for Resemble Similarity Search System
=======================================================
Web controller and REST API interfaces.
Coordinates the LSH query and collection engines with the SHA-256 bucketing service.
Routes:
  - GET  /                  : Minimalist primary dashboard (index.html)
  - POST /                  : Analyze uploaded files
  - GET  /detailed-analysis : In-depth performance, pairwise verification, diagnostics (detailed.html)
  - GET  /performance        : Backward-compatible performance route
  - GET  /results            : Backward-compatible results route
  - JSON APIs               : /api/search, /api/analyze, /api/performance, /api/buckets
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, redirect, render_template, request, url_for

# Ensure local packages (Algorithm, services, tools) are importable
ROOT_DIR: Path = Path(__file__).resolve().parent
for sub in ("", "Algorithm", "services", "tools"):
    sub_path = str(ROOT_DIR / sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

from Algorithm.lsh_engine import (
    LSHEngine,
    MINHASH_METHODS,
    NUM_HASHES_DEFAULT,
    VALID_BANDS,
    SearchResult,
    extract_text_from_bytes,
    load_folder,
    run_analysis,
)
from services.bucketing import (
    HASH_TRUNCATE_LEN,
    MOCK_RECORDS,
    NUM_BUCKETS,
    build_bucketed_dataset,
    compute_sha256_bucket,
)

# New Directory Structure
SYNTHETIC_FOLDER: Path = ROOT_DIR / "synthetic_datasets"
REAL_WORLD_FOLDER: Path = ROOT_DIR / "real_world_dataset"
RESEARCH_FOLDER: Path = ROOT_DIR / "research_papers"
SAMPLE_FOLDER: Path = SYNTHETIC_FOLDER

ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt"}

DEFAULT_CORPUS: Dict[str, str] = {
    "CV_2014_Simonyan_VGG": (
        "Very deep convolutional networks for large-scale image recognition using "
        "small 3x3 convolution filters and deep architecture layers."
    ),
    "CV_2015_He_ResNet": (
        "Deep residual learning for image recognition introducing skip connections "
        "and residual blocks to train very deep convolutional neural networks."
    ),
    "CV_2020_Dosovitskiy_ViT": (
        "An image is worth 16x16 words: transformers for image recognition at scale "
        "applying self-attention directly to sequences of image patches."
    ),
    "GRAPH_2016_Kipf_GCN": (
        "Semi-supervised classification with graph convolutional networks scaling "
        "spectral graph convolutions linearly in the number of graph edges."
    ),
    "GRAPH_2017_Hamilton_GraphSAGE": (
        "Inductive representation learning on large graphs generating low-dimensional "
        "node embeddings by sampling and aggregating features from local neighborhoods."
    ),
}


# ===========================================================================
# Validation & Sanitization Helpers
# ===========================================================================

def _unique_display_name(raw_name: str, used: Set[str]) -> str:
    """Preserves Unicode filenames, strips path traversal sequences, and deduplicates."""
    base = re.split(r"[\\/]", str(raw_name or "document.txt"))[-1]
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(". ")
    if not cleaned:
        cleaned = "document.txt"

    if cleaned not in used:
        used.add(cleaned)
        return cleaned

    stem, ext = os.path.splitext(cleaned)
    counter = 2
    while f"{stem}_{counter}{ext}" in used:
        counter += 1
    unique_name = f"{stem}_{counter}{ext}"
    used.add(unique_name)
    return unique_name


def validate_search_params(
    params: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validates threshold, bands, shingle_mode, shingle_size, and minhash_method."""
    raw_thresh = params.get("threshold", 80)
    try:
        thresh_val = float(raw_thresh)
    except (TypeError, ValueError):
        return None, "Threshold must be a valid number between 0 and 100."

    if thresh_val < 0 or thresh_val > 100:
        return None, "Threshold must be between 0 and 100%."
    threshold = (
        thresh_val / 100.0
        if thresh_val > 1.0 or str(raw_thresh).strip() in {"1", "100"}
        else thresh_val
    )

    try:
        bands = int(params.get("bands", 20))
        if str(params.get("bands", 20)).strip() != str(bands):
            raise ValueError
    except (TypeError, ValueError):
        return None, f"Bands must be an integer in {VALID_BANDS}."
    if bands not in VALID_BANDS:
        return None, f"Invalid band count ({bands}). Choose from {VALID_BANDS}."

    mode = str(params.get("shingle_mode") or params.get("mode") or "word").strip().lower()
    if mode not in ("word", "char"):
        return None, "Shingle mode must be 'word' or 'char'."

    default_k = 3 if mode == "word" else 9
    raw_k = params.get("shingle_size", params.get("k", default_k))
    try:
        k = int(raw_k)
        if str(raw_k).strip() != str(k):
            raise ValueError
    except (TypeError, ValueError):
        if mode == "char":
            return None, "Shingle size for char shingles must be 1-30 characters."
        return None, "Shingle size for word shingles must be 1-20 words."

    if mode == "char" and not (1 <= k <= 30):
        return None, "Shingle size for char shingles must be 1-30 characters."
    if mode == "word" and not (1 <= k <= 20):
        return None, "Shingle size for word shingles must be 1-20 words."

    minhash_method = str(params.get("minhash_method", "fast")).strip().lower()
    if minhash_method not in MINHASH_METHODS:
        return None, f"MinHash method must be one of {MINHASH_METHODS}."

    return {
        "threshold": threshold,
        "bands": bands,
        "shingle_mode": mode,
        "shingle_size": k,
        "minhash_method": minhash_method,
    }, None


# ===========================================================================
# Flask Application Factory
# ===========================================================================

def create_app(engine: Optional[LSHEngine] = None) -> Flask:
    app = Flask(__name__)

    if engine is None:
        engine = LSHEngine(num_hashes=100, num_bands=20, shingle_size=2)
        for doc_id, text in DEFAULT_CORPUS.items():
            engine.add_document(doc_id, text)

    app.config["LSH_ENGINE"] = engine
    app.config["LAST_ANALYSIS"] = None
    app.config["MOCK_RECORDS"] = list(MOCK_RECORDS)

    def _template_context(**extra: Any) -> Dict[str, Any]:
        current_engine: LSHEngine = app.config["LSH_ENGINE"]
        bucket_data = build_bucketed_dataset(
            app.config["MOCK_RECORDS"], num_buckets=NUM_BUCKETS
        )
        ctx: Dict[str, Any] = {
            "metrics": current_engine.get_performance_metrics(),
            "analysis": app.config.get("LAST_ANALYSIS"),
            "error": None,
            "results": [],
            "query": "",
            "active_tab": "all",
            "records": bucket_data["records"],
            "buckets": bucket_data["buckets"],
            "bucket_summaries": bucket_data["bucket_summaries"],
            "total_records": bucket_data["total_records"],
            "total_value": bucket_data["total_value"],
            "average_value": bucket_data["average_value"],
            "num_buckets": bucket_data["num_buckets"],
        }
        ctx.update(extra)
        return ctx

    def _execute_analysis_from_request() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        form_data = (
            request.form.to_dict()
            if request.form
            else (request.get_json(silent=True) or {})
        )
        validated, err = validate_search_params(form_data)
        if err or validated is None:
            return None, err

        source = str(form_data.get("source", "upload")).strip().lower()
        t0 = time.perf_counter()
        docs: Dict[str, str] = {}

        if source == "sample":
            target_folder = SYNTHETIC_FOLDER if SYNTHETIC_FOLDER.is_dir() else ROOT_DIR / "Document"
            if not target_folder.is_dir():
                return None, f"Sample folder '{target_folder}' was not found."
            docs = load_folder(target_folder)
            source_label = "Synthetic Datasets (10 docs)"
        elif source == "real_world":
            if not REAL_WORLD_FOLDER.is_dir():
                return None, f"Real-world folder '{REAL_WORLD_FOLDER}' was not found."
            docs = load_folder(REAL_WORLD_FOLDER)
            source_label = "Real-World Dataset (17 papers)"
        else:
            uploaded_files = [
                f for f in request.files.getlist("documents") if f and f.filename
            ]
            if not uploaded_files:
                return None, "Please choose at least one PDF or TXT file to upload."

            used_names: Set[str] = set()
            for storage in uploaded_files:
                ext = Path(storage.filename or "").suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    return (
                        None,
                        f"Unsupported file '{storage.filename}'. Allowed: {ALLOWED_EXTENSIONS}",
                    )
                disp_name = _unique_display_name(
                    storage.filename or "document.txt", used_names
                )
                raw_bytes = storage.read()
                docs[disp_name] = extract_text_from_bytes(disp_name, raw_bytes)
            source_label = f"{len(docs)} uploaded file(s)"

        extraction_ms = (time.perf_counter() - t0) * 1000.0
        try:
            analysis = run_analysis(
                docs,
                threshold=validated["threshold"],
                bands=validated["bands"],
                shingle_mode=validated["shingle_mode"],
                shingle_size=validated["shingle_size"],
                minhash_method=validated["minhash_method"],
                extraction_ms=extraction_ms,
                source_label=source_label,
                timing_repeats=3,
            )
        except ValueError as exc:
            return None, str(exc)

        app.config["LAST_ANALYSIS"] = analysis
        return analysis, None

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.route("/", methods=["GET", "POST"])
    def index() -> Any:
        if request.method == "POST":
            analysis, err = _execute_analysis_from_request()
            if (
                request.headers.get("X-Requested-With") == "XMLHttpRequest"
                or request.is_json
            ):
                if err:
                    return jsonify({"error": err}), 400
                return jsonify({"analysis": analysis, "redirect_url": "/detailed-analysis"}), 200

            return render_template(
                "index.html",
                **_template_context(
                    analysis=analysis,
                    error=err,
                ),
            ), (400 if err else 200)

        return render_template("index.html", **_template_context())

    @app.route("/detailed-analysis", methods=["GET"])
    def detailed_analysis() -> Any:
        return render_template("detailed.html", **_template_context(active_tab="detailed"))

    @app.route("/results", methods=["GET"])
    def results() -> Any:
        return render_template("detailed.html", **_template_context(active_tab="results"))

    @app.route("/performance", methods=["GET"])
    def performance() -> Any:
        current_engine: LSHEngine = app.config["LSH_ENGINE"]
        if (
            request.is_json
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.args.get("format") == "json"
        ):
            payload = current_engine.get_performance_metrics()
            if app.config.get("LAST_ANALYSIS"):
                payload["analysis_metrics"] = app.config["LAST_ANALYSIS"]["metrics"]
            return jsonify(payload), 200

        return render_template(
            "detailed.html", **_template_context(active_tab="performance")
        )

    @app.route("/api/buckets", methods=["GET"])
    def api_buckets() -> Any:
        dataset = build_bucketed_dataset(
            app.config["MOCK_RECORDS"], num_buckets=NUM_BUCKETS
        )
        return jsonify(dataset), 200

    @app.route("/api/performance", methods=["GET"])
    def api_performance() -> Any:
        current_engine: LSHEngine = app.config["LSH_ENGINE"]
        payload = current_engine.get_performance_metrics()
        if app.config.get("LAST_ANALYSIS"):
            payload["analysis_metrics"] = app.config["LAST_ANALYSIS"]["metrics"]
        return jsonify(payload), 200

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze() -> Any:
        analysis, err = _execute_analysis_from_request()
        if err:
            return jsonify({"error": err}), 400
        return jsonify({"analysis": analysis, "redirect_url": "/detailed-analysis"}), 200

    @app.route("/api/search", methods=["POST", "GET"])
    @app.route("/search", methods=["POST", "GET"])
    def search() -> Any:
        current_engine: LSHEngine = app.config["LSH_ENGINE"]

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            query_text = str(payload.get("query", "")).strip()
            top_k = int(payload.get("top_k", 5))
            threshold = float(payload.get("threshold", 0.0))
        else:
            data = request.form if request.method == "POST" else request.args
            query_text = str(data.get("query", "")).strip()
            top_k = int(data.get("top_k", 5))
            threshold = float(data.get("threshold", 0.0))

        if not query_text:
            response_data = {
                "query": "",
                "results": [],
                "count": 0,
                "metrics": current_engine.get_performance_metrics(),
                "error": "Query string cannot be empty.",
            }
            if (
                request.path.startswith("/api/")
                or request.is_json
                or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            ):
                return jsonify(response_data), 400
            return render_template(
                "detailed.html", **_template_context(**response_data)
            ), 400

        hits = [
            r.to_dict()
            for r in current_engine.query(
                query_text=query_text, top_k=top_k, threshold=threshold
            )
        ]
        metrics = current_engine.get_performance_metrics()
        response_data = {
            "query": query_text,
            "results": hits,
            "count": len(hits),
            "metrics": metrics,
        }

        if (
            request.path.startswith("/api/")
            or request.is_json
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        ):
            return jsonify(response_data), 200

        return render_template(
            "detailed.html", **_template_context(**response_data)
        ), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
