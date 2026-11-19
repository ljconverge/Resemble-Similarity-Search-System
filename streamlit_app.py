from __future__ import annotations

import io
import time
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

from Algorithm.preprocessing import extract_term_frequencies, probability_jaccard
from Algorithm.brute_force import brute_force_all_pairs, brute_force_query
from Algorithm.lsh import LSH, ProbMinHashLSH
from Algorithm.probminhash import ProbMinHasher, estimate_similarity
from Algorithm.evaluation import evaluate_accuracy, calculate_speedup
from services.dataset_service import load_corpus, get_dataset_stats


# Page Configuration & Styling
st.set_page_config(
    page_title="Resemble Similarity Search System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMainBlockContainer {
        max-width: 80rem;
        padding-top: 3rem;
    }
    .takeaway-card {
        background: #edf4f4;
        border-left: 3px solid #246565;
        border-radius: 0 6px 6px 0;
        color: #233b41;
        line-height: 1.55;
        padding: 14px 18px;
    }
    .metric-card-bf, .metric-card-lsh {
        background: #fff;
        border: 1px solid #dce4e8;
        border-radius: 8px;
        color: #182b36;
        padding: 18px 20px;
        font-variant-numeric: tabular-nums;
    }
    .metric-card-bf h4, .metric-card-lsh h4 {
        color: #182b36;
        font-size: 1rem;
        margin: 0 0 8px;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions & Cached Resources
@st.cache_resource
def get_benchmark_corpus():
    # Loads the 100 documents benchmark corpus and pre-indexes LSH.
    docs, manifest = load_corpus()
    freqs = {d_id: extract_term_frequencies(text) for d_id, text in docs.items()}
    doc_ids = sorted(docs.keys())
    hasher = ProbMinHasher(num_hashes=100, seed=42)

    # Pre-index LSH models for different thresholds
    lsh_high = ProbMinHashLSH(num_hashes=100, num_bands=20, hasher=hasher)  # r=5, threshold ~55%
    lsh_high.index_corpus(docs, cached_freqs=freqs)

    lsh_mid = ProbMinHashLSH(num_hashes=100, num_bands=25, hasher=hasher)   # r=4, threshold ~45%
    lsh_mid.index_corpus(docs, cached_freqs=freqs)

    lsh_low = ProbMinHashLSH(num_hashes=100, num_bands=50, hasher=hasher)   # r=2, threshold ~14%
    lsh_low.index_corpus(docs, cached_freqs=freqs)

    # Computing top match for each document
    best_matches = {}
    for i in range(len(doc_ids)):
        id_a = doc_ids[i]
        best_s, best_id = 0.0, ""
        for j in range(len(doc_ids)):
            if i == j:
                continue
            id_b = doc_ids[j]
            s = probability_jaccard(freqs[id_a], freqs[id_b])
            if s > best_s:
                best_s, best_id = s, id_b
        best_matches[id_a] = (best_id, round(best_s * 100.0, 1))

    return docs, manifest, freqs, doc_ids, best_matches, {"low": lsh_low, "mid": lsh_mid, "high": lsh_high}


def style_lsh_status(status: str) -> str:
    # Highlights retrieval status while keeping the text label visible.
    return {
        "Retrieved by LSH": "background-color: #dcfce7; color: #14532d;",
        "Missed by LSH": "background-color: #fee2e2; color: #7f1d1d;",
    }.get(status, "")


def extract_uploaded_text(uploaded_file) -> str:
    # Extracts text content from uploaded .txt, .pdf, or .docx file.
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    if name.endswith(".pdf"):
        if pypdf is None:
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = [p.extract_text() or "" for p in reader.pages[:50]]
            return "\n".join(pages).strip()
        except Exception:
            return ""
    elif name.endswith(".docx"):
        if docx is None:
            return ""
        try:
            doc = docx.Document(io.BytesIO(raw_bytes))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()
        except Exception:
            return ""
    else:
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore")


@st.cache_resource
def get_shared_hasher():
    return ProbMinHasher(num_hashes=100, seed=42)


# Main UI Layout
st.title("Resemble Similarity Search System")
st.caption("Compare exhaustive brute-force search with ProbMinHash locality-sensitive hashing (LSH).")

# Mode Selector
mode = st.radio(
    "Document source",
    ["Benchmark corpus", "Upload documents"],
    horizontal=True,
)

docs, manifest, corpus_freqs, doc_ids, best_matches, lsh_models = get_benchmark_corpus()


# MODE 1: BENCHMARK CORPUS
if mode == "Benchmark corpus":
    st.markdown("### Benchmark settings")

    col1, col2, col3 = st.columns([2.5, 1.5, 1])
    with col1:
        doc_options = ["ALL (Compare All 4,950 Pairs)"] + [
            f"{d_id} (Top match: {best_matches[d_id][1]}% with {best_matches[d_id][0]})"
            for d_id in doc_ids
        ]
        selected_option = st.selectbox("Document or corpus", doc_options, index=1)
        selected_doc_id = "ALL" if selected_option.startswith("ALL") else selected_option.split(" ")[0]

    with col2:
        threshold_val = st.slider(
            "Similarity threshold (%)",
            min_value=15,
            max_value=95,
            value=35,
            step=5,
            help="Pairs with similarity ≥ threshold are classified as near-duplicate matches.",
        ) / 100.0

    with col3:
        st.write("")
        st.write("")
        btn_placeholder = st.empty()
        run_corpus_btn = btn_placeholder.button("Compare documents", type="primary", use_container_width=True, key="btn_corpus_main")

    # Auto-run on first load or when user clicks Compare
    if run_corpus_btn or "corpus_evaluated" not in st.session_state:
        st.session_state["corpus_evaluated"] = True
        btn_placeholder.button("Processing...", disabled=True, use_container_width=True, key="btn_corpus_proc")

        with st.spinner("Processing comparisons with Brute Force & LSH..."):
            # Execution
            if selected_doc_id == "ALL":
                # Brute Force All Pairs
                bf_res = brute_force_all_pairs(docs, threshold=threshold_val, cached_freqs=corpus_freqs)

                # LSH All Pairs
                lsh_key = "low" if threshold_val < 0.35 else ("mid" if threshold_val < 0.60 else "high")
                lsh_res = lsh_models[lsh_key].find_all_matches(threshold=threshold_val)

                # Actual measured execution times
                bf_time = max(bf_res["execution_ms"], 0.001)
                lsh_time = max(lsh_res["search_ms"], 0.0005)
                speedup_ratio, speedup = calculate_speedup(bf_time, lsh_time)
                speed_label = f"{speedup} faster" if speedup_ratio >= 1.0 else f"{speedup} — LSH was slower"

                eval_metrics = evaluate_accuracy(lsh_res["matches"], bf_res["matches"])
                recall_str = eval_metrics["recall_display"]

                bf_pair_keys = {(min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"])): m for m in bf_res["matches"]}
                lsh_pair_keys = {(min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"])): m for m in lsh_res["matches"]}
                missed_count = eval_metrics["false_negatives_count"]

                # Render Takeaway
                if eval_metrics["has_ground_truth"]:
                    if speedup_ratio >= 1.0:
                        takeaway_speed = f"LSH is <strong>{speedup} faster</strong> than Brute Force ({lsh_time:.2f} ms vs {bf_time:.2f} ms)"
                    else:
                        takeaway_speed = f"LSH took <strong>{lsh_time:.2f} ms</strong> vs Brute Force <strong>{bf_time:.2f} ms</strong> — Brute Force was faster here (index was prebuilt at startup)"
                    st.markdown(f"""
                    <div class="takeaway-card">
                        Across all 100 documents (4,950 pairs), {takeaway_speed}, achieving <strong>{recall_str} recall</strong> of true ground-truth matches.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="takeaway-card">
                        Across all 100 documents (4,950 pairs), LSH search executed in <strong>{lsh_time:.2f} ms</strong> vs Brute Force {bf_time:.2f} ms. 
                        <strong>No document pairs had similarity ≥ {int(threshold_val*100)}% (0 reference matches).</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Render Metric Cards
                c_bf, c_lsh = st.columns(2)
                with c_bf:
                    st.markdown(f"""
                    <div class="metric-card-bf">
                        <h4>Brute Force (Ground Truth)</h4>
                        <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#1e293b;">{bf_time:.2f} ms</p>
                        <p style="color:#64748b; margin:0 0 10px 0;">Exhaustive 4,950 pairwise comparisons</p>
                        <p style="margin:0; font-size:1.3rem; font-weight:600; color:#315978;">100.0% Ground Truth</p>
                        <p style="color:#64748b; margin:0;">{len(bf_res['matches'])} True Pairs Found</p>
                    </div>
                    """, unsafe_allow_html=True)

                with c_lsh:
                    st.markdown(f"""
                    <div class="metric-card-lsh">
                        <h4>LSH (Locality-Sensitive Hashing)</h4>
                        <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#246565;">{lsh_time:.2f} ms <span style="font-size:1rem; color:#246565;">({speed_label})</span></p>
                        <p style="color:#64748b; margin:0 0 10px 0;">Banded bucket lookup (Pruned {lsh_res['pruned_pct']}% pairs)</p>
                        <p style="margin:0; font-size:1.3rem; font-weight:600; color:#246565;">{recall_str} Recall</p>
                        <p style="color:#64748b; margin:0;">Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} True Pairs</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("#### Evaluation metrics")
                summary_df = pd.DataFrame([
                    {"Metric": "Recall", "Brute Force": "100.0% (Ground Truth)", "LSH": f"{recall_str}", "LSH Result": f"Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} true pairs"},
                ])
                st.table(summary_df)

                # Unified Match Table
                st.markdown("#### Matched Document Pairs")
                table_rows = []
                for pair_key, m in bf_pair_keys.items():
                    is_found = pair_key in lsh_pair_keys
                    lsh_item = lsh_pair_keys.get(pair_key)
                    lsh_est = f"{lsh_item['estimated_similarity']:.1f}%" if lsh_item else "— (pruned)"
                    status_str = "Retrieved by LSH" if is_found else "Missed by LSH"
                    table_rows.append({
                        "Document A": m["doc_a"],
                        "Document B": m["doc_b"],
                        "Exact Similarity (BF)": f"{m['similarity']:.1f}%",
                        "LSH Estimated Similarity": lsh_est,
                        "Status in LSH": status_str,
                    })

                if table_rows:
                    st.dataframe(pd.DataFrame(table_rows).style.map(style_lsh_status, subset=["Status in LSH"]), use_container_width=True)
                else:
                    st.info("No document pairs met the similarity threshold.")

            else:
                # Single Document Query
                query_text = docs[selected_doc_id]
                bf_raw = brute_force_query(
                    query_text,
                    docs,
                    threshold=threshold_val,
                    top_k=None,
                    cached_freqs=corpus_freqs,
                    exclude_doc_id=selected_doc_id,
                )
                bf_hits = bf_raw["all_matches"]

                lsh_key = "low" if threshold_val < 0.35 else ("mid" if threshold_val < 0.60 else "high")
                lsh_raw = lsh_models[lsh_key].query(
                    query_text,
                    threshold=threshold_val,
                    top_k=None,
                    exclude_doc_id=selected_doc_id,
                )
                lsh_hits = lsh_raw["all_matches"]

                # Measured execution times
                bf_time = max(bf_raw["execution_ms"], 0.001)
                lsh_time = max(lsh_raw["execution_ms"], 0.0005)
                speedup_ratio, speedup = calculate_speedup(bf_time, lsh_time)
                speed_label = f"{speedup} faster" if speedup_ratio >= 1.0 else f"{speedup} — LSH was slower"

                # Evaluate complete sets of threshold-qualified matches
                eval_metrics = evaluate_accuracy(lsh_hits, bf_hits)
                recall_str = eval_metrics["recall_display"]
                missed_count = eval_metrics["false_negatives_count"]

                # Render Takeaway
                if eval_metrics["has_ground_truth"]:
                    if speedup_ratio >= 1.0:
                        takeaway_speed = f"LSH is <strong>{speedup} faster</strong> than Brute Force ({lsh_time:.2f} ms vs {bf_time:.2f} ms)"
                    else:
                        takeaway_speed = f"LSH took <strong>{lsh_time:.2f} ms</strong> vs Brute Force <strong>{bf_time:.2f} ms</strong> — Brute Force was faster here (index was prebuilt at startup)"
                    st.markdown(f"""
                    <div class="takeaway-card">
                        For document <code>{selected_doc_id}</code>, {takeaway_speed}, achieving <strong>{recall_str} recall</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="takeaway-card">
                        For document <code>{selected_doc_id}</code>, LSH query executed in <strong>{lsh_time:.2f} ms</strong> vs Brute Force {bf_time:.2f} ms. 
                        <strong>No corpus documents exceed {int(threshold_val*100)}% similarity with this document.</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Metric Cards
                c_bf, c_lsh = st.columns(2)
                with c_bf:
                    st.markdown(f"""
                    <div class="metric-card-bf">
                        <h4>Brute Force (Ground Truth)</h4>
                        <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#1e293b;">{bf_time:.2f} ms</p>
                        <p style="color:#64748b; margin:0 0 10px 0;">Exhaustive scan of remaining 99 documents</p>
                        <p style="margin:0; font-size:1.3rem; font-weight:600; color:#315978;">100.0% Ground Truth</p>
                        <p style="color:#64748b; margin:0;">{len(bf_hits)} Reference Matches</p>
                    </div>
                    """, unsafe_allow_html=True)

                with c_lsh:
                    st.markdown(f"""
                    <div class="metric-card-lsh">
                        <h4>LSH (Locality-Sensitive Hashing)</h4>
                        <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#246565;">{lsh_time:.2f} ms <span style="font-size:1rem; color:#246565;">({speed_label})</span></p>
                        <p style="color:#64748b; margin:0 0 10px 0;">Banded bucket search</p>
                        <p style="margin:0; font-size:1.3rem; font-weight:600; color:#246565;">{recall_str} Recall</p>
                        <p style="color:#64748b; margin:0;">Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} Matches</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("#### Evaluation metrics")
                summary_df = pd.DataFrame([
                    {"Metric": "Recall", "Brute Force": "100.0% (Ground Truth)", "LSH": f"{recall_str}", "LSH Result": f"Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} matches"},
                ])
                st.table(summary_df)

                # Table of similar documents
                st.markdown(f"#### Similar Documents to `{selected_doc_id}`")
                lsh_map = {r["doc_id"]: r for r in lsh_hits}
                table_rows = []
                for item in bf_hits:
                    d_id = item["doc_id"]
                    is_found = d_id in lsh_map
                    lsh_item = lsh_map.get(d_id)
                    lsh_est = f"{lsh_item['estimated_similarity']:.1f}%" if lsh_item else "— (pruned)"
                    table_rows.append({
                        "Document ID": d_id,
                        "Exact Similarity (BF)": f"{item['similarity']:.1f}%",
                        "LSH Estimated Similarity": lsh_est,
                        "Status in LSH": "Retrieved by LSH" if is_found else "Missed by LSH",
                    })

                if table_rows:
                    st.dataframe(pd.DataFrame(table_rows).style.map(style_lsh_status, subset=["Status in LSH"]), use_container_width=True)
                else:
                    best_partner, best_sim = best_matches.get(selected_doc_id, ("", 0.0))
                    st.info(f"No documents have similarity ≥ {int(threshold_val*100)}% with `{selected_doc_id}`. (Its highest similarity in the corpus is {best_sim}% with `{best_partner}`). Lower the threshold slider to see matches.")

        # Reset button label back
        btn_placeholder.button("Compare documents", type="primary", use_container_width=True, key="btn_corpus_done")



# MODE 2: USER DOCUMENT UPLOAD (MINIMUM 2 REQUIRED)
else:
    st.markdown("### Upload documents")
    st.markdown("Upload at least **2 documents** (`.txt`, `.pdf`, `.docx`) to compute exact Brute Force vs. fast LSH similarity.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload 2 or more documents to compare their pairwise similarities.",
    )

    col_up_s, col_up_b = st.columns([3, 1])
    with col_up_s:
        upload_threshold = st.slider(
            "Similarity threshold for uploads (%)",
            min_value=15,
            max_value=95,
            value=35,
            step=5,
            help="Pairs with similarity ≥ threshold are classified as matches.",
            key="slider_upload",
        ) / 100.0

    with col_up_b:
        st.write("")
        st.write("")
        up_btn_placeholder = st.empty()
        compare_upload_btn = up_btn_placeholder.button("Compare uploads", type="primary", use_container_width=True, key="btn_up_main")

    # Empty State: If no files or <2 files uploaded, keep space clean and empty
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("**No comparison results yet.** Please upload at least **2 documents** above and click **'Compare uploads'**.")
    elif compare_upload_btn:
        up_btn_placeholder.button("Processing...", disabled=True, use_container_width=True, key="btn_up_proc")

        # Parse text from all uploaded documents
        uploaded_docs: Dict[str, str] = {}
        for f in uploaded_files:
            text = extract_uploaded_text(f)
            if text.strip():
                fname = f.name
                counter = 1
                while fname in uploaded_docs:
                    fname = f"{f.name}_{counter}"
                    counter += 1
                uploaded_docs[fname] = text

        if len(uploaded_docs) < 2:
            st.error("At least 2 non-empty documents are required to perform a similarity comparison.")
        else:
            with st.spinner("Processing uploaded documents with Brute Force & LSH..."):
                # Extract term frequencies once
                up_freqs = {d_id: extract_term_frequencies(text) for d_id, text in uploaded_docs.items()}

                # Brute Force
                bf_res = brute_force_all_pairs(uploaded_docs, threshold=upload_threshold, cached_freqs=up_freqs)

                # LSH
                num_bands = 50 if upload_threshold < 0.35 else (25 if upload_threshold < 0.60 else 20)
                upload_lsh = ProbMinHashLSH(num_hashes=100, num_bands=num_bands, hasher=get_shared_hasher())
                
                # Measure indexing construction time vs search time
                idx_ms = upload_lsh.index_corpus(uploaded_docs, cached_freqs=up_freqs)
                lsh_res = upload_lsh.find_all_matches(threshold=upload_threshold)
                search_ms = lsh_res["search_ms"]
                total_lsh_ms = idx_ms + search_ms

                # Actual measured execution times
                bf_time = max(bf_res["execution_ms"], 0.001)
                lsh_time = max(search_ms, 0.0005)
                speedup_ratio, speedup = calculate_speedup(bf_time, lsh_time)
                speed_label = f"{speedup} faster" if speedup_ratio >= 1.0 else f"{speedup} — LSH was slower"

                eval_metrics = evaluate_accuracy(lsh_res["matches"], bf_res["matches"])
                recall_str = eval_metrics["recall_display"]

                bf_pair_keys = {(min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"])): m for m in bf_res["matches"]}
                lsh_pair_keys = {(min(m["doc_a"], m["doc_b"]), max(m["doc_a"], m["doc_b"])): m for m in lsh_res["matches"]}
                missed_count = eval_metrics["false_negatives_count"]

            # Render Takeaway
            if eval_metrics["has_ground_truth"]:
                if speedup_ratio >= 1.0:
                    takeaway_speed = f"LSH Search is <strong>{speedup} faster</strong> ({lsh_time:.2f} ms search vs {bf_time:.2f} ms Brute Force)"
                else:
                    takeaway_speed = f"LSH Search took <strong>{lsh_time:.2f} ms</strong> vs Brute Force <strong>{bf_time:.2f} ms</strong> — Brute Force was faster here (small corpus; LSH advantage grows with more documents)"
                st.markdown(f"""
                <div class="takeaway-card">
                    Across {len(uploaded_docs)} uploaded documents ({bf_res['comparisons']} pairs), {takeaway_speed} with <strong>{recall_str} recall</strong>.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="takeaway-card">
                    Across {len(uploaded_docs)} uploaded documents ({bf_res['comparisons']} pairs), LSH Search took <strong>{lsh_time:.2f} ms</strong> 
                    vs Brute Force <strong>{bf_time:.2f} ms</strong>. 
                    <strong>No uploaded pairs had similarity ≥ {int(upload_threshold*100)}% (0 reference matches).</strong>
                </div>
                """, unsafe_allow_html=True)

            # Metric Cards
            c_bf, c_lsh = st.columns(2)
            with c_bf:
                st.markdown(f"""
                <div class="metric-card-bf">
                    <h4>Brute Force (Ground Truth)</h4>
                    <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#1e293b;">{bf_time:.2f} ms</p>
                    <p style="color:#64748b; margin:0 0 10px 0;">Evaluated all {bf_res['comparisons']} pairs directly</p>
                    <p style="margin:0; font-size:1.3rem; font-weight:600; color:#315978;">100.0% Ground Truth</p>
                    <p style="color:#64748b; margin:0;">{len(bf_res['matches'])} True Pairs Found</p>
                </div>
                """, unsafe_allow_html=True)

            with c_lsh:
                st.markdown(f"""
                <div class="metric-card-lsh">
                    <h4>LSH (Locality-Sensitive Hashing)</h4>
                    <p style="margin:0; font-size: 1.8rem; font-weight:700; color:#246565;">{lsh_time:.2f} ms search <span style="font-size:1rem; color:#246565;">({speed_label})</span></p>
                    <p style="color:#64748b; margin:0 0 10px 0;">Indexing: {idx_ms:.2f} ms | Total: {total_lsh_ms:.2f} ms</p>
                    <p style="margin:0; font-size:1.3rem; font-weight:600; color:#246565;">{recall_str} Recall</p>
                    <p style="color:#64748b; margin:0;">Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} True Pairs</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Evaluation metrics")
            summary_df = pd.DataFrame([
                {"Metric": "Recall", "Brute Force": "100.0% (Ground Truth)", "LSH": f"{recall_str}", "LSH Result": f"Retrieved {eval_metrics['true_positives_count']} of {eval_metrics['ground_truth_count']} true pairs"},
            ])
            st.table(summary_df)

            # Unified Match Table
            st.markdown("#### Matches in uploaded documents")
            table_rows = []
            for pair_key, m in bf_pair_keys.items():
                is_found = pair_key in lsh_pair_keys
                lsh_item = lsh_pair_keys.get(pair_key)
                lsh_est = f"{lsh_item['estimated_similarity']:.1f}%" if lsh_item else "— (pruned)"
                table_rows.append({
                    "Document A": m["doc_a"],
                    "Document B": m["doc_b"],
                    "Exact Similarity (BF)": f"{m['similarity']:.1f}%",
                    "LSH Estimated Similarity": lsh_est,
                    "Status in LSH": "Retrieved by LSH" if is_found else "Missed by LSH",
                })

            if table_rows:
                st.dataframe(pd.DataFrame(table_rows).style.map(style_lsh_status, subset=["Status in LSH"]), use_container_width=True)
            else:
                st.info(f"No document pairs had similarity ≥ {int(upload_threshold*100)}%. Lower the threshold slider to see matches.")

        # Reset button label back
        up_btn_placeholder.button("Compare uploads", type="primary", use_container_width=True, key="btn_up_done")
