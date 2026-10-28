"""
extract_books.py  (version 2: quality-checked extraction)
=========================================================
Extracts text from every PDF (and .txt) in a book folder, in parallel, and
CHECKS the result. If a book comes out empty, garbled, or as random numbers,
it automatically tries the next method, and finally OCR. Books in the wrong
language are rejected.

Usage:
    python extract_books.py books/ extracted/
    python extract_books.py books/ extracted/ --lang en --min-words 1000
    python extract_books.py books/ extracted/ --no-ocr        (skip the slow OCR step)
    python extract_books.py books/ extracted/ --keep-bad      (also keep rejected books)

Needs text_quality.py in the same folder.

Methods tried, in order, until the text passes the quality check:
    1. PyMuPDF    (pip install pymupdf)
    2. pdftotext  (poppler-utils)
    3. pdfminer.six (pip install pdfminer.six)
    4. pypdf      (pip install pypdf)
    5. OCR        pdftoppm + tesseract (command-line tools), only if still bad

Output:
    extracted/<category>/<book>.txt      accepted books
    extracted/manifest.json              accepted books (doc_id, category, backend, ...)
    extracted/extraction_report.json     EVERY book with status, reason, methods tried
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from text_quality import analyze, classify


# ---------------------------------------------------------------- backends

def _import_pymupdf():
    """New PyMuPDF versions are imported as 'pymupdf'; 'fitz' is the deprecated name."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz


def _pages_pymupdf(path: Path) -> list[str]:
    fitz = _import_pymupdf()
    with fitz.open(path) as doc:
        return [page.get_text("text") for page in doc]


def _pages_pdftotext(path: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                         capture_output=True, check=True).stdout
    pages = out.decode("utf-8", errors="ignore").split("\f")
    if pages and not pages[-1].strip():          # trailing form-feed -> empty page
        pages.pop()
    return pages


def _pages_pdfminer(path: Path) -> list[str]:
    from pdfminer.high_level import extract_text
    pages = extract_text(str(path)).split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def _pages_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def _pages_ocr(path: Path, dpi: int = 200, ocr_lang: str = "eng") -> list[str]:
    """Render every page to an image and read it with Tesseract.
    Slow (seconds per page) but works on scanned books and broken fonts."""
    env = dict(os.environ, OMP_THREAD_LIMIT="1")   # one thread per process
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png",
                        str(path), f"{tmp}/p"], check=True, capture_output=True)
        pages = []
        for img in sorted(Path(tmp).glob("p*.png")):
            res = subprocess.run(["tesseract", str(img), "stdout", "-l", ocr_lang],
                                 capture_output=True, env=env)
            pages.append(res.stdout.decode("utf-8", errors="ignore"))
        return pages


def available_backends():
    """Text-layer methods that can run on this machine, fastest first."""
    out = []
    try:
        _import_pymupdf()
        out.append(("pymupdf", _pages_pymupdf))
    except ImportError:
        pass
    if shutil.which("pdftotext"):
        out.append(("pdftotext", _pages_pdftotext))
    try:
        import pdfminer  # noqa: F401
        out.append(("pdfminer", _pages_pdfminer))
    except ImportError:
        pass
    try:
        import pypdf  # noqa: F401
        out.append(("pypdf", _pages_pypdf))
    except ImportError:
        pass
    return out


def ocr_available() -> bool:
    return bool(shutil.which("tesseract") and shutil.which("pdftoppm"))


# ---------------------------------------------------- header/footer removal

def remove_running_lines(pages: list[str], min_share: float = 0.3) -> list[str]:
    """Drop lines that repeat on many pages (running headers, footers, page
    numbers, site watermarks). Digits are ignored when comparing."""
    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    n = max(1, len(pages))
    counts = Counter()
    for p in pages:
        counts.update({key(l) for l in p.splitlines() if l.strip()})
    repeated = {k for k, c in counts.items()
                if c >= max(3, min_share * n) and len(k) < 80}

    return ["\n".join(l for l in p.splitlines() if l.strip() and key(l) not in repeated)
            for p in pages]


# ------------------------------------------------------------------ worker

BAD = ("EMPTY", "TOO_SHORT", "GARBLED", "UNRECOGNISED")   # worth trying another method


def extract_one(job):
    src, dst, opts = job
    src, dst = Path(src), Path(dst)
    attempts = []                       # what was tried: [(method, status, reason)]
    final = None                        # (method, text, metrics, status, reason)

    def evaluate(method, pages):
        text = "\n\n".join(remove_running_lines(pages))
        m = analyze(text)
        status, reason = classify(m, opts["lang"], opts["min_words"])
        # Header removal can erase lines made only of numbers, which hides a
        # 'random numbers' extraction. Judge the RAW text for that problem too.
        raw_status, raw_reason = classify(analyze("\n\n".join(pages)),
                                          opts["lang"], opts["min_words"])
        if raw_status == "GARBLED":
            status, reason = raw_status, raw_reason
        attempts.append({"method": method, "status": status, "reason": reason,
                         "words": m["n_words"]})
        return text, m, status, reason

    try:
        if src.suffix.lower() == ".txt":
            text = src.read_text(encoding="utf-8", errors="ignore")
            m = analyze(text)
            status, reason = classify(m, opts["lang"], opts["min_words"])
            attempts.append({"method": "plain_text", "status": status,
                             "reason": reason, "words": m["n_words"]})
            final = ("plain_text", text, m, status, reason)
        else:
            # 1) text-layer methods, stop at the first readable result
            for name, fn in available_backends():
                try:
                    pages = fn(src)
                except Exception as exc:
                    attempts.append({"method": name, "status": "ERROR",
                                     "reason": str(exc)[:120], "words": 0})
                    continue
                text, m, status, reason = evaluate(name, pages)
                final = (name, text, m, status, reason)
                if status not in BAD:
                    break

            # 2) OCR, only if every text-layer method failed
            if (final is None or final[3] in BAD) and opts["ocr"] and ocr_available():
                try:
                    pages = _pages_ocr(src, opts["ocr_dpi"], opts["ocr_lang"])
                    text, m, status, reason = evaluate("ocr", pages)
                    if final is None or status not in BAD or m["n_words"] > final[2]["n_words"]:
                        final = ("ocr", text, m, status, reason)
                except Exception as exc:
                    attempts.append({"method": "ocr", "status": "ERROR",
                                     "reason": str(exc)[:120], "words": 0})

        if final is None:
            return {"src": str(src), "status": "ERROR",
                    "reason": "no extraction method available or all failed",
                    "attempts": attempts}

        method, text, m, status, reason = final
        keep = status == "OK" or opts["keep_bad"]
        if keep:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
        return {"src": str(src), "dst": str(dst) if keep else None, "status": status,
                "reason": reason, "method": method, "words": m["n_words"],
                "language": m["best_lang"], "attempts": attempts}
    except Exception as exc:
        return {"src": str(src), "status": "ERROR", "reason": str(exc)[:200],
                "attempts": attempts}


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("books_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--lang", default="en", help="target language, or 'any' (default en)")
    ap.add_argument("--min-words", type=int, default=1000)
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR fallback")
    ap.add_argument("--ocr-dpi", type=int, default=200)
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language code (default eng)")
    ap.add_argument("--keep-bad", action="store_true",
                    help="also write books that failed the quality check")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.books_dir), Path(args.out_dir)
    opts = {"lang": args.lang, "min_words": args.min_words, "ocr": not args.no_ocr,
            "ocr_dpi": args.ocr_dpi, "ocr_lang": args.ocr_lang, "keep_bad": args.keep_bad}

    jobs = []
    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() in (".pdf", ".txt"):
            rel = p.relative_to(in_dir).with_suffix(".txt")
            jobs.append((str(p), str(out_dir / rel), opts))

    methods = [n for n, _ in available_backends()]
    print(f"{len(jobs)} files | text methods: {methods or 'NONE'} | "
          f"OCR: {'on' if opts['ocr'] and ocr_available() else 'off'} | "
          f"target language: {args.lang}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(extract_one, jobs))

    documents = []
    for r in results:
        if r.get("dst") and r["status"] == "OK":
            rel = Path(r["dst"]).relative_to(out_dir)
            documents.append({
                "doc_id": rel.with_suffix("").as_posix(),
                "category": rel.parts[0] if len(rel.parts) > 1 else "",
                "text_path": rel.as_posix(),
                "words": r["words"], "method": r["method"], "language": r["language"],
            })

    (out_dir / "manifest.json").write_text(
        json.dumps({"documents": documents}, indent=2), encoding="utf-8")
    (out_dir / "extraction_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    counts = Counter(r["status"] for r in results)
    print("\nresult: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for r in results:
        if r["status"] != "OK":
            tried = ", ".join(f"{a['method']}:{a['status']}" for a in r["attempts"])
            print(f"  {r['status']:<15}{Path(r['src']).name}\n"
                  f"{'':<17}{r['reason']}\n{'':<17}tried: {tried}")
    fixed = [r for r in results if r["status"] == "OK" and r.get("method") == "ocr"]
    if fixed:
        print(f"\n{len(fixed)} book(s) were recovered with OCR.")
    print(f"\n{len(documents)} accepted books -> {out_dir}/manifest.json")


if __name__ == "__main__":
    main()