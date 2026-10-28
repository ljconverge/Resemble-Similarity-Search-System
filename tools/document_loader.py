"""
document_loader.py
==================
Reads every .pdf / .docx / .txt file in a folder into memory, for the
brute_force_k*.py and compare_k1_k3.py scripts.

    corpus, info = load_documents(Path("Document"))
    corpus -> {doc_id: text}          doc_id = file name without extension, e.g. "Doc1"
    info   -> [(doc_id, word_count, method), ...]

PDFs: the text-layer methods from extract_books.py are tried in order
(PyMuPDF, pdftotext, pdfminer, pypdf -- whichever are installed). The first
result that text_quality.py accepts is used; if none passes, the result with
the most words is kept and a warning is printed. OCR is NOT run here (slow);
use extract_books.py for scanned books.

DOCX: read with the standard library (a .docx is a zip of XML files).
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from extract_books import available_backends, remove_running_lines
from text_quality import analyze, classify

SUPPORTED = (".pdf", ".docx", ".txt")
MIN_WORDS = 20            # a document shorter than this is reported as TOO_SHORT
_WORD_RE = re.compile(r"[a-z0-9']+")
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def count_words(text: str) -> int:
    """Same word definition as brute_force_k*.py (letters, digits, apostrophes)."""
    return len(_WORD_RE.findall(text.lower()))


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    paragraphs = []
    for p in root.iter(f"{_W_NS}p"):
        paragraphs.append("".join(t.text or "" for t in p.iter(f"{_W_NS}t")))
    return "\n".join(paragraphs)


def read_pdf(path: Path, lang: str = "any", min_words: int = MIN_WORDS):
    """Return (text, method, status, reason) using the best available backend."""
    backends = available_backends()
    if not backends:
        raise RuntimeError("No PDF reader installed. Run: python -m pip install pypdf")
    best = None
    for name, fn in backends:
        try:
            pages = fn(path)
        except Exception as exc:                       # this backend failed, try next
            print(f"  {path.name}: {name} failed ({str(exc)[:80]})", file=sys.stderr)
            continue
        text = "\n\n".join(remove_running_lines(pages))
        status, reason = classify(analyze(text), lang, min_words)
        if status == "OK":
            return text, name, status, reason
        if best is None or count_words(text) > count_words(best[0]):
            best = (text, name, status, reason)
    if best is None:
        raise RuntimeError("every PDF reader failed")
    return best


def load_documents(folder, lang: str = "any", min_words: int = MIN_WORDS, verbose: bool = True):
    """Load a folder. Unreadable or empty files are skipped with a message."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")
    corpus, info = {}, []
    used = set()
    for path in sorted(folder.iterdir(), key=lambda p: _natural_key(p.name)):
        if path.suffix.lower() not in SUPPORTED or path.name.startswith("~$"):
            continue
        doc_id = path.stem
        if doc_id in used:                              # Doc1.pdf and Doc1.txt
            doc_id = f"{path.stem}_{path.suffix[1:].lower()}"
        try:
            if path.suffix.lower() == ".pdf":
                text, method, status, reason = read_pdf(path, lang, min_words)
            else:
                text = (read_docx(path) if path.suffix.lower() == ".docx"
                        else path.read_text(encoding="utf-8", errors="ignore"))
                method = path.suffix[1:].lower()
                status, reason = classify(analyze(text), lang, min_words)
        except Exception as exc:
            if verbose:
                print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            continue
        if status == "EMPTY":
            if verbose:
                print(f"  skipped {path.name}: no text (scanned? use extract_books.py with OCR)",
                      file=sys.stderr)
            continue
        if status != "OK" and verbose:
            print(f"  warning {path.name}: {status} - {reason}", file=sys.stderr)
        used.add(doc_id)
        corpus[doc_id] = text
        info.append((doc_id, count_words(text), method))
    return corpus, info


def _natural_key(name: str):
    """Doc2 before Doc10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


if __name__ == "__main__":
    c, inf = load_documents(sys.argv[1] if len(sys.argv) > 1 else "Document")
    for d, w, m in inf:
        print(f"{d:<20}{w:>8} words   via {m}")
