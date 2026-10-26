# Text Extraction from PDF Books: Methods, Our Choice, and Complexity

Project: Near-Duplicate and Similarity Search (LSH vs Brute Force)
Scope: how the book PDFs are turned into plain text before shingling.

---

## 1. Why extraction matters for this project

Every later stage (shingling, Jaccard, MinHash, LSH) works on the extracted words. Two things matter:

1. **Consistency.** Brute force and LSH must see exactly the same text, otherwise the accuracy comparison is unfair.
2. **Cost.** Books are long (hundreds of pages). Extraction is a one-time cost for the corpus, but it is also paid on every uploaded query document in the Flask app.

Shingling only needs the *words* to come out consistently. Perfect layout, tables, or equation rendering are not required.

### Notation

| Symbol | Meaning |
|---|---|
| P | number of pages in one book |
| C | number of characters in one book |
| n | number of text elements (glyphs, words, lines) on one page |
| B | number of books |
| K | number of CPU cores |

---

## 2. The extraction methods

### 2.1 Text-layer extractors (the PDF already contains text)

**PyMuPDF (`fitz`)**
C library (MuPDF) with a Python binding. Parses each page's content stream and returns its text directly. Pages can be loaded one at a time.

**pdftotext (Poppler)**
Command-line tool written in C++. Writes the whole book's text to a file or stdout. Pages are separated by a form-feed character.

**pypdf**
Pure-Python library. Parses the PDF object tree and decodes text operators in Python, so every operation runs in the interpreter.

### 2.2 Layout-aware extractors (text layer, plus reconstruction of reading order)

**pdfminer.six**
Pure Python. Builds character objects, then groups them into lines, text boxes, and columns (layout analysis) before emitting text.

**pdfplumber**
Built on pdfminer. Adds a per-page object model (words, lines, tables), which is useful for table extraction but keeps more objects in memory.

### 2.3 OCR (the PDF is scanned images, no text layer)

**ocrmypdf + Tesseract**
Renders each page to an image, runs character recognition, and writes a text layer. Needed only when the extractors above return little or no text.

### 2.4 ML document converters (best for equations and complex layout)

**Marker, Nougat, Docling**
Neural models that read the page visually and produce Markdown or LaTeX-like text. Best output for maths and physics books, but by far the heaviest to run.

---

## 3. Time and space complexity

All text-layer methods are linear in the size of the book, so the **big-O class is the same**. The real difference is the constant factor, plus the super-linear layout step in pdfminer and the per-pixel or per-token cost of OCR and ML.

| Method | Time (per book) | Space (per book) | Constant factor / speed |
|---|---|---|---|
| **PyMuPDF** | O(P + C) | O(n) per page if iterated page by page, plus O(C) for the collected text | Fastest in Python (C core) |
| **pdftotext** | O(P + C) | O(C) for the captured output (the tool itself streams page by page) | About as fast as PyMuPDF. Adds a small process-start cost per book |
| **pypdf** | O(P + C) | O(file object tree) plus O(C) | Several times slower (pure Python) |
| **pdfminer.six** | Per page, layout grouping is super-linear in n (worst case roughly O(n²) in text elements). Total roughly O(P · n²) worst case | O(n) layout objects per page | Slow; the layout step dominates |
| **pdfplumber** | Same class as pdfminer | Higher than pdfminer (extra object model per page) | Slowest of the text-layer group |
| **OCR (Tesseract)** | O(P · W·H), proportional to pixels rendered and recognised per page | One page image in memory, for example about 8.7 million pixels for an A4 page at 300 dpi | Seconds per page |
| **ML converters** | O(P · cost of one model pass); for transformer decoders, attention can be quadratic in tokens per page | GBs of model weights plus activations (GPU memory) | Often minutes per book |

Notes on the table:

- "Time" counts work to extract one book. For B books run on K cores, wall-clock time is roughly (total work) / K, because books are independent.
- Space for the parallel run is roughly K × (memory for the largest book being processed at once).
- The pdfminer quadratic bound is a worst case that comes from grouping text elements by proximity. Real pages are usually far from the worst case, but the layout step is still why it is slower.

### Measured sanity check (small test, not a rigorous benchmark)

On a generated 12-page test PDF, run on one CPU core:

| Backend | Time |
|---|---|
| pdftotext | 0.016 s |
| pypdf | 0.117 s |

pypdf was about 7 times slower than pdftotext. PyMuPDF was not installed in the test environment, so it was not timed. It is generally reported to be in the same speed class as pdftotext, so measure it on one real book before relying on that. A 12-page synthetic PDF is tiny, so treat the ratio as indicative only and re-measure on real books.

---

## 4. Our choice and why

**Primary: PyMuPDF. Fallbacks: pdftotext, then pypdf. Run books in parallel (one per CPU core).**

The script `extract_books.py` picks the fastest backend that is available.

Reasons:

1. **Speed.** C-based extraction has the lowest constant factor. For hundreds of pages per book and dozens of books, this keeps the whole corpus extraction short.
2. **Same big-O, much smaller constant.** All text-layer methods are linear, so there is no complexity penalty for choosing the fast one. The slower methods give nothing we need in return.
3. **Layout accuracy is not needed.** Shingles are k-word windows. Small reading-order differences at column breaks affect a tiny fraction of shingles, and they affect brute force and LSH equally.
4. **Equations do not need to be correct.** Garbled math symbols appear identically for every copy of a book, so near-duplicate detection is unaffected. This removes the main reason to use an ML converter.
5. **Low setup cost.** PyMuPDF installs with one `pip` command. pdftotext and pypdf are fallbacks if installation is a problem.
6. **Easy parallelism.** Books are independent, so a process pool over files gives close to a K-fold speedup with no shared state.
7. **Extraction is not part of the benchmark.** The LSH vs brute force comparison measures search time, not extraction. Extraction only affects the upload path, where a fast single-book extraction keeps the Flask page responsive.

### When we would switch

| Situation | Switch to |
|---|---|
| Extracted text is empty or very short (scanned book) | OCR (ocrmypdf + Tesseract) for that book only |
| Two-column pages produce badly interleaved text | pdfminer.six or pdfplumber for that book only |
| The project later needs readable equations or tables | An ML converter (Marker, Docling, Nougat) |

---

## 5. Cleanup applied after extraction

Extraction output contains page headers, footers, and page numbers, which repeat on every page. Left in, they create shared shingles between unrelated books and add noise.

**Method:** treat a line as running text if, after replacing digits with `#`, it appears on at least max(3, 30% of pages) pages and is shorter than 80 characters. Remove those lines.

| | Complexity |
|---|---|
| Time | O(total lines in the book), one counting pass and one filtering pass |
| Space | O(number of distinct lines) for the counter |

Limitation: a genuine line that repeats very often (for example a repeated formula label) could be removed. Check one extracted book by eye before running the full corpus.

Further normalisation (lowercasing, Unicode NFKC to fix ligatures such as "ﬁ", removing hyphenation at line breaks, collapsing whitespace) happens inside `brute_force.py`. LSH must import the same functions, so both methods see identical text.

---

## 6. How extraction connects to the rest of the pipeline

```
books/<category>/*.pdf
        |
        |  extract_books.py  (PyMuPDF / pdftotext / pypdf, parallel)
        v
extracted/<category>/*.txt  +  extracted/manifest.json
        |
        |  BruteForceIndex("extracted/manifest.json")   (shared normalise + shingling)
        v
shingle sets  ->  brute force (exact Jaccard)
              ->  MinHash -> bands -> buckets -> candidates (LSH)
```

The `manifest.json` format is `{"documents": [{"doc_id", "category", "text_path", "pages", "words"}, ...]}`. If the extraction team uses a different format, only the loader functions in `brute_force.py` need to change.
