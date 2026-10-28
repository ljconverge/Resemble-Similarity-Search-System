"""Tests for the brute-force tools, the two recreated modules, and the Flask app."""
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
DOCS = ROOT / "Document"


# ---------------------------------------------------------------- text_quality
def test_text_quality_ok_and_problems():
    from text_quality import analyze, classify
    prose = ("The quick brown fox jumps over the lazy dog and it is the best of all the "
             "animals in this story that we have read. ") * 10
    assert classify(analyze(prose), "en", 50)[0] == "OK"
    assert classify(analyze(""), "en", 1)[0] == "EMPTY"
    assert classify(analyze(prose), "en", 10_000)[0] == "TOO_SHORT"
    assert classify(analyze("12 3456 789 1011 1213 1415 " * 50), "en", 1)[0] == "GARBLED"
    assert classify(analyze("t h i s i s b r o k e n " * 50), "en", 1)[0] == "GARBLED"
    assert classify(analyze("�� text " * 50), "en", 1)[0] == "GARBLED"
    french = "le chat est sur la table et il ne veut pas descendre pour les enfants " * 20
    assert classify(analyze(french), "en", 1)[0] == "WRONG_LANGUAGE"
    assert classify(analyze(french), "any", 1)[0] == "OK"


# ---------------------------------------------------------------- document_loader
def _make_docx(path, text):
    xml = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + text +
           '</w:t></w:r></w:p></w:body></w:document>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def test_document_loader_all_types(tmp_path):
    from document_loader import load_documents
    (tmp_path / "a.txt").write_text("plain text file with enough words " * 5)
    _make_docx(tmp_path / "b.docx", "word document text with enough words " * 5)
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    (tmp_path / "skip.png").write_bytes(b"x")
    corpus, info = load_documents(tmp_path, verbose=False)
    assert set(corpus) == {"a", "b"}
    assert {d for d, _, _ in info} == {"a", "b"}


def test_document_loader_sample_pdfs():
    from document_loader import load_documents
    corpus, info = load_documents(DOCS, verbose=False)
    assert list(corpus) == [f"Doc{i}" for i in range(1, 11)]       # natural order
    assert dict((d, w) for d, w, _ in info)["Doc1"] == 923            # matches team report


# ---------------------------------------------------------------- scripts run end to end
@pytest.mark.parametrize("script", ["brute_force_k1.py", "brute_force_k3.py", "brute_force_k9.py"])
def test_brute_force_scripts(script, tmp_path):
    r = subprocess.run([sys.executable, script, str(DOCS), "--out", str(tmp_path)],
                       cwd=TOOLS, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "pairs with Jaccard >= 0.8 : 6 of 45" in r.stdout
    assert len(list(tmp_path.glob("*.csv"))) == 2


def test_compare_script(tmp_path):
    r = subprocess.run([sys.executable, "compare_k1_k3.py", str(DOCS), "--out", str(tmp_path)],
                       cwd=TOOLS, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "45-pair rankings" in r.stdout


def test_extract_books_script(tmp_path):
    r = subprocess.run([sys.executable, "extract_books.py", str(DOCS), str(tmp_path),
                        "--min-words", "100", "--no-ocr"],
                       cwd=TOOLS, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["documents"]) == 10


def test_generic_brute_force_module(tmp_path):
    import brute_force
    (tmp_path / "x.txt").write_text("alpha beta gamma delta epsilon zeta eta theta")
    (tmp_path / "y.txt").write_text("alpha beta gamma delta epsilon zeta eta iota")
    idx = brute_force.BruteForceIndex(str(tmp_path), shingle_k=2)
    assert idx.all_pairs(0.5) and idx.stats()["comparisons"] == 1


def test_engine_command_line():
    r = subprocess.run([sys.executable, "Algorithm/lsh_engine.py", "Document", "--mode", "char", "--k", "9"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "recall 1.0" in r.stdout


# ---------------------------------------------------------------- Flask app
@pytest.fixture()
def client():
    import app
    app.app.config["TESTING"] = True
    return app.app.test_client()


def _flash(resp):
    return re.findall(r'class="flash error">([^<]*)', resp.get_data(as_text=True))


def test_pages_without_analysis(client):
    for url in ("/", "/results", "/performance"):
        assert client.get(url).status_code == 200


@pytest.mark.parametrize("mode,k,method", [("word", "3", "fast"), ("char", "9", "fast"), ("char", "9", "notebook")])
def test_sample_analysis(client, mode, k, method):
    r = client.post("/", data={"source": "samples", "threshold": "80", "bands": "20",
                               "shingle_mode": mode, "shingle_size": k, "minhash_method": method})
    assert r.status_code == 302
    res = client.get("/results").get_data(as_text=True)
    perf = client.get("/performance").get_data(as_text=True)
    assert res.count("Near duplicate") >= 6
    assert "LSH vs brute force" in perf and "100.0" in perf


def test_missed_pairs_shown(client):
    client.post("/", data={"source": "samples", "threshold": "10", "bands": "20",
                           "shingle_mode": "word", "shingle_size": "1"})
    assert "Missed by LSH" in client.get("/results").get_data(as_text=True)


@pytest.mark.parametrize("form,msg", [
    ({"threshold": "abc"}, "whole numbers"),
    ({"bands": "7"}, "band settings"),
    ({"shingle_mode": "char", "shingle_size": "99"}, "1-30 characters"),
    ({"minhash_method": "magic"}, "valid MinHash"),
    ({"source": "upload"}, "choose at least one"),
])
def test_bad_input_messages(client, form, msg):
    data = {"source": "samples", "threshold": "80", "bands": "20", **form}
    r = client.post("/", data=data, follow_redirects=True)
    assert any(msg in m for m in _flash(r))


def test_upload_txt_and_pdf(client):
    files = [(io.BytesIO(b"machine learning is useful for healthcare and research"), "a.txt"),
             (io.BytesIO(b"machine learning is very useful for healthcare and research"), "b.txt"),
             (io.BytesIO(b"hello"), "c.docx"),
             (open(DOCS / "Doc1.pdf", "rb"), "Doc1.pdf")]
    r = client.post("/", data={"source": "upload", "threshold": "50", "bands": "20",
                               "shingle_mode": "word", "shingle_size": "2", "documents": files},
                    content_type="multipart/form-data", follow_redirects=True)
    page = r.get_data(as_text=True)
    assert "66.67%" in page and "c.docx: skipped" in page


def test_single_document_upload(client):
    r = client.post("/", data={"source": "upload", "threshold": "80", "bands": "20",
                               "documents": [(io.BytesIO(b"just one document"), "one.txt")]},
                    content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200 and client.get("/performance").status_code == 200
