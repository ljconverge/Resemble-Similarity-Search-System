"""
text_quality.py
===============
Checks whether extracted text is real, readable text. Used by extract_books.py
and document_loader.py to decide if a PDF extraction method worked or if the
next method (or OCR) should be tried.

    m = analyze(text)                       -> dict of measurements
    status, reason = classify(m, "en", 100) -> ("OK", "...") or a problem status

Statuses (extract_books.py retries on the first four):
    EMPTY           no words at all
    TOO_SHORT       fewer than min_words words
    GARBLED         mostly digits / symbols / broken characters
    UNRECOGNISED    real-looking words but no known language detected
    WRONG_LANGUAGE  readable, but not the requested language
    OK              usable

Standard library only. The language check counts very common words
("stopwords"); it is a heuristic, not a full language detector.
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "en": "the of and to in a is that for it as was with be by on not he this are or "
          "his from at which but have an they you we can all their has one there been".split(),
    "fr": "le la les de des et un une est que en du dans qui pour pas sur au par "
          "ne se ce il elle sont avec plus".split(),
    "de": "der die das und ist in zu den von nicht mit sich des auf ein eine dem "
          "auch es an als werden aus".split(),
    "es": "el la de que y en los del se las por un una para con no es al lo como "
          "más pero sus le ya".split(),
    "it": "il di che la e per un in non una sono del della le con si da gli è "
          "nel al anche come".split(),
    "pt": "o de a que e do da em um para é com não uma os no se na por mais as "
          "dos como mas".split(),
    "nl": "de het een en van is dat in te op niet zijn met voor die er aan ook "
          "als maar bij".split(),
}
_STOP = {lang: set(words) for lang, words in STOPWORDS.items()}
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

# Thresholds (tuned so normal prose passes and digit/symbol soup fails)
MIN_LETTER_SHARE = 0.55      # letters / non-space characters
MAX_DIGIT_WORD_SHARE = 0.40  # words that are pure numbers
MAX_ONE_CHAR_SHARE = 0.35    # single-character words ("t h i s  i s" broken spacing)
MAX_BAD_CHAR_SHARE = 0.02    # replacement chars / private-use glyphs
AVG_WORD_LEN = (2.0, 15.0)
MIN_STOPWORD_SHARE = 0.05    # share of words that are stopwords of the best language


def analyze(text: str) -> dict:
    """Measure the text. Never raises; empty input gives zeros."""
    text = text or ""
    non_space = [c for c in text if not c.isspace()]
    n_chars = len(non_space)
    letters = sum(c.isalpha() for c in non_space)
    bad = sum(c == "�" or unicodedata.category(c) in ("Co", "Cs", "Cn")
              for c in non_space)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n_words = len(words)
    digit_words = sum(w.isdigit() for w in words)
    one_char = sum(len(w) == 1 and not w.isdigit() for w in words)
    alpha_words = [w for w in words if not w.isdigit()]

    lang_scores = {}
    if alpha_words:
        for lang, stop in _STOP.items():
            lang_scores[lang] = round(sum(w in stop for w in alpha_words) / len(alpha_words), 4)
    best_lang, best_score = (max(lang_scores.items(), key=lambda kv: kv[1])
                             if lang_scores else (None, 0.0))
    if best_score < MIN_STOPWORD_SHARE:
        best_lang = None

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "letter_share": round(letters / n_chars, 4) if n_chars else 0.0,
        "digit_word_share": round(digit_words / n_words, 4) if n_words else 0.0,
        "one_char_share": round(one_char / n_words, 4) if n_words else 0.0,
        "bad_char_share": round(bad / n_chars, 4) if n_chars else 0.0,
        "avg_word_len": round(sum(map(len, alpha_words)) / len(alpha_words), 3) if alpha_words else 0.0,
        "lang_scores": lang_scores,
        "best_lang": best_lang,
        "best_lang_score": best_score,
    }


def classify(m: dict, lang: str = "en", min_words: int = 1000) -> tuple[str, str]:
    """Turn measurements into (status, human-readable reason)."""
    if m["n_words"] == 0:
        return "EMPTY", "no words were extracted"
    if m["bad_char_share"] > MAX_BAD_CHAR_SHARE:
        return "GARBLED", f"{m['bad_char_share']:.1%} unreadable characters (broken font encoding)"
    if m["letter_share"] < MIN_LETTER_SHARE:
        return "GARBLED", f"only {m['letter_share']:.0%} of characters are letters"
    if m["digit_word_share"] > MAX_DIGIT_WORD_SHARE:
        return "GARBLED", f"{m['digit_word_share']:.0%} of words are numbers"
    if m["one_char_share"] > MAX_ONE_CHAR_SHARE:
        return "GARBLED", f"{m['one_char_share']:.0%} of words are single letters (broken spacing)"
    lo, hi = AVG_WORD_LEN
    if not lo <= m["avg_word_len"] <= hi:
        return "GARBLED", f"average word length {m['avg_word_len']} is not like real text"
    if m["n_words"] < min_words:
        return "TOO_SHORT", f"{m['n_words']} words (minimum {min_words})"
    if lang != "any":
        if m["best_lang"] is None:
            return "UNRECOGNISED", "no known language detected"
        if m["best_lang"] != lang:
            return "WRONG_LANGUAGE", f"looks like '{m['best_lang']}', expected '{lang}'"
    return "OK", f"{m['n_words']} words, language {m['best_lang'] or 'not checked'}"


if __name__ == "__main__":
    import sys
    from pathlib import Path
    for p in sys.argv[1:]:
        meas = analyze(Path(p).read_text(encoding="utf-8", errors="ignore"))
        print(p, classify(meas, "en", 1), meas)
