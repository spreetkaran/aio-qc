# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
OCR languages: one friendly list, mapped to each engine's own codes (None = the engine cannot read it).

EasyOCR and Tesseract read several languages in ONE pass. PaddleOCR (and RapidOCR's extra models) use one model per
language — with several languages AIO QC either keeps the most confident read of all models ("best"), or reads with
the first language and tries the others only when unsure ("fallback", faster).
"""
from __future__ import annotations

# key, name, paddle, easyocr, tesseract, rapidocr (3.x LangRec name)
LANGS = [
    ("en", "English", "en", "en", "eng", "en"),
    ("latin", "Latin script (European languages)", "latin", None, "eng+fra+deu+spa+ita+por", "latin"),
    ("fr", "French", "fr", "fr", "fra", "latin"),
    ("de", "German", "german", "de", "deu", "latin"),
    ("es", "Spanish", "es", "es", "spa", "latin"),
    ("it", "Italian", "it", "it", "ita", "latin"),
    ("pt", "Portuguese", "pt", "pt", "por", "latin"),
    ("ru", "Russian", "ru", "ru", "rus", "cyrillic"),
    ("cyrillic", "Cyrillic script (other)", "cyrillic", None, "rus+ukr+bul+srp", "cyrillic"),
    ("ar", "Arabic", "ar", "ar", "ara", "arabic"),
    ("fa", "Persian", "fa", "fa", "fas", "arabic"),
    ("ur", "Urdu", "ur", "ur", "urd", "arabic"),
    ("hi", "Hindi", "hi", "hi", "hin", "devanagari"),
    ("mr", "Marathi", "mr", "mr", "mar", "devanagari"),
    ("ne", "Nepali", "ne", "ne", "nep", "devanagari"),
    ("bn", "Bengali", None, "bn", "ben", None),
    ("ta", "Tamil", "ta", "ta", "tam", "ta"),
    ("te", "Telugu", "te", "te", "tel", "te"),
    ("kn", "Kannada", "ka", "kn", "kan", "ka"),
    ("th", "Thai", "th", "th", "tha", "th"),
    ("ch", "Chinese (simplified) + English", "ch", "ch_sim", "chi_sim+eng", "ch"),
    ("cht", "Chinese (traditional)", "chinese_cht", "ch_tra", "chi_tra", "chinese_cht"),
    ("ja", "Japanese", "japan", "ja", "jpn", "japan"),
    ("ko", "Korean", "korean", "ko", "kor", "korean"),
]
NAMES = {k: n for k, n, *_ in LANGS}
_COL = {"paddleocr": 2, "easyocr": 3, "tesseract": 4, "rapidocr": 5}
NATIVE_MULTI = {"easyocr", "tesseract"}           # several languages in one pass


def code(engine: str, key: str) -> str | None:
    row = next((r for r in LANGS if r[0] == key), None)
    return row[_COL[engine]] if row and engine in _COL else None


def resolve(engine: str, keys: list[str]) -> tuple[list[str], list[str]]:
    """(engine codes in order, unsupported language names)."""
    codes, missing = [], []
    for k in keys:
        c = code(engine, k)
        if c is None:
            missing.append(NAMES.get(k, k))
        elif c not in codes:
            codes.append(c)
    return codes, missing


def describe(keys: list[str]) -> str:
    return ", ".join(NAMES.get(k, k) for k in keys) or "English"
