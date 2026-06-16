from __future__ import annotations

import re
from typing import Optional

from src.text_normalizer import normalize_text


_WALLET_WORDS = (
    "bca",
    "dana",
    "gopay",
    "go pay",
    "ovo",
    "shopeepay",
    "shopee pay",
    "cash",
    "tunai",
)

_ACTION_WORDS = (
    "beli",
    "bayar",
    "jajan",
    "buat",
    "untuk",
    "pakai",
    "pake",
    "via",
    "dengan",
    "dari",
    "ke",
)

_NUMBER_WORDS = (
    "nol",
    "satu",
    "sa",
    "hiji",
    "dua",
    "tilu",
    "tiga",
    "opat",
    "empat",
    "lima",
    "genep",
    "enam",
    "tujuh",
    "dalapan",
    "delapan",
    "salapan",
    "sembilan",
    "sepuluh",
    "sapuluh",
    "sebelas",
    "sabelas",
    "belas",
    "puluh",
    "ratus",
    "ribu",
    "rebu",
    "rb",
    "juta",
    "jt",
)


def _clean_title(value: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", value).strip(" -_/")
    if not cleaned:
        return None

    tokens = [token for token in cleaned.split() if token not in _ACTION_WORDS]
    cleaned = " ".join(tokens).strip()
    if not cleaned:
        return None

    return cleaned[:1].upper() + cleaned[1:]


def parse_transaction_title(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None

    value = normalized
    for wallet in _WALLET_WORDS:
        value = re.sub(rf"\b(pakai|pake|via|dengan|dari|ke)?\s*{re.escape(wallet)}\b", " ", value)

    value = re.sub(r"\brp\s*\d+(?:[\s.,]\d{3})*\b", " ", value)
    value = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(ribu|rebu|rb|k|juta|jt|m)\b", " ", value)
    value = re.sub(r"\b\d{4,}\b", " ", value)

    number_pattern = "|".join(re.escape(word) for word in _NUMBER_WORDS)
    value = re.sub(rf"\b(?:{number_pattern})(?:\s+(?:{number_pattern}))*\b", " ", value)

    return _clean_title(value)
