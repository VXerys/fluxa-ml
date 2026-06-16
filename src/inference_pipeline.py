"""End-to-end inference pipeline for Fluxa voice transaction parser.

Pipeline:
text
-> type classifier
-> category classifier
-> transaction type resolver
-> amount parser
-> transaction JSON draft

The resolver is important because some category labels imply a fixed transaction type:
- Gaji, Freelance -> income
- Transfer -> transfer
- Other categories -> expense or model prediction
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

from src.amount_parser import parse_amount
from src.text_normalizer import normalize_text


INCOME_CATEGORIES = {"Gaji", "Freelance"}
TRANSFER_CATEGORIES = {"Transfer"}

CATEGORY_KEYWORDS = {
    "Tagihan & Utilitas": [
        "wifi",
        "wi-fi",
        "internet",
        "kuota",
        "listrik",
        "token",
        "pulsa",
        "kos",
        "kontrakan",
    ],
    "Transportasi": [
        "bensin",
        "parkir",
        "ojek",
        "ojol",
        "angkot",
        "grab",
        "gojek",
        "transport",
    ],
    "Makan & Minum": [
        "kopi",
        "nasi",
        "cilok",
        "seblak",
        "makan",
        "jajan",
        "sangu",
        "minum",
    ],
    "Kesehatan": [
        "obat",
        "apotik",
        "apotek",
        "dokter",
        "klinik",
        "vitamin",
    ],
}


@dataclass
class TransactionDraft:
    type: str
    amount: Optional[int]
    category: str
    wallet: Optional[str]
    description: Optional[str]
    currency: str = "IDR"


@dataclass
class ClassificationResult:
    raw_type: str
    resolved_type: str
    category: str
    raw_type_confidence: Optional[float] = None
    category_confidence: Optional[float] = None


@dataclass
class VoiceTransactionResult:
    transcript: dict[str, Any]
    transaction: TransactionDraft
    classification: ClassificationResult
    warnings: list[str]


def resolve_category_by_keywords(text: str, pred_category: str) -> tuple[str, str | None]:
    """Resolve obvious category conflicts using finance-domain keywords.

    Example:
    text = "mayar wifi rp 300000"
    model category = Gaji
    resolved category = Tagihan & Utilitas
    """
    lowered = text.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            if pred_category != category:
                return category, f"category_resolved_by_keyword:{pred_category}->{category}"
            return pred_category, None

    return pred_category, None


def resolve_transaction_type(pred_type: str, pred_category: str) -> str:
    """Resolve inconsistent model outputs using finance domain rules.

    Rules:
    - Gaji, Freelance -> income
    - Transfer -> transfer
    - All other categories -> expense
    """
    if pred_category in INCOME_CATEGORIES:
        return "income"

    if pred_category in TRANSFER_CATEGORIES:
        return "transfer"

    return "expense"


def _prediction_confidence(model: Any, text: str, predicted_label: str) -> Optional[float]:
    """Return classifier confidence when the sklearn model exposes probabilities."""
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return None

    probabilities = model.predict_proba([text])[0]
    class_labels = [str(label) for label in model.classes_]

    if predicted_label not in class_labels:
        return None

    return float(probabilities[class_labels.index(predicted_label)])


def infer_transaction(
    text: str,
    type_model: Any,
    category_model: Any,
    wallet_model: Any | None = None,
) -> dict[str, Any]:
    """Run complete transaction inference from raw text.

    Args:
        text: Raw user text / Whisper transcript.
        type_model: Trained type classifier.
        category_model: Trained category classifier.
        wallet_model: Optional wallet classifier.

    Returns:
        Dict compatible with Fluxa backend response draft.
    """
    normalized = normalize_text(text)

    raw_pred_type = str(type_model.predict([text])[0])
    raw_pred_category = str(category_model.predict([text])[0])
    raw_type_confidence = _prediction_confidence(type_model, text, raw_pred_type)
    category_confidence = _prediction_confidence(category_model, text, raw_pred_category)
    pred_category, category_warning = resolve_category_by_keywords(
        normalized,
        raw_pred_category,
    )
    resolved_type = resolve_transaction_type(raw_pred_type, pred_category)

    pred_wallet: Optional[str] = None
    if wallet_model is not None:
        wallet_prediction = wallet_model.predict([text])[0]
        if wallet_prediction is not None and str(wallet_prediction).lower() not in {"none", "null", "nan"}:
            pred_wallet = str(wallet_prediction)

    pred_amount = parse_amount(text)

    warnings: list[str] = []

    if category_warning is not None:
        warnings.append(category_warning)

    if pred_amount is None:
        warnings.append("amount_not_detected")

    if pred_amount is not None and pred_amount > 50_000_000:
        warnings.append("amount_unusually_large")

    if raw_pred_type != resolved_type:
        warnings.append(
            f"type_resolved_by_category:{raw_pred_type}->{resolved_type}"
        )

    transaction = TransactionDraft(
        type=resolved_type,
        amount=pred_amount,
        category=pred_category,
        wallet=pred_wallet,
        description=None,
        currency="IDR",
    )

    classification = ClassificationResult(
        raw_type=raw_pred_type,
        resolved_type=resolved_type,
        category=pred_category,
        raw_type_confidence=raw_type_confidence,
        category_confidence=category_confidence,
    )

    return {
        "transcript": {
            "raw": text,
            "normalized": normalized,
            "language_hint": None,
            "confidence": None,
        },
        "transaction": asdict(transaction),
        "classification": asdict(classification),
        "warnings": warnings,
    }
