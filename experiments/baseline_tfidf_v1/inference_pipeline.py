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


@dataclass
class VoiceTransactionResult:
    transcript: dict[str, Any]
    transaction: TransactionDraft
    classification: ClassificationResult
    warnings: list[str]


def resolve_transaction_type(pred_type: str, pred_category: str) -> str:
    """Resolve inconsistent model outputs using finance domain rules.

    Example:
    raw type = expense, category = Gaji
    resolved type = income
    """
    if pred_category in INCOME_CATEGORIES:
        return "income"

    if pred_category in TRANSFER_CATEGORIES:
        return "transfer"

    return pred_type


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
    pred_category = str(category_model.predict([text])[0])
    resolved_type = resolve_transaction_type(raw_pred_type, pred_category)

    pred_wallet: Optional[str] = None
    if wallet_model is not None:
        wallet_prediction = wallet_model.predict([text])[0]
        if wallet_prediction is not None and str(wallet_prediction).lower() not in {"none", "null", "nan"}:
            pred_wallet = str(wallet_prediction)

    pred_amount = parse_amount(text)

    warnings: list[str] = []
    if pred_amount is None:
        warnings.append("amount_not_detected")

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