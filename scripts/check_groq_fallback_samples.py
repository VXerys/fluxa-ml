from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_backend.app.services.groq_fallback_service import (
    GroqFallbackConfig,
    GroqFallbackService,
)
from src.text_normalizer import normalize_text


class SampleGroqFallbackService(GroqFallbackService):
    def _request_correction(self, result: dict[str, Any]) -> dict[str, Any]:
        raw = result["transcript"]["raw"].lower()

        if "nasi padang" in raw:
            return {
                "title": "Nasi padang",
                "description": "Beli nasi padang",
                "type": "expense",
                "category": "Makan & Minum",
                "wallet": "BCA",
                "reason": "Food purchase with BCA wallet mention.",
            }

        if "jeruk nipis" in raw:
            return {
                "title": "Jeruk nipis",
                "description": "Beli jeruk nipis",
                "type": "expense",
                "category": "Belanja",
                "wallet": "BCA",
                "reason": "Grocery purchase with BCA wallet mention.",
            }

        if "transfer" in raw:
            return {
                "title": "Transfer BCA ke GoPay",
                "description": "Transfer BCA ke GoPay",
                "type": "transfer",
                "category": "Transfer",
                "wallet": "BCA",
                "reason": "Transfer phrase is explicit.",
            }

        raise RuntimeError("sample Groq failure")


def make_result(
    text: str,
    amount: int | None,
    transaction_type: str,
    category: str,
    wallet: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    return {
        "transcript": {
            "raw": text,
            "normalized": normalize_text(text),
            "language_hint": None,
            "confidence": None,
        },
        "transaction": {
            "type": transaction_type,
            "amount": amount,
            "category": category,
            "wallet": wallet,
            "title": title,
            "description": title,
            "currency": "IDR",
        },
        "classification": {
            "raw_type": transaction_type,
            "resolved_type": transaction_type,
            "category": category,
            "raw_type_confidence": 0.9,
            "category_confidence": 0.9,
        },
        "warnings": [],
    }


def main() -> None:
    service = SampleGroqFallbackService(
        GroqFallbackConfig(
            enabled=True,
            api_key="sample-key",
            base_url="https://example.test/openai/v1",
            model="sample-model",
            timeout_seconds=1,
        )
    )

    samples = [
        make_result(
            text="beli nasi padang rp21 ribu pakai bca",
            amount=21000,
            transaction_type="transfer",
            category="Transfer",
            title="Nasi padang",
        ),
        make_result(
            text="12 ribu jeruk nipis dengan bca",
            amount=12000,
            transaction_type="expense",
            category="Belanja",
            title="Jeruk nipis",
        ),
        make_result(
            text="transfer bca ke gopay 50 ribu",
            amount=50000,
            transaction_type="transfer",
            category="Transfer",
            title="Transfer bca gopay",
        ),
    ]

    for sample in samples:
        corrected = service.apply_if_needed(sample)
        transaction = corrected["transaction"]
        assert transaction["amount"] == sample["transaction"]["amount"]
        assert transaction["currency"] == "IDR"
        assert "groq_fallback_used" in corrected["warnings"]
        print(
            transaction["title"],
            transaction["type"],
            transaction["category"],
            transaction["wallet"],
            transaction["amount"],
        )

    failed = service.apply_if_needed(
        make_result(
            text="maser rasi padang rp21",
            amount=21000,
            transaction_type="expense",
            category="Belanja",
            title="Maser rasi padang",
        )
    )
    assert failed["transaction"]["amount"] == 21000
    assert "groq_fallback_failed" in failed["warnings"]
    print("failure fallback kept local result")


if __name__ == "__main__":
    main()
