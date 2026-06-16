from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from app.services.groq_fallback_service import groq_fallback_service
from app.services.title_parser_service import parse_transaction_title
from src.inference_pipeline import infer_transaction


class TransactionParserService:
    def __init__(self) -> None:
        # File ini ada di:
        # ai_backend/app/services/transaction_parser_service.py
        # parents[3] = root project fluxa_voice_ai_training_pipeline
        self.project_root = Path(__file__).resolve().parents[2]

        self.type_model_path = (
            self.project_root / "models" / "baseline_type_classifier.joblib"
        )
        self.category_model_path = (
            self.project_root / "models" / "baseline_category_classifier.joblib"
        )

        if not self.type_model_path.exists():
            raise FileNotFoundError(f"Type model not found: {self.type_model_path}")

        if not self.category_model_path.exists():
            raise FileNotFoundError(
                f"Category model not found: {self.category_model_path}"
            )

        self.type_model: Any = joblib.load(self.type_model_path)
        self.category_model: Any = joblib.load(self.category_model_path)

    def parse_text(self, text: str) -> dict[str, Any]:
        result = infer_transaction(
            text=text,
            type_model=self.type_model,
            category_model=self.category_model,
        )

        self._apply_local_title(result)
        return groq_fallback_service.apply_if_needed(result)

    def _apply_local_title(self, result: dict[str, Any]) -> None:
        transaction = result.setdefault("transaction", {})
        title = parse_transaction_title(result.get("transcript", {}).get("raw", ""))

        if title is not None:
            transaction.setdefault("title", title)
            transaction["description"] = transaction.get("description") or title


transaction_parser_service = TransactionParserService()
