from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional


ALLOWED_CATEGORIES = {
    "Makan & Minum",
    "Transportasi",
    "Belanja",
    "Tagihan & Utilitas",
    "Hiburan",
    "Kesehatan",
    "Gaji",
    "Freelance",
    "Transfer",
}

ALLOWED_TYPES = {"expense", "income", "transfer"}

WALLET_ALIASES = {
    "bca": "BCA",
    "dana": "DANA",
    "gopay": "GoPay",
    "go pay": "GoPay",
    "ovo": "OVO",
    "shopeepay": "ShopeePay",
    "shopee pay": "ShopeePay",
    "cash": "Cash",
    "tunai": "Cash",
}

INCOME_CATEGORIES = {"Gaji", "Freelance"}
EXPENSE_CATEGORIES = ALLOWED_CATEGORIES - INCOME_CATEGORIES - {"Transfer"}

TRANSFER_PHRASES = (
    "transfer",
    "tf",
    "kirim",
    "pindah saldo",
    "top up",
    "topup",
)

WALLET_WORDS = tuple(WALLET_ALIASES)
SUSPICIOUS_STT_WORDS = (
    "maser",
    "rasi",
    "repi",
    "repe",
    "rupiya",
    "rp21",
    "rp 21",
)

AMOUNT_NOISE_PATTERN = re.compile(
    r"\b(?:rp\s*\d+|\d+(?:[\.,]\d+)?\s*(?:ribu|rebu|rb|k|juta|jt|m)|\d{4,}|ribu|rebu)\b",
    re.IGNORECASE,
)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


@dataclass(frozen=True)
class GroqFallbackConfig:
    enabled: bool
    api_key: Optional[str]
    base_url: str
    model: Optional[str]
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "GroqFallbackConfig":
        return cls(
            enabled=_env_bool("ENABLE_GROQ_FALLBACK", default=False),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
            model=os.getenv("GROQ_MODEL"),
            timeout_seconds=_env_float("GROQ_TIMEOUT_SECONDS", default=8.0),
        )


class GroqFallbackService:
    def __init__(self, config: GroqFallbackConfig | None = None) -> None:
        self.config = config or GroqFallbackConfig.from_env()

    def apply_if_needed(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enabled or not self.should_attempt(result):
            return result

        updated = deepcopy(result)
        warnings = updated.setdefault("warnings", [])

        try:
            correction = self._request_correction(updated)
            changed = self._apply_correction(updated, correction)
        except Exception:
            if "groq_fallback_failed" not in warnings:
                warnings.append("groq_fallback_failed")
            return updated

        if changed and "groq_fallback_used" not in warnings:
            warnings.append("groq_fallback_used")

        return updated

    def should_attempt(self, result: dict[str, Any]) -> bool:
        transcript = result.get("transcript") or {}
        transaction = result.get("transaction") or {}
        classification = result.get("classification") or {}
        warnings = [
            warning
            for warning in result.get("warnings", [])
            if not str(warning).startswith("groq_fallback_")
        ]

        raw_text = str(transcript.get("raw") or "")
        normalized_text = str(transcript.get("normalized") or raw_text).lower()
        title = str(transaction.get("title") or transaction.get("description") or "")
        category = str(transaction.get("category") or "")
        transaction_type = str(transaction.get("type") or "")
        wallet = transaction.get("wallet")

        if warnings:
            return True

        if self._title_is_uncertain(title):
            return True

        if category == "Transfer" and not self._is_transfer_phrase(normalized_text):
            return True

        if (
            category == "Transfer"
            and transaction_type in {"expense", "income"}
            and not self._is_transfer_phrase(normalized_text)
        ):
            return True

        if wallet is None and self._contains_wallet_word(normalized_text):
            return True

        if self._looks_noisy(raw_text, normalized_text):
            return True

        if self._low_confidence(classification):
            return True

        return False

    def _request_correction(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise RuntimeError("GROQ_API_KEY is required when Groq fallback is enabled.")

        if not self.config.model:
            raise RuntimeError("GROQ_MODEL is required when Groq fallback is enabled.")

        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You correct Indonesian/Sundanese finance transcript parsing. "
                        "Return strict JSON only. Correct only title, description, "
                        "category, wallet, and type when clearly supported by the input. "
                        "Do not change amount. Do not change currency."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(self._build_prompt_input(result), ensure_ascii=False),
                },
            ],
        }

        request = urllib.request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Groq fallback request failed.") from exc

        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not isinstance(content, str):
            raise ValueError("Groq fallback returned an empty response.")

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Groq fallback response must be a JSON object.")

        return parsed

    def _build_prompt_input(self, result: dict[str, Any]) -> dict[str, Any]:
        transcript = result.get("transcript") or {}
        transaction = result.get("transaction") or {}

        return {
            "raw_transcript": transcript.get("raw"),
            "normalized_transcript": transcript.get("normalized"),
            "local_amount": transaction.get("amount"),
            "local_type": transaction.get("type"),
            "local_category": transaction.get("category"),
            "local_wallet": transaction.get("wallet"),
            "local_title": transaction.get("title"),
            "local_description": transaction.get("description"),
            "warnings": result.get("warnings", []),
            "allowed_categories": sorted(ALLOWED_CATEGORIES),
            "allowed_types": sorted(ALLOWED_TYPES),
            "allowed_wallets": sorted(set(WALLET_ALIASES.values())),
            "required_response_shape": {
                "title": "string or null",
                "description": "string or null",
                "type": "expense|income|transfer|null",
                "category": "allowed category or null",
                "wallet": "allowed wallet or null",
                "reason": "short reason",
            },
        }

    def _apply_correction(
        self,
        result: dict[str, Any],
        correction: dict[str, Any],
    ) -> bool:
        transaction = result.setdefault("transaction", {})
        changed = False

        title = self._valid_text(correction.get("title"))
        if title is not None and title != transaction.get("title"):
            transaction["title"] = title
            changed = True

        description = self._valid_text(correction.get("description"))
        if description is not None and description != transaction.get("description"):
            transaction["description"] = description
            changed = True

        category = self._valid_category(correction.get("category"))
        if category is not None and category != transaction.get("category"):
            transaction["category"] = category
            result.setdefault("classification", {})["category"] = category
            changed = True

        wallet = self._valid_wallet(correction.get("wallet"))
        if wallet is not None and wallet != transaction.get("wallet"):
            transaction["wallet"] = wallet
            changed = True

        corrected_type = self._valid_type(correction.get("type"))
        safe_type = self._safe_corrected_type(
            result=result,
            corrected_type=corrected_type,
            corrected_category=category,
        )
        if safe_type is not None and safe_type != transaction.get("type"):
            transaction["type"] = safe_type
            result.setdefault("classification", {})["resolved_type"] = safe_type
            changed = True

        return changed

    def _safe_corrected_type(
        self,
        result: dict[str, Any],
        corrected_type: Optional[str],
        corrected_category: Optional[str],
    ) -> Optional[str]:
        transaction = result.get("transaction") or {}
        transcript = result.get("transcript") or {}
        normalized_text = str(transcript.get("normalized") or transcript.get("raw") or "").lower()
        current_type = str(transaction.get("type") or "")
        category = corrected_category or str(transaction.get("category") or "")

        implied_type = self._type_implied_by_category(category)
        if implied_type is not None and implied_type != current_type:
            return implied_type

        if corrected_type is None or corrected_type == current_type:
            return None

        if corrected_type == "transfer":
            return corrected_type if self._is_transfer_phrase(normalized_text) else None

        if current_type == "transfer" and category != "Transfer":
            return corrected_type

        if category in EXPENSE_CATEGORIES and corrected_type == "expense":
            return corrected_type

        if category in INCOME_CATEGORIES and corrected_type == "income":
            return corrected_type

        return None

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _title_is_uncertain(self, title: str) -> bool:
        stripped = title.strip()
        return len(stripped) < 3 or AMOUNT_NOISE_PATTERN.search(stripped) is not None

    def _looks_noisy(self, raw_text: str, normalized_text: str) -> bool:
        lowered_raw = raw_text.lower()
        if any(word in lowered_raw for word in SUSPICIOUS_STT_WORDS):
            return True
        return bool(re.search(r"\b[a-z]{1,2}\d+[a-z]*\b", normalized_text))

    def _low_confidence(self, classification: dict[str, Any]) -> bool:
        for key in ("raw_type_confidence", "category_confidence"):
            value = classification.get(key)
            if isinstance(value, (float, int)) and float(value) < 0.55:
                return True
        return False

    def _is_transfer_phrase(self, value: str) -> bool:
        return any(
            re.search(rf"\b{re.escape(phrase)}\b", value) is not None
            for phrase in TRANSFER_PHRASES
        )

    def _contains_wallet_word(self, value: str) -> bool:
        return any(
            re.search(rf"\b{re.escape(word)}\b", value) is not None
            for word in WALLET_WORDS
        )

    def _valid_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = re.sub(r"\s+", " ", value).strip()
        if len(stripped) < 3 or AMOUNT_NOISE_PATTERN.search(stripped):
            return None
        return stripped

    def _valid_category(self, value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value in ALLOWED_CATEGORIES else None

    def _valid_type(self, value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value in ALLOWED_TYPES else None

    def _valid_wallet(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        return WALLET_ALIASES.get(value.strip().lower())

    def _type_implied_by_category(self, category: str) -> Optional[str]:
        if category == "Transfer":
            return "transfer"
        if category in INCOME_CATEGORIES:
            return "income"
        if category in EXPENSE_CATEGORIES:
            return "expense"
        return None


groq_fallback_service = GroqFallbackService()
