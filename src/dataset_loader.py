"""Dataset loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["text", "type"]
VALID_TYPES = {"expense", "income", "transfer"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return pd.DataFrame(rows)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_reasons: list[list[str]] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        text = row.get("text")
        tx_type = row.get("type")
        amount = row.get("amount") if "amount" in df.columns else None

        if text is None or str(text).strip() == "":
            reasons.append("empty_text")
        if tx_type not in VALID_TYPES:
            reasons.append("invalid_type")
        if amount is not None and str(amount).strip() != "" and not pd.isna(amount):
            try:
                int(float(amount))
            except (ValueError, TypeError):
                reasons.append("invalid_amount")
        invalid_reasons.append(reasons)

    working = df.copy()
    working["_invalid_reasons"] = invalid_reasons
    invalid_df = working[working["_invalid_reasons"].map(len) > 0].copy()
    valid_df = working[working["_invalid_reasons"].map(len) == 0].drop(columns=["_invalid_reasons"]).copy()

    summary = {
        "total_rows": int(len(df)),
        "valid_rows": int(len(valid_df)),
        "invalid_rows": int(len(invalid_df)),
        "duplicate_texts": int(df["text"].duplicated().sum()) if "text" in df.columns else 0,
        "type_distribution": df["type"].value_counts(dropna=False).to_dict() if "type" in df.columns else {},
        "category_distribution": df["category"].value_counts(dropna=False).head(50).to_dict() if "category" in df.columns else {},
        "wallet_null_count": int(df["wallet"].isna().sum()) if "wallet" in df.columns else None,
    }
    return valid_df, invalid_df, summary


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
