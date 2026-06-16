"""Metrics helpers for Fluxa classifier experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def evaluate_end_to_end(df: pd.DataFrame) -> dict[str, Any]:
    required = ["type", "pred_type", "category", "pred_category", "amount", "pred_amount"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for E2E evaluation: {missing}")

    type_correct = df["type"] == df["pred_type"]
    category_correct = df["category"] == df["pred_category"]
    amount_correct = df["amount"].astype("Int64") == df["pred_amount"].astype("Int64")
    all_correct = type_correct & category_correct & amount_correct

    return {
        "samples": int(len(df)),
        "type_accuracy": float(type_correct.mean()),
        "category_accuracy": float(category_correct.mean()),
        "amount_accuracy": float(amount_correct.mean()),
        "all_fields_accuracy": float(all_correct.mean()),
    }
