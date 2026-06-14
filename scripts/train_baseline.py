from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.amount_parser import parse_amount
from src.dataset_loader import load_dataset, save_json, validate_dataset
from src.metrics_utils import classification_metrics, evaluate_end_to_end
from src.text_normalizer import normalize_text


def build_classifier(max_features: int = 12000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline Fluxa intent classifiers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    df, invalid_df, summary = validate_dataset(df)
    save_json(summary, report_dir / "dataset_summary.json")
    if not invalid_df.empty:
        invalid_df.to_csv(report_dir / "invalid_rows.csv", index=False)

    df = df.copy()
    df["normalized_text"] = df["text"].map(normalize_text)
    df = df.dropna(subset=["type", "category"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=args.random_state,
        stratify=df["type"],
    )

    type_model = build_classifier()
    category_model = build_classifier()

    type_model.fit(train_df["normalized_text"], train_df["type"])
    category_model.fit(train_df["normalized_text"], train_df["category"])

    test_df = test_df.copy()
    test_df["pred_type"] = type_model.predict(test_df["normalized_text"])
    test_df["pred_category"] = category_model.predict(test_df["normalized_text"])
    test_df["pred_amount"] = test_df["normalized_text"].map(parse_amount)

    type_metrics = classification_metrics(test_df["type"], test_df["pred_type"])
    category_metrics = classification_metrics(test_df["category"], test_df["pred_category"])
    amount_accuracy = float((test_df["amount"].astype("Int64") == test_df["pred_amount"].astype("Int64")).mean()) if "amount" in test_df else None
    e2e_metrics = evaluate_end_to_end(test_df)

    joblib.dump(type_model, model_dir / "baseline_type_classifier.joblib")
    joblib.dump(category_model, model_dir / "baseline_category_classifier.joblib")

    save_json(type_metrics, report_dir / "baseline_type_metrics.json")
    save_json(category_metrics, report_dir / "baseline_category_metrics.json")
    save_json({"amount_accuracy": amount_accuracy}, report_dir / "amount_parser_metrics.json")
    save_json(e2e_metrics, report_dir / "end_to_end_metrics.json")
    test_df.to_csv(report_dir / "baseline_test_predictions.csv", index=False)

    print("Baseline training complete")
    print(json.dumps(e2e_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
