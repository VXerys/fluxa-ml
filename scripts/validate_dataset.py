from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import load_dataset, save_json, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Fluxa voice intent dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/JSON/CSV dataset")
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    valid_df, invalid_df, summary = validate_dataset(df)

    save_json(summary, report_dir / "dataset_summary.json")
    if not invalid_df.empty:
        invalid_df.to_csv(report_dir / "invalid_rows.csv", index=False)

    print("Dataset validation complete")
    print(summary)


if __name__ == "__main__":
    main()
