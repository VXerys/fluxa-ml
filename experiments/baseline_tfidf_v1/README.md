# Baseline TF-IDF v1 — Fluxa Voice Intent Parser

## Dataset

- Dataset: fluxa_voice_intent_dataset_sunda_5000_fixed.jsonl
- Total rows: 5000
- Test rows: 500

## Pipeline

- Text input: raw_text converted to text
- Type classifier: TF-IDF + Logistic Regression
- Category classifier: TF-IDF + Logistic Regression
- Amount parser: rule-based Indonesian/Sundanese parser

## Result

Final result after amount parser patch:

- Type accuracy: 1.0
- Category accuracy: 1.0
- Amount accuracy: 1.0
- End-to-end all fields accuracy: 1.0

## Notes

This result is produced on a controlled synthetic dataset. For thesis reporting, this should be treated as baseline performance on curated data. Further evaluation should use real Whisper transcript output to test robustness against STT noise.
