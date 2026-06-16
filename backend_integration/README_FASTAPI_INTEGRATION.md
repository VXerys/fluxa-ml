# FastAPI Integration Notes

Notebook ini menghasilkan artifact yang dapat dipakai backend FastAPI Fluxa.

## Endpoint final

```text
POST /api/v1/voice/parse
Content-Type: multipart/form-data
file: audio file
```

## Response contract

```json
{
  "transcript": {
    "raw": "mayar parkir motor lima ribu",
    "normalized": "bayar parkir motor lima ribu",
    "language_hint": "su-id",
    "confidence": 0.91
  },
  "transaction": {
    "type": "expense",
    "amount": 5000,
    "category": "Transportasi",
    "wallet": null,
    "description": "bayar parkir motor",
    "currency": "IDR"
  },
  "classification": {
    "type_confidence": 0.97,
    "category_confidence": 0.94,
    "wallet_confidence": 0.0,
    "overall_confidence": 0.91
  },
  "warnings": []
}
```

## Backend service flow

```text
voice_routes.py
→ stt_service.py              # faster-whisper small
→ text_normalizer_service.py  # same logic as src/text_normalizer.py
→ transaction_classifier.py   # load model artifact
→ amount_parser_service.py    # same logic as src/amount_parser.py
→ transaction_builder.py      # build response contract
```

## Minimal model loading for baseline

```python
import joblib

class TransactionClassifierService:
    def __init__(self):
        self.type_model = joblib.load("app/models/classifier/baseline_type_classifier.joblib")
        self.category_model = joblib.load("app/models/classifier/baseline_category_classifier.joblib")

    def predict(self, normalized_text: str):
        return {
            "type": self.type_model.predict([normalized_text])[0],
            "category": self.category_model.predict([normalized_text])[0],
        }
```

Untuk production/research final, ganti baseline joblib dengan artifact IndoBERT/XLM-R dari notebook.
