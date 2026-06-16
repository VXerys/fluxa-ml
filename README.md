# Fluxa Voice AI Training Pipeline

Research/training pipeline untuk fitur **Voice Transaction** Fluxa.

Pipeline final:

```text
Flutter record audio
→ FastAPI backend
→ faster-whisper small STT
→ text normalizer Indonesia/Sunda
→ transaction classifier
→ rule-based amount parser
→ transaction JSON
→ Flutter draft transaction card
```

## Fokus repository folder ini

Folder ini **tidak untuk fine-tuning Whisper**. Dataset 5000 JSON/JSONL dipakai untuk:

- `text -> type` classifier
- `text -> category` classifier
- `text -> wallet` classifier optional
- `text -> expected_amount` untuk evaluasi rule-based amount parser
- end-to-end transaction JSON evaluation

## Struktur

```text
fluxa_voice_ai_training_pipeline/
├── notebooks/
│   └── fluxa_voice_transaction_training_pipeline.ipynb
├── src/
│   ├── amount_parser.py
│   ├── dataset_loader.py
│   ├── inference_pipeline.py
│   ├── metrics_utils.py
│   └── text_normalizer.py
├── scripts/
│   ├── validate_dataset.py
│   └── train_baseline.py
├── configs/
│   └── training_config.yaml
├── data/
│   ├── README.md
│   └── sample_dataset.jsonl
├── models/
├── reports/
├── backend_integration/
│   └── README_FASTAPI_INTEGRATION.md
├── requirements.txt
└── .gitignore
```

## Cara pakai cepat

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Pindahkan dataset kamu ke:

```text
data/fluxa_voice_intent_dataset_sunda_5000.jsonl
```

3. Jalankan validasi:

```bash
python scripts/validate_dataset.py --dataset data/fluxa_voice_intent_dataset_sunda_5000.jsonl
```

4. Jalankan baseline cepat:

```bash
python scripts/train_baseline.py --dataset data/fluxa_voice_intent_dataset_sunda_5000.jsonl
```

5. Buka notebook:

```text
notebooks/fluxa_voice_transaction_training_pipeline.ipynb
```

## Format dataset yang direkomendasikan

Satu row per transaksi:

```json
{
  "text": "mayar parkir motor lima rebu",
  "type": "expense",
  "amount": 5000,
  "category": "Transportasi",
  "wallet": null,
  "description": "bayar parkir motor"
}
```

Field wajib minimal:

| Field | Required | Keterangan |
|---|---:|---|
| `text` | yes | input natural language user |
| `type` | yes | `expense`, `income`, atau `transfer` |
| `amount` | recommended | integer rupiah, boleh `null` untuk ambiguous case |
| `category` | recommended | kategori final |
| `wallet` | optional | dompet/bank/e-wallet, boleh `null` |
| `description` | optional | catatan singkat |

## Output training

```text
models/
├── baseline_type_classifier.joblib
├── baseline_category_classifier.joblib
├── type_label_mapping.json
├── category_label_mapping.json
└── transformer_* optional

reports/
├── dataset_summary.json
├── invalid_rows.csv
├── baseline_type_metrics.json
├── baseline_category_metrics.json
├── amount_parser_metrics.json
└── end_to_end_metrics.json
```

## Rekomendasi mesin

- CPU Xxsmall: validasi dataset, parser, baseline TF-IDF.
- GPU Xsmall: fine-tuning IndoBERT/XLM-R.

Mulai dari baseline dulu sebelum memakai GPU.

## Optional Groq fallback backend

Groq fallback is disabled by default and only runs after the local parser returns
an uncertain draft. It does not replace Whisper STT, does not replace the local
amount parser, and must never be called from Flutter.

Required backend environment variables:

```env
ENABLE_GROQ_FALLBACK=false
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=
GROQ_TIMEOUT_SECONDS=8
```

Set `ENABLE_GROQ_FALLBACK=true`, provide `GROQ_API_KEY`, and choose a current
Groq chat model for `GROQ_MODEL` in the backend environment only. The response
stays backward-compatible; `transaction.title` and optional confidence fields
may be present, while existing fields remain unchanged.

Fallback is attempted only for uncertain cases such as parser warnings, noisy
titles, Transfer/category conflicts, missing wallet despite wallet-like words,
suspicious STT tokens, or low classifier confidence. Groq corrections are
validated against the allowed category, type, and wallet lists before being
applied. Invalid JSON, timeout, invalid values, or request failure keeps the
local parser result and adds `groq_fallback_failed` when fallback was attempted.
