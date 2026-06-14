# Fluxa Voice Intent Golden Dataset — Sundanese 5000 Synthetic Samples

Dataset ini adalah **synthetic expected/golden dataset bahasa Sunda** untuk evaluasi intent parser Fluxa.

Tujuan utama:
1. Menguji parsing intent transaksi dari kalimat Sunda sehari-hari.
2. Menguji normalisasi nominal Sunda seperti `lima rebu`, `tilu puluh rébu`, `Rp15.000`, `15rb`.
3. Menguji mapping ke kontrak output Fluxa: `amount`, `type`, `category`, `wallet`, `note`, `currency`.
4. Menjadi dasar evaluasi, benchmark API AI, atau fine-tuning di masa depan.

## Files
- `fluxa_voice_intent_dataset_sunda_5000.jsonl` — format utama untuk evaluator/parser.
- `fluxa_voice_intent_dataset_sunda_5000.csv` — format inspeksi cepat di spreadsheet.

## Schema JSONL

```json
{
  "id": "VX-SU-000001",
  "split": "train|validation|test",
  "group": "expense_basic",
  "case_type": "expense_basic",
  "language_style": "su",
  "raw_text": "tadi meuli kopi lima rebu",
  "normalized_text": "tadi meuli kopi 5000",
  "expected": {
    "amount": 5000,
    "type": "expense",
    "category": "Makan & Minum",
    "wallet": null,
    "note": "kopi",
    "currency": "IDR"
  },
  "difficulty": "easy|medium|hard"
}
```

## Distribution by Group

- expense_basic: 1500
- income_basic: 700
- transport_parkir_bensin: 600
- wallet_mention: 500
- sundanese_daily_conversation: 700
- slang_typo_noisy_sunda: 500
- ambiguous_hard_case: 300
- transfer_future_optional: 200

## Distribution by Type

- expense: 3076
- income: 1724
- transfer: 200

## Distribution by Category

- Tagihan & Utilitas: 438
- Kesehatan: 388
- Makan & Minum: 403
- Transportasi: 1015
- Hiburan: 413
- Belanja: 419
- Freelance: 900
- Gaji: 824
- Transfer: 200

## Split

- train: 4000
- validation: 500
- test: 500

## Difficulty

- easy: 3500
- medium: 1000
- hard: 500

## Design Notes
- `raw_text` dibuat dalam gaya bahasa Sunda percakapan.
- `normalized_text` mempertahankan struktur Sunda, tetapi nominal dinormalisasi menjadi angka.
- `category` tetap memakai kategori aplikasi Fluxa agar kompatibel dengan parser dan database.
- `confidence` tidak dimasukkan karena confidence adalah output prediksi model/API, bukan ground truth.
- Group `transfer_future_optional` disediakan untuk future case; bisa diabaikan untuk Basic MVP.
- Dataset ini adalah data sintetis, bukan data transaksi user asli.
