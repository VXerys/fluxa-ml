# Fluxa Voice AI - Audio Pipeline Evaluation v1

## Objective
Evaluate the Fluxa voice transaction pipeline from audio input to structured transaction JSON.

## Pipeline
Audio `.m4a` → faster-whisper STT → text normalization → transaction parser → transaction JSON.

## Model Configuration
- STT: faster-whisper small
- Device: CUDA GPU
- Compute type: float16
- Classifier: TF-IDF + Logistic Regression
- Amount parser: rule-based Indonesian/Sundanese parser

## Result Summary
- Total audio samples: 10
- Amount detected: 10
- Amount missing: 0
- Warning rows: 4
- Type distribution:
  - expense: 9
  - income: 1
- Category distribution:
  - Transportasi: 4
  - Makan & Minum: 3
  - Tagihan & Utilitas: 2
  - Gaji: 1

## Key Improvements from Base to Small GPU
Whisper small GPU significantly improved transcript quality compared with Whisper base. Several previously broken transcripts became usable, including parking, food, salary, and transport samples.

## Known Limitations
Some outputs still require manual user review:
- Unrealistic amount detected from STT, for example Rp 300.000.000.
- Ambiguous numeric transcript such as `1,20 ribu`.
- Strange transcript text that still produces plausible transaction fields.

## Conclusion
The audio pipeline v1 is functional and suitable for prototype integration. The current system should return a draft transaction to the Flutter app, not directly save the transaction without user confirmation.
