# Fluxa Audio Pipeline v1 - Whisper Small GPU

## Configuration
- STT model: faster-whisper small
- Device: CUDA GPU
- Compute type: float16
- Parser: rule-based amount parser + TF-IDF Logistic Regression classifier
- Dataset: 5000 synthetic Sunda-Indonesian transaction text samples

## Final Audio Evaluation Summary
- Total audio: 10
- Amount detected: 10
- Amount missing: 0
- Warnings: 4
- Type distribution:
  - expense: 9
  - income: 1
- Category distribution:
  - Transportasi: 4
  - Makan & Minum: 3
  - Tagihan & Utilitas: 2
  - Gaji: 1

## Notes
This version separates STT errors from transaction parser errors. Some warnings are expected because real audio transcription may produce ambiguous or unrealistic amounts.
