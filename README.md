# music-ml-lab

Day 1–2: Audio DSP feature extraction + baseline genre classification.

## Structure
- `01_audio_representation/` — audio loading, DSP features (MFCC, spectral centroid, ZCR), baseline classifier
- `data/` — datasets (ignored in git)
- `outputs/` — figures + trained models (ignored in git)

## How to run
Use Google Colab notebooks in `01_audio_representation/`.

## Baseline Classification Results

Using MFCC-based acoustic features (mean and standard deviation), spectral centroid,
and zero-crossing rate, a multinomial Logistic Regression classifier achieved
66.5% accuracy on the GTZAN dataset.

Genres such as jazz, classical, and metal were classified reliably, reflecting
their distinct timbral and spectral characteristics. Confusions primarily occurred
among rhythm-driven genres (rock, disco, hip-hop, reggae), highlighting the
limitations of frame-level timbral features in modeling temporal rhythmic structure.

These results establish a strong, interpretable baseline and motivate the use of
temporal models and deep representations in future work.
