# music-ml-lab

🎵 Music-ML-Lab — Audio & Symbolic Music Representation Learning

This repository explores multiple approaches to music understanding using machine learning, ranging from classical DSP-based feature engineering to deep learning and symbolic music modeling.

The project is structured as controlled experimental phases, emphasizing interpretability, representation choice, and scientific comparison.

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

🔹 Phase 1 — Classical Audio Feature Baseline

Goal:
Establish a strong, interpretable baseline using signal processing features.

Features used:

MFCC (mean & standard deviation)

Spectral centroid

Zero-crossing rate

Model:

Multinomial Logistic Regression

Result:

66.5% accuracy on GTZAN

Strong performance on timbre-distinct genres (jazz, classical, metal)

Key insight:
Timbre-based features are effective for genres with distinct instrumentation, but struggle with rhythm-driven distinctions.

🔹 Phase 2A — CNN on Mel-Spectrograms

Goal:
Evaluate representation learning directly from time–frequency data.

Approach:

Mel-spectrograms (128 × T)

Shallow CNN architecture

Result:

CNN underperformed classical baseline

Key insight:
With limited data, deep models struggle without strong inductive bias. Classical features remain competitive in small-data regimes.

🔹 Phase 2B — Rhythm & Harmonic Feature Extension

Added features:

Chroma (harmonic structure)

Tempo (rhythmic cue)

Feature dimension:
30 → 55

Result:

No significant accuracy improvement over baseline

Key insight:
Global tempo and chroma statistics are insufficient for resolving rhythm-based genre confusions when used with linear classifiers, motivating temporal or symbolic representations.

🔹 Phase 2C — Symbolic Music Modeling (MIDI) (in progress)

Goal:
Move beyond audio into explicit musical structure using symbolic representations.

Focus:

MIDI analysis (notes, timing, velocity)

Structural pattern learning

Music generation

This phase explores how symbolic representations can capture musical form more effectively than frame-based audio features.