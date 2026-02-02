import numpy as np
import librosa


def extract_audio_features(
    audio_path: str,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    Extracts a fixed-length feature vector from an audio file.

    Features:
    - MFCC mean (n_mfcc)
    - MFCC std (n_mfcc)
    - Spectral centroid mean + std
    - Zero-crossing rate mean + std

    Returns:
    - feature_vector: shape (2*n_mfcc + 4,)
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Aggregate statistics
    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(zcr),
        np.std(zcr),
    ])

    return feature_vector
