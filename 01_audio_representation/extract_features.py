import numpy as np
import librosa


def extract_audio_features(
    audio_path: str,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    Extended audio feature extractor.

    Features:
    - MFCC mean + std
    - Spectral centroid mean + std
    - Zero-crossing rate mean + std
    - Chroma mean + std
    - Tempo (BPM)

    Returns:
    - feature_vector
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(spectral_centroid),
        np.std(spectral_centroid),

        np.mean(zcr),
        np.std(zcr),

        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),

        tempo
    ])

    return feature_vector
