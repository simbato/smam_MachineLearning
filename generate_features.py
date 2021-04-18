from generate import get_filenames
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from pathlib import Path

def to_rows(name, x):
    return {f'{name}_{i}': x[i] for i in range(len(x))}

def get_features(y):
    agg_features = {
        # Duljina tišine na početku i na kraju pjesme
        'silence':  librosa.get_duration(y) - librosa.get_duration(librosa.effects.trim(y)[0]),
        # Estimate the global tempo
        'tempo':    librosa.beat.tempo(y, sr=sr)[0],
    }
    features = {
        # Spectral centroid
        'spectral_centroid':    librosa.feature.spectral_centroid(y).ravel(),
        # Roll-off frequency
        'spectral_rolloff':     librosa.feature.spectral_rolloff(y).ravel(),
        # Zero-crossing rate
        'zero_crossing_rate':   librosa.feature.zero_crossing_rate(y).ravel(),
        # Root-mean-square value
        'rms':                  librosa.feature.rms(y).ravel(),
        # 2nd-order spectral bandwidth
        'spectral_bandwidth':   librosa.feature.spectral_bandwidth(y).ravel(),
        # Spectral flatness
        'spectral_flatness':    librosa.feature.spectral_flatness(y).ravel(),
        # Estimate the tempo (beats per minute)
        'tempo':                librosa.beat.tempo(y, aggregate=None).ravel(),
    }
    # Chromagram
    features.update(to_rows('chroma_stft', librosa.feature.chroma_stft(y)))
    # Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    features.update(to_rows('poly_features', librosa.feature.poly_features(y)))
    # Tonal centroid features
    features.update(to_rows('tonnetz', librosa.feature.tonnetz(y)))
    # Mel-frequency cepstral coefficients
    features.update(to_rows('mfcc', librosa.feature.mfcc(y, n_mfcc=20)))
    # Spectral contrast
    features.update(to_rows('spectral_contrast', librosa.feature.spectral_contrast(y)))

    # Računamo različite mjere za distribucije svih featura
    for k, v in features.items():
        agg_features.update({
            f'{k}_mean':        np.mean(v),
            f'{k}_std':         np.std(v),
            f'{k}_skew':        skew(v),
            f'{k}_kurtosis':    kurtosis(v),
            f'{k}_median':      np.median(v),
            f'{k}_min':         np.min(v),
            f'{k}_max':         np.max(v),
        })

    return agg_features


if __name__ == "__main__":
    rows = []
    files = get_filenames('raw', '.npy')
    for i in range(len(files)):
        track_id = int(files[i][0][:-9])
        y, sr = np.load(files[i][1], allow_pickle=True)
        features = get_features(y)
        features['track_id'] = track_id
        rows.append(features)
        print(f'{(i+1)/len(files)*100:.1f}% done')

    df = pd.DataFrame(rows).set_index('track_id')
    df.to_csv('handcrafted.csv')
