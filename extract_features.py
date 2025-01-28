from pprint import pprint
import librosa
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

file_path = "data/wrld_smb_drm_8br_id_001_wav/96bpm_wrld_smb_drm_8br_id_001_1158.wav"


def min_max_normalize(arr):

    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


y, sr = librosa.load(file_path, sr=44100, duration=10, dtype=np.float64)

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
zcr = librosa.feature.zero_crossing_rate(
    y + 1e-6)[0]
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
rms_energy = librosa.feature.rms(y=y)[0]

mfcc_normalized = min_max_normalize(mfcc)
zcr_normalized = min_max_normalize(zcr)
spectral_contrast_normalized = min_max_normalize(spectral_contrast)
onset_strength_normalized = min_max_normalize(onset_strength)
rms_energy_normalized = min_max_normalize(rms_energy)


features = {
    'mfcc_median': np.median(mfcc_normalized, axis=1),
    'mfcc_var': np.var(mfcc_normalized, axis=1),
    'mfcc_skew': skew(mfcc_normalized, axis=1),
    'zcr_median': np.array([np.median(zcr_normalized)]),
    'zcr_var': np.array([np.var(zcr_normalized)]),
    'spectral_contrast_median': np.median(spectral_contrast_normalized, axis=1),
    'spectral_contrast_var': np.var(spectral_contrast_normalized, axis=1),
    'onset_strength_median': np.array([np.median(onset_strength_normalized)]),
    'onset_strength_var': np.array([np.var(onset_strength_normalized)]),
    'rms_energy_median': np.array([np.median(rms_energy_normalized)]),
    'rms_energy_var': np.array([np.var(rms_energy_normalized)]),
}

all_features = np.concatenate([
    features['mfcc_median'], features['mfcc_var'], features['mfcc_skew'],
    features['zcr_median'], features['zcr_var'],
    features['spectral_contrast_median'], features['spectral_contrast_var'],
    features['onset_strength_median'], features['onset_strength_var'],
    features['rms_energy_median'], features['rms_energy_var']
])

pprint(features)


def plot_normalized_stuff():
    plt.figure(figsize=(15, 15))

    plt.subplot(5, 1, 1)
    plt.imshow(mfcc_normalized, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title("Normalized MFCCs")
    plt.xlabel("Frames")
    plt.ylabel("MFCC Coefficients")

    plt.subplot(5, 1, 2)
    plt.plot(zcr_normalized, label="Normalized ZCR", color="purple")
    plt.title("Normalized Zero-Crossing Rate")
    plt.xlabel("Frames")
    plt.ylabel("ZCR")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.imshow(spectral_contrast_normalized, aspect='auto',
               origin='lower', cmap='plasma')
    plt.colorbar()
    plt.title("Normalized Spectral Contrast")
    plt.xlabel("Frames")
    plt.ylabel("Frequency Bands")

    plt.subplot(5, 1, 4)
    plt.plot(onset_strength_normalized,
             label="Normalized Onset Strength", color="green")
    plt.title("Normalized Onset Strength")
    plt.xlabel("Frames")
    plt.ylabel("Onset Strength")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(rms_energy_normalized,
             label="Normalized RMS Energy", color="orange")
    plt.title("Normalized RMS Energy")
    plt.xlabel("Frames")
    plt.ylabel("RMS Energy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("features_plot3.png")
