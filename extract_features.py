import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew
from tqdm import tqdm
import matplotlib.pyplot as plt


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def extract_features(file_path, normalized_bpm):
    try:
        y, sr = librosa.load(file_path, sr=44100,
                             duration=10, dtype=np.float64)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y + 1e-6)[0]
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

        feature_vector = np.concatenate([
            features['mfcc_median'], features['mfcc_var'], features['mfcc_skew'],
            features['zcr_median'], features['zcr_var'],
            features['spectral_contrast_median'], features['spectral_contrast_var'],
            features['onset_strength_median'], features['onset_strength_var'],
            features['rms_energy_median'], features['rms_energy_var'],
            [normalized_bpm]
        ])

        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


csv_path = "data/wrld_smb_drm_8br_id_001_wav.csv"
data = pd.read_csv(csv_path)

unique_label = data['genre'].iloc[0]
bpm_values = data['bpm'].values
normalized_bpms = min_max_normalize(bpm_values)

feature_list = []

for i, row in tqdm(data.iterrows()):
    file_path = row['file_path']
    normalized_bpm = normalized_bpms[i]

    features = extract_features(file_path, normalized_bpm)
    if features is not None:
        feature_list.append(features)

X = np.array(feature_list)
y = np.array([unique_label] * len(feature_list))

output_npz_path = "data/wrld_smb_drm_features_and_labels.npz"
np.savez(output_npz_path, features=X, labels=y)
print(f"Features and labels saved to {output_npz_path}")


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
