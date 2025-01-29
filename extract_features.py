import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew
from tqdm import tqdm
import matplotlib.pyplot as plt
from pprint import pprint


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def extract_features(file_path, bpm):
    try:
        y, sr = librosa.load(file_path, sr=44100,
                             duration=6, dtype=np.float64)

        fade_out_samples = int(0.5 * sr)  # fade out by half a second
        fade_out_envelope = np.linspace(1, 0, fade_out_samples)
        y[-fade_out_samples:] *= fade_out_envelope

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
            features['mfcc_median'],
            features['mfcc_skew'],
            features['mfcc_var'],
            features['onset_strength_median'],
            features['onset_strength_var'],
            features['rms_energy_median'],
            features['rms_energy_var'],
            features['spectral_contrast_median'],
            features['spectral_contrast_var'],
            features['zcr_median'],
            features['zcr_var'],
            [bpm]
        ])

        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == '__main__':

    print("done")
    file_paths = [
        ("data/wrld_smb_drm_8br_id_001_wav.csv",
         "data/wrld_smb_drm_features_and_labels.npz"),
        ('data/hh_lfbb_lps_mid_001-009.csv', 'data/hh_lfbb_lps_mid_001-009.npz'),
        ("data/edm_tr9_drm_id_001.csv", "data/edm_tr9_drm_id_001.npz"),
        ("data/pop_rok_drm_id_001_wav.csv", "data/pop_rok_drm_id_001_wav.csv")
    ]

    for file in file_paths:
        csv_path = file[0]

        # csv_path = "data/wrld_smb_drm_8br_id_001_wav.csv"
        # csv_path = 'data/hh_lfbb_lps_mid_001-009.csv'
        # csv_path = "data/edm_tr9_drm_id_001.csv"
        # csv_path = "data/pop_rok_drm_id_001_wav.csv"
        data = pd.read_csv(csv_path)

        # unique_label = [data['genre'].iloc[0], data['subgenre'].iloc[0]]
        bpm_values = data['bpm'].values

        feature_list = []

        for i, row in tqdm(data.iterrows()):
            file_path = row['file_path']
            bpm = bpm_values[i]

            features = extract_features(file_path, bpm)
            if features is not None:
                feature_list.append(features)

        X = np.array(feature_list)
        # y = np.array([unique_label] * len(feature_list))
        output_npz_path = file[1]
        # output_npz_path = "data/wrld_smb_drm_features_and_labels.npz"
        # output_npz_path = "data/hh_lfbb_lps_mid_001-009.npz"
        # output_npz_path = "data/edm_tr9_drm_id_001.npz"
        # output_npz_path = "data/pop_rok_drm_id_001_wav.npz"
        np.savez(output_npz_path, features=X)
        print(f"Features and labels saved to {output_npz_path}")
