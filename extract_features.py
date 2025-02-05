import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pprint import pprint
from multiprocessing import Pool, cpu_count


file_paths = [
    (
        "data/wrld_smb_drm_8br_id_001_wav.csv",
        "data/wrld_smb_drm_features_and_labels.npz",
    ),
    ("data/hh_lfbb_lps_mid_001-009.csv", "data/hh_lfbb_lps_mid_001-009.npz"),
    ("data/edm_tr9_drm_id_001.csv", "data/edm_tr9_drm_id_001.npz"),
    ("data/pop_rok_drm_id_001_wav.csv", "data/pop_rok_drm_id_001_wav.npz"),
]


def process_file(file):
    csv_path, output_npz_path = file

    # Read CSV
    data = pd.read_csv(csv_path)
    bpm_values = data["bpm"].values

    feature_list = []

    # Process each row
    for i, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {csv_path}"):
        file_path = row["file_path"]
        bpm = bpm_values[i]

        # Extract features (Assumes extract_features is defined)
        features = extract_features(file_path, bpm)
        if features is not None:
            feature_list.append(features)

    # Convert to NumPy array
    X = np.array(feature_list)

    # Save features
    np.savez(output_npz_path, features=X)
    print(f"Features and labels saved to {output_npz_path}")


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def extract_features(file_path, bpm):
    try:
        y, sr = librosa.load(file_path, sr=44100, duration=6, dtype=np.float64)

        fade_out_samples = int(0.5 * sr)  # fade out by half a second
        fade_out_envelope = np.linspace(1, 0, fade_out_samples)
        y[-fade_out_samples:] *= fade_out_envelope

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y + 1e-6)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)[0]

        features = {
            "mfcc_mean": np.mean(mfcc, axis=1),
            "mfcc_std": np.std(mfcc, axis=1),
            "zcr_mean": np.array([np.mean(zcr)]),
            "zcr_std": np.array([np.std(zcr)]),
            "spectral_contrast_mean": np.mean(spectral_contrast, axis=1),
            "spectral_contrast_std": np.std(spectral_contrast, axis=1),
            "onset_strength_mean": np.array([np.mean(onset_strength)]),
            "onset_strength_std": np.array([np.std(onset_strength)]),
            "rms_energy_mean": np.array([np.mean(rms_energy)]),
            "rms_energy_std": np.array([np.std(rms_energy)]),
            "spectral_centroid_mean": np.mean(spectral_centroid, axis=1),
            "spectral_centroid_std": np.std(spectral_centroid, axis=1),
        }

        
        feature_vector = np.concatenate(
            [
                features["mfcc_mean"],
                features["mfcc_std"],
                features["onset_strength_mean"],
                features["onset_strength_std"],
                features["rms_energy_mean"],
                features["rms_energy_std"],
                features["spectral_contrast_mean"],
                features["spectral_contrast_std"],
                features["zcr_mean"],
                features["zcr_std"],
                features["spectral_centroid_mean"],
                features["spectral_centroid_std"],
                [bpm],
            ]
        )

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == "__main__":
    # for file_path in file_paths:
    #     process_file(file_path)

    # Use up to the number of available CPUs
    num_workers = min(len(file_paths), cpu_count())
    with Pool(num_workers) as pool:
        pool.map(process_file, file_paths)
