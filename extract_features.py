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
        y, sr = librosa.load(file_path, duration=4, dtype=np.float64)

        fade_out_samples = int(0.5 * sr)  # fade out by half a second
        fade_out_envelope = np.linspace(1, 0, fade_out_samples)
        y[-fade_out_samples:] *= fade_out_envelope

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
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


def experiment1_features(file_path):
    pass


if __name__ == "__main__":
    # for file_path in file_paths:
    #     process_file(file_path)

    # Use up to the number of available CPUs
    # num_workers = min(len(file_paths), cpu_count())
    # with Pool(num_workers) as pool:
    #     pool.map(process_file, file_paths)
    # Load the drum audio file
    audio_path = "data/pop_rok_drm_id_001_wav/100bpm_pop_rok_drm_id_001_0001.wav"
    y, sr = librosa.load(audio_path, duration=4, dtype=np.float64)

    fade_out_samples = int(0.5 * sr)  # fade out by half a second
    fade_out_envelope = np.linspace(1, 0, fade_out_samples)
    y[-fade_out_samples:] *= fade_out_envelope

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Compute the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Compute inter-onset intervals (IOIs) in seconds
    if len(beat_times) > 1:
        iois = np.diff(beat_times)
    else:
        iois = np.array([0])  # default value when only one beat is detected

    # Compute the zero crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Create a time axis for onset envelope and ZCR
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    zcr_times = librosa.frames_to_time(np.arange(zcr.shape[1]), sr=sr)

    # Create subplots with 2 rows sharing the x-axis
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Plot Onset Envelope on the first subplot
    axs[0].plot(times, onset_env, label="Onset Strength", color="blue")
    axs[0].vlines(
        beat_times,
        ymin=onset_env.min(),
        ymax=onset_env.max(),
        colors="green",
        linestyles="dashed",
        label="Beat Times",
    )
    axs[0].set_facecolor("0.9")
    axs[0].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    axs[0].set_title("Onset Envelope with Beat Times")
    axs[0].set_ylabel("Onset Strength")
    axs[0].set_ylim(0)
    axs[0].legend()

    # Plot Zero Crossing Rate on the second subplot
    axs[1].plot(zcr_times, zcr[0], label="Zero Crossing Rate", color="red")
    axs[1].vlines(
        beat_times,
        ymin=zcr[0].min(),
        ymax=zcr[0].max(),
        colors="green",
        linestyles="dashed",
        label="Beat Times",
    )
    axs[1].set_facecolor("0.9")
    axs[1].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    axs[1].set_title("Zero Crossing Rate with Beat Times")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("ZCR")
    axs[1].set_ylim(0)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("temp.png")

    # Now compute aggregated descriptors
    aggregated_features = {
        "mean_onset": np.mean(onset_env),
        "std_onset": np.std(onset_env),
        "mean_zcr": np.mean(zcr),
        "std_zcr": np.std(zcr),
        "mean_ioi": np.mean(iois),
        "std_ioi": np.std(iois),
        "tempo": tempo,  # overall BPM estimation
    }

    pprint(aggregated_features)

    spec_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr
    )  # gives us the brightness
    spec_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr
    )  # variety of frequencies
    spec_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr
    )  # under which frequency per bin does most of the energy exist?
    spec_flatness = librosa.feature.spectral_flatness(
        y=y
    )  # how uniform is the distribution? found more in percussive sounds.

    # Aggregate the spectral features across time (using mean and standard deviation)
    aggregated_spectral_features = {
        "mean_centroid": np.mean(spec_centroid),
        "std_centroid": np.std(spec_centroid),
        "mean_bandwidth": np.mean(spec_bandwidth),
        "std_bandwidth": np.std(spec_bandwidth),
        "mean_rolloff": np.mean(spec_rolloff),
        "std_rolloff": np.std(spec_rolloff),
        "mean_flatness": np.mean(spec_flatness),
        "std_flatness": np.std(spec_flatness),
    }

    print("Aggregated Spectral Features:", aggregated_spectral_features)
