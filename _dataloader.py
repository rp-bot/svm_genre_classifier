import numpy as np
from extract_features import extract_features
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


MIN_BPM = 30
MAX_BPM = 300
SCALER = MinMaxScaler(feature_range=(0, 1))
SCALER.fit(np.array([[MIN_BPM], [MAX_BPM]]))


def scale_bpm(bpm):
    bpm_scaled = SCALER.transform(bpm)
    return bpm_scaled


def inverse_scale_bpm(scaled_bpm):
    bpm_original = SCALER.inverse_transform(np.array([[scaled_bpm]]))
    return bpm_original[0, 0]


def clean_and_split_data():
    sm_features_labels = np.load('data/wrld_smb_drm_features_and_labels.npz')

    hh_features_labels = np.load("data/hh_lfbb_lps_mid_001-009.npz")

    tr9_feature_labels = np.load("data/edm_tr9_drm_id_001.npz")

    pop_feature_labels = np.load("data/pop_rok_drm_id_001_wav.npz")

    sm_data = sm_features_labels['features']

    hh_data = hh_features_labels['features']

    tr9_data = tr9_feature_labels['features']

    pop_data = pop_feature_labels['features']

    indices_hh = np.linspace(0, hh_data.shape[0] - 1, 1100, dtype=int)
    hh_data_len_reduced = hh_data[indices_hh]

    indices_tr9 = np.linspace(0, tr9_data.shape[0] - 1, 1100, dtype=int)
    tr9_data_len_reduced = tr9_data[indices_tr9]

    indices_pop = np.linspace(0, pop_data.shape[0] - 1, 1100, dtype=int)
    pop_data_len_reduced = pop_data[indices_pop]

    feature_slices = [
        # slice(0, 13),   # [0:12] 'mfcc_median'
        # slice(13, 26),  # [13:25]'mfcc_skew'
        # slice(26, 39),  # [26:38]'mfcc_var'
        # slice(39, 40),  # [39]  'onset_strength_median'
        # slice(40, 41),  # [40] 'onset_strength_var'
        # slice(41, 42),  # [41] 'rms_energy_median'
        # slice(42, 43),  # [42]  'rms_energy_var'
        # slice(43, 50),  # [43:49] 'spectral_contrast_median'
        # slice(50, 57),  # [50:56]'spectral_contrast_var'
        # slice(57, 58),  # [57]'zcr_median'
        # slice(58, 59),  # [58]'zcr_var'
        # slice(59, 60),  # [59]  'bpm'
    ]

    concatenated_dataset = np.concatenate(
        [sm_data,
         hh_data_len_reduced,
         tr9_data_len_reduced,
         pop_data_len_reduced]
    )

    labels = ["world_samba", "hip_hop_lofi_boom_bap", "pop_rock", "edm_tr_909"]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    expanded_labels = np.concatenate([
        np.full(1100, encoded_labels[0]),  # world_samba
        np.full(1100, encoded_labels[1]),  # hip_hop_lofi_boom_bap
        np.full(1100, encoded_labels[2]),  # pop_rock
        np.full(1100, encoded_labels[3])   # edm_tr_909
    ])

    # final_data = np.zeros_like(concatenated_dataset)
    # for idx_range in feature_slices:
    #     # final_data[:, idx_range] = scale_bpm(
    #     #     concatenated_dataset[:, idx_range])
    #     concatenated_dataset[:, idx_range] = 0

    X_train, X_test, y_train, y_test = train_test_split(
        concatenated_dataset, expanded_labels, test_size=0.3, shuffle=True)

    return X_train, X_test, y_train, y_test, label_encoder


def get_feature_vector_for_file(file_path, bpm):
    feature_slices = [
        # slice(0, 13),   # [0:12] 'mfcc_median'
        # slice(13, 26),  # [13:25]'mfcc_skew'
        # slice(26, 39),  # [26:38]'mfcc_var'
        # slice(39, 40),  # [39]  'onset_strength_median'
        # slice(40, 41),  # [40] 'onset_strength_var'
        # slice(41, 42),  # [41] 'rms_energy_median'
        # slice(42, 43),  # [42]  'rms_energy_var'
        # slice(43, 50),  # [43:49] 'spectral_contrast_median'
        # slice(50, 57),  # [50:56]'spectral_contrast_var'
        # slice(57, 58),  # [57]'zcr_median'
        # slice(58, 59),  # [58]'zcr_var'
        # slice(59, 60),  # [59]  'bpm'
    ]
    feature_vector = extract_features(file_path, bpm)
    # feature_vector[59:60] = scale_bpm(np.array([[bpm]]))
    # for idx_range in feature_slices:
    #     # final_data[:, idx_range] = scale_bpm(
    #     #     concatenated_dataset[:, idx_range])
    #     feature_vector[idx_range] = 0

    return feature_vector


def TSNE_plotter(final_data_set):
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(final_data_set[:, 0:59])

    labels = np.array([0] * 1100 + [1] *
                      1100 + [2] * 1100 + [3]*1100)
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels ==
                                                           0, 1], alpha=0.7, label='World Music - Brazilian Samba', color='blue')
    plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels ==
                                                           1, 1], alpha=0.7, label='lofi hip-hop', color='red')
    plt.scatter(tsne_results[labels == 2, 0], tsne_results[labels ==
                                                           2, 1], alpha=0.7, label='EDM - TR909', color='green')
    plt.scatter(tsne_results[labels == 3, 0], tsne_results[labels ==
                                                           3, 1], alpha=0.7, label='Pop - Rock', color='orange')
    plt.title("t-SNE Visualization of Combined Feature Sets")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig("plots/tsne_combined_visualization.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    print("hello")
