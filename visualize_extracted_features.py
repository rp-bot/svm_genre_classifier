import numpy as np
from pprint import pprint
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sm_features_labels = np.load('data/wrld_smb_drm_features_and_labels.npz')

hh_features_labels = np.load("data/hh_lfbb_lps_mid_001-009.npz")

tr9_feature_labels = np.load("data/edm_tr9_drm_id_001.npz")

sm_data = sm_features_labels['features']
hh_data = hh_features_labels['features']
tr9_data = tr9_feature_labels['features']

indices_hh = np.linspace(0, hh_data.shape[0] - 1, 1100, dtype=int)
hh_sampled = hh_data[indices_hh]

indices_tr9 = np.linspace(0, tr9_data.shape[0] - 1, 1100, dtype=int)
tr9_sampled = tr9_data[indices_tr9]

combined_matrix = np.vstack([sm_data, hh_sampled, tr9_sampled])
labels = np.array([0] * len(sm_data) + [1] *
                  len(hh_sampled) + [2] * len(tr9_sampled))

pca = PCA(n_components=2)
pca_results = pca.fit_transform(combined_matrix)

plt.figure(figsize=(10, 7))
plt.scatter(pca_results[labels == 0, 0], pca_results[labels ==
            0, 1], alpha=0.7, label='World Music - Brazilian Samba', color='blue')
plt.scatter(pca_results[labels == 1, 0], pca_results[labels ==
            1, 1], alpha=0.7, label='lofi hip-hop', color='red')
plt.scatter(pca_results[labels == 2, 0], pca_results[labels ==
            2, 1], alpha=0.7, label='EDM - TR909', color='green')
plt.title("PCA Visualization of Combined Feature Sets")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.savefig("plots/pca_combined_visualization.png", dpi=300)
plt.close()

tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
tsne_results = tsne.fit_transform(combined_matrix)

plt.figure(figsize=(10, 7))
plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels ==
            0, 1], alpha=0.7, label='World Music - Brazilian Samba', color='blue')
plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels ==
            1, 1], alpha=0.7, label='lofi hip-hop', color='red')
plt.scatter(tsne_results[labels == 2, 0], tsne_results[labels ==
            2, 1], alpha=0.7, label='EDM - TR909', color='green')
plt.title("t-SNE Visualization of Combined Feature Sets")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("plots/tsne_combined_visualization.png", dpi=300)
plt.close()

corr_1 = np.corrcoef(sm_data, rowvar=False)
corr_2 = np.corrcoef(hh_sampled, rowvar=False)
corr_3 = np.corrcoef(tr9_sampled, rowvar=False)

plt.figure(figsize=(12, 6))
sns.heatmap(corr_1, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix - World Music - Brazilian Samba")
plt.savefig("plots/corr_matrix_1.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.heatmap(corr_2, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix - lofi hip-hop")
plt.savefig("plots/corr_matrix_2.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.heatmap(corr_3, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix - EDM - TR909")
plt.savefig("plots/corr_matrix_3.png", dpi=300)
plt.close()
