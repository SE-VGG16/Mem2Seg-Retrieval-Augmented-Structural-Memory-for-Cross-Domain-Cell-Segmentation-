from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap


# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_PATH = ROOT / "tools" / "outputs" / "all_embeddings.pt"
SAVE_DIR = ROOT / "tools" / "outputs" / "pmb_visualization"

K = 64
MAX_POINTS = 2500
SEED = 42

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD EMBEDDINGS
# =========================

embeddings = torch.load(EMBEDDING_PATH, map_location="cpu")

if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.numpy()

embeddings = embeddings.astype(np.float32)

# normalize embeddings
embeddings = embeddings / (
    np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
)

print("All embeddings:", embeddings.shape)


# =========================
# SAMPLE POINTS
# =========================

rng = np.random.default_rng(SEED)

num_points = min(MAX_POINTS, len(embeddings))
sample_idx = rng.choice(len(embeddings), num_points, replace=False)

X = embeddings[sample_idx]

print("Sampled embeddings:", X.shape)


# =========================
# K-MEANS CLUSTERING
# =========================

print("Running K-means...")

kmeans = KMeans(
    n_clusters=K,
    random_state=SEED,
    n_init=20,
    max_iter=500
)

cluster_ids = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

centroids = centroids / (
    np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
)

print("Centroids:", centroids.shape)


# =========================
# UMAP REDUCTION
# =========================

print("Running UMAP...")

all_features = np.concatenate([X, centroids], axis=0)

reducer = umap.UMAP(
    n_components=2,

    # smaller neighbors make more local, separated groups
    n_neighbors=6,

    # larger min_dist spreads clusters apart
    min_dist=0.85,

    # larger spread increases global spacing
    spread=3.5,

    metric="cosine",
    random_state=SEED,
    init="spectral",
    low_memory=False,
)

all_2d = reducer.fit_transform(all_features)

points_2d = all_2d[:len(X)]
centroids_2d = all_2d[len(X):]


# =========================
# OPTIONAL EXTRA SPACING
# =========================
# This makes the final plot visually more separated.
# It does NOT change clustering results.

points_2d = points_2d * 1.25
centroids_2d = centroids_2d * 1.25


# =========================
# PLOT
# =========================

print("Plotting...")

plt.figure(figsize=(14, 12))

scatter = plt.scatter(
    points_2d[:, 0],
    points_2d[:, 1],
    c=cluster_ids,
    cmap="turbo",
    s=70,                          #for better points vision
    alpha=0.82,
    linewidths=0.2,
    edgecolors="black",
)

plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    marker="*",
    s=760,
    c="black",
    edgecolors="white",
    linewidths=2.2,
    label="Prototype centroids",
    zorder=100,
)

# centroid labels
for i, (x, y) in enumerate(centroids_2d):
    plt.text(
        x + 0.08,
        y + 0.08,
        f"p{i}",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=101,
    )

plt.title(
    "UMAP Visualization of Prototype Memory Bank Clusters (K=64)",
    fontsize=18,
    fontweight="bold",
)

plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)

plt.grid(alpha=0.18)
plt.legend(fontsize=12, loc="best")

cbar = plt.colorbar(scatter, fraction=0.035, pad=0.02)
cbar.set_label("Cluster ID", fontsize=12)

plt.tight_layout()

save_path = SAVE_DIR / "pmb_umap_k64_clusters_far.png"
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", save_path)