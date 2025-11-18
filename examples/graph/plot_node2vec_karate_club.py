"""
=================================================
Node2Vec on Karate Club Network
=================================================

This example demonstrates the use of Node2Vec for learning node embeddings
on Zachary's Karate Club network, a well-known social network dataset.

We visualize the learned embeddings in 2D space and show how nodes from
the same community cluster together.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.graph import Node2Vec

# %%
# Create Zachary's Karate Club network
# -------------------------------------
# The network represents friendships between 34 members of a karate club.
# The network is known to split into two communities.

# Edges of the Karate Club network
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10),
    (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

n_nodes = 34

# Create adjacency matrix
adj_matrix = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1

adj_matrix = sparse.csr_matrix(adj_matrix)

# Known community labels (0: Mr. Hi's group, 1: Officer's group)
true_labels = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

# %%
# Learn node embeddings with Node2Vec
# ------------------------------------
# We use Node2Vec to learn 2D embeddings of the nodes. The p and q parameters
# control the exploration strategy of the random walks.

print("Learning node embeddings...")
node2vec = Node2Vec(
    n_components=16,  # Use higher dimensions for better quality
    walk_length=80,
    num_walks=10,
    p=1.0,
    q=1.0,
    window_size=10,
    n_iter=3,
    random_state=42,
)

embeddings = node2vec.fit_transform(adj_matrix)

# Project to 2D for visualization
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# %%
# Visualize the learned embeddings
# ---------------------------------
# We plot the 2D embeddings, coloring nodes by their true community labels.

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=true_labels,
    cmap="viridis",
    s=200,
    alpha=0.7,
    edgecolors="black",
    linewidth=1.5,
)

# Add node labels
for i, (x, y) in enumerate(embeddings_2d):
    plt.annotate(
        str(i),
        (x, y),
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        weight="bold",
    )

plt.colorbar(scatter, label="Community")
plt.title("Node2Vec Embeddings of Karate Club Network", fontsize=14, weight="bold")
plt.xlabel("First Principal Component", fontsize=12)
plt.ylabel("Second Principal Component", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# The visualization shows that Node2Vec successfully learns embeddings that
# capture the community structure of the network. Nodes from the same community
# (shown by color) tend to cluster together in the embedding space.
