"""
=================================================
Community Detection with Louvain Algorithm
=================================================

This example demonstrates community detection using the Louvain algorithm
on a synthetic network with clear community structure.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from sklearn.graph import LouvainCommunityDetection

# %%
# Create a synthetic network with community structure
# ---------------------------------------------------
# We create a network with 4 communities, where nodes within communities
# are densely connected, and connections between communities are sparse.

np.random.seed(42)

n_communities = 4
nodes_per_community = 10
n_nodes = n_communities * nodes_per_community

# Probability of edge within community
p_within = 0.4
# Probability of edge between communities
p_between = 0.05

# Create adjacency matrix
adj_matrix = np.zeros((n_nodes, n_nodes))

for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        comm_i = i // nodes_per_community
        comm_j = j // nodes_per_community
        
        if comm_i == comm_j:
            # Same community
            if np.random.rand() < p_within:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        else:
            # Different communities
            if np.random.rand() < p_between:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

adj_matrix = sparse.csr_matrix(adj_matrix)

# Ground truth labels
true_labels = np.repeat(np.arange(n_communities), nodes_per_community)

# %%
# Detect communities using Louvain algorithm
# ------------------------------------------

print("Detecting communities...")
louvain = LouvainCommunityDetection(resolution=1.0, random_state=42)
detected_labels = louvain.fit_predict(adj_matrix)

print(f"Number of detected communities: {louvain.n_communities_}")
print(f"Modularity: {louvain.modularity_:.3f}")

# %%
# Compute accuracy metrics
# ------------------------
# We compute the Adjusted Rand Index to measure how well the detected
# communities match the ground truth.

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(true_labels, detected_labels)
print(f"Adjusted Rand Index: {ari:.3f}")

# %%
# Visualize the network and detected communities
# ----------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Function to plot network
def plot_network(ax, labels, title):
    """Plot network as a grid with community colors."""
    # Create a 2D layout (grid)
    nodes_per_row = int(np.ceil(np.sqrt(n_nodes)))
    positions = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        positions[i] = [i % nodes_per_row, i // nodes_per_row]
    
    # Plot edges
    edges_i, edges_j = adj_matrix.nonzero()
    for i, j in zip(edges_i, edges_j):
        if i < j:  # Plot each edge only once
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                "k-",
                alpha=0.2,
                linewidth=0.5,
            )
    
    # Plot nodes
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=labels,
        cmap="tab10",
        s=200,
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5,
    )
    
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return scatter

# Plot ground truth
scatter1 = plot_network(axes[0], true_labels, "Ground Truth Communities")
plt.colorbar(scatter1, ax=axes[0], label="Community ID")

# Plot detected communities
scatter2 = plot_network(axes[1], detected_labels, "Detected Communities (Louvain)")
plt.colorbar(scatter2, ax=axes[1], label="Community ID")

plt.tight_layout()
plt.show()

# %%
# Compare different resolution parameters
# ---------------------------------------
# The resolution parameter controls the size of communities. Higher values
# lead to more (smaller) communities.

resolutions = [0.5, 1.0, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, resolution in zip(axes, resolutions):
    louvain = LouvainCommunityDetection(resolution=resolution, random_state=42)
    labels = louvain.fit_predict(adj_matrix)
    
    plot_network(ax, labels, f"Resolution = {resolution}")
    ax.text(
        0.05,
        0.95,
        f"Communities: {louvain.n_communities_}\nModularity: {louvain.modularity_:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
    )

plt.tight_layout()
plt.show()

# %%
# The Louvain algorithm successfully identifies the community structure in
# the network. The resolution parameter allows fine-tuning the granularity
# of the detected communities.
