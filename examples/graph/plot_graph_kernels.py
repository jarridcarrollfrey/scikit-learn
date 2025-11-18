"""
=================================================
Graph Classification with Graph Kernels
=================================================

This example demonstrates the use of graph kernels for graph classification.
We create synthetic molecular-like graphs and classify them using the
Weisfeiler-Lehman kernel combined with an SVM classifier.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from sklearn.graph import WeisfeilerLehmanKernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# %%
# Generate synthetic graph dataset
# ---------------------------------
# We create two classes of graphs with different structural properties.


def create_star_graph(n_nodes, add_noise=False):
    """Create a star graph (one center connected to all others)."""
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes):
        adj_matrix[0, i] = 1
        adj_matrix[i, 0] = 1
    
    # Add some noise edges
    if add_noise:
        n_noise = max(1, n_nodes // 5)
        for _ in range(n_noise):
            i, j = np.random.randint(1, n_nodes, 2)
            if i != j:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    return sparse.csr_matrix(adj_matrix)


def create_cycle_graph(n_nodes, add_noise=False):
    """Create a cycle graph (nodes connected in a ring)."""
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Add some noise edges
    if add_noise:
        n_noise = max(1, n_nodes // 5)
        for _ in range(n_noise):
            i, j = np.random.randint(0, n_nodes, 2)
            if i != j and abs(i - j) > 1 and abs(i - j) < n_nodes - 1:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    return sparse.csr_matrix(adj_matrix)


# Generate dataset
np.random.seed(42)
n_graphs_per_class = 50
min_nodes = 5
max_nodes = 15

graphs = []
labels = []

# Class 0: Star graphs
for _ in range(n_graphs_per_class):
    n_nodes = np.random.randint(min_nodes, max_nodes + 1)
    graph = create_star_graph(n_nodes, add_noise=True)
    graphs.append(graph)
    labels.append(0)

# Class 1: Cycle graphs
for _ in range(n_graphs_per_class):
    n_nodes = np.random.randint(min_nodes, max_nodes + 1)
    graph = create_cycle_graph(n_nodes, add_noise=True)
    graphs.append(graph)
    labels.append(1)

labels = np.array(labels)

print(f"Generated {len(graphs)} graphs")
print(f"Class distribution: {np.bincount(labels)}")

# %%
# Split data and compute graph features
# --------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    graphs, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Compute Weisfeiler-Lehman features
print("\nComputing Weisfeiler-Lehman features...")
wl_kernel = WeisfeilerLehmanKernel(n_iter=3, normalize=True)
X_train_features = wl_kernel.fit_transform(X_train)
X_test_features = wl_kernel.transform(X_test)

print(f"Feature dimension: {X_train_features.shape[1]}")

# %%
# Train SVM classifier
# --------------------

print("\nTraining SVM classifier...")
clf = SVC(kernel="rbf", C=1.0, random_state=42)
clf.fit(X_train_features, y_train)

train_score = clf.score(X_train_features, y_train)
test_score = clf.score(X_test_features, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# %%
# Visualize example graphs from each class
# -----------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for class_idx in range(2):
    # Get examples from this class
    class_graphs = [graphs[i] for i in range(len(graphs)) if labels[i] == class_idx]
    
    for col in range(4):
        ax = axes[class_idx, col]
        
        if col < len(class_graphs):
            graph = class_graphs[col]
            n_nodes = graph.shape[0]
            
            # Create circular layout
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            positions = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Plot edges
            edges_i, edges_j = graph.nonzero()
            for i, j in zip(edges_i, edges_j):
                if i < j:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        "k-",
                        alpha=0.5,
                        linewidth=1.5,
                    )
            
            # Plot nodes
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c="lightblue" if class_idx == 0 else "lightcoral",
                s=300,
                edgecolors="black",
                linewidth=2,
                zorder=3,
            )
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            
            if col == 0:
                class_name = "Star Graphs" if class_idx == 0 else "Cycle Graphs"
                ax.set_title(f"{class_name}\n({n_nodes} nodes)", fontsize=11, weight="bold")
            else:
                ax.set_title(f"({n_nodes} nodes)", fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Compare different kernel parameters
# -----------------------------------

n_iters = [1, 2, 3, 5]
scores = []

for n_iter in n_iters:
    wl = WeisfeilerLehmanKernel(n_iter=n_iter, normalize=True)
    X_train_feat = wl.fit_transform(X_train)
    X_test_feat = wl.transform(X_test)
    
    clf = SVC(kernel="rbf", C=1.0, random_state=42)
    clf.fit(X_train_feat, y_train)
    
    test_score = clf.score(X_test_feat, y_test)
    scores.append(test_score)
    print(f"WL iterations: {n_iter}, Test accuracy: {test_score:.3f}")

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(n_iters, scores, "o-", linewidth=2, markersize=8)
plt.xlabel("Number of WL Iterations", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Effect of WL Iterations on Classification Accuracy", fontsize=14, weight="bold")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# %%
# The Weisfeiler-Lehman kernel successfully captures structural differences
# between star and cycle graphs, achieving high classification accuracy.
# The number of WL iterations controls how much global structure is captured.
