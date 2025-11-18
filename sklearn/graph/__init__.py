"""Graph machine learning algorithms.

This module provides algorithms for machine learning on graph-structured data,
including graph embeddings, graph kernels, and community detection.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.graph._community import (
    LabelPropagationCommunity,
    LouvainCommunityDetection,
)
from sklearn.graph._graph_kernels import RandomWalkKernel, WeisfeilerLehmanKernel
from sklearn.graph._node_embeddings import DeepWalk, Node2Vec

__all__ = [
    "DeepWalk",
    "LabelPropagationCommunity",
    "LouvainCommunityDetection",
    "Node2Vec",
    "RandomWalkKernel",
    "WeisfeilerLehmanKernel",
]
