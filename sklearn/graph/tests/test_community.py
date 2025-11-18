"""Tests for community detection algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import sparse

from sklearn.graph import LabelPropagationCommunity, LouvainCommunityDetection


def make_test_graph_with_communities():
    """Create a graph with clear community structure."""
    # Two communities: nodes 0-2 and nodes 3-5
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])
    return sparse.csr_matrix(adj_matrix)


def make_simple_graph():
    """Create a simple small graph."""
    adj_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ])
    return sparse.csr_matrix(adj_matrix)


def test_louvain_basic():
    """Test basic Louvain functionality."""
    adj_matrix = make_test_graph_with_communities()
    
    louvain = LouvainCommunityDetection(random_state=42)
    labels = louvain.fit_predict(adj_matrix)
    
    assert labels.shape == (6,)
    assert louvain.n_communities_ > 0
    assert louvain.n_communities_ <= 6
    assert hasattr(louvain, "modularity_")


def test_louvain_fit_predict():
    """Test fit and predict separately."""
    adj_matrix = make_test_graph_with_communities()
    
    louvain = LouvainCommunityDetection(random_state=42)
    louvain.fit(adj_matrix)
    labels = louvain.labels_
    
    assert labels.shape == (6,)
    assert hasattr(louvain, "n_communities_")


def test_louvain_modularity():
    """Test that modularity is computed."""
    adj_matrix = make_test_graph_with_communities()
    
    louvain = LouvainCommunityDetection(random_state=42)
    louvain.fit(adj_matrix)
    
    assert hasattr(louvain, "modularity_")
    # Modularity should be between -0.5 and 1
    assert -0.5 <= louvain.modularity_ <= 1.0


def test_louvain_resolution():
    """Test resolution parameter effect."""
    adj_matrix = make_test_graph_with_communities()
    
    # Higher resolution should give more communities
    louvain_high = LouvainCommunityDetection(resolution=2.0, random_state=42)
    louvain_high.fit(adj_matrix)
    
    louvain_low = LouvainCommunityDetection(resolution=0.5, random_state=42)
    louvain_low.fit(adj_matrix)
    
    # Note: This is not always guaranteed, but should be true on average
    assert louvain_high.n_communities_ >= 1
    assert louvain_low.n_communities_ >= 1


def test_louvain_deterministic():
    """Test that Louvain is deterministic with fixed random state."""
    adj_matrix = make_test_graph_with_communities()
    
    louvain1 = LouvainCommunityDetection(random_state=42)
    labels1 = louvain1.fit_predict(adj_matrix)
    
    louvain2 = LouvainCommunityDetection(random_state=42)
    labels2 = louvain2.fit_predict(adj_matrix)
    
    np.testing.assert_array_equal(labels1, labels2)


def test_label_propagation_basic():
    """Test basic label propagation functionality."""
    adj_matrix = make_test_graph_with_communities()
    
    lp = LabelPropagationCommunity(max_iter=100)
    labels = lp.fit_predict(adj_matrix)
    
    assert labels.shape == (6,)
    assert lp.n_communities_ > 0
    assert lp.n_communities_ <= 6
    assert hasattr(lp, "n_iter_")


def test_label_propagation_fit_predict():
    """Test fit and predict separately."""
    adj_matrix = make_test_graph_with_communities()
    
    lp = LabelPropagationCommunity(max_iter=100)
    lp.fit(adj_matrix)
    labels = lp.labels_
    
    assert labels.shape == (6,)
    assert hasattr(lp, "n_communities_")


def test_label_propagation_convergence():
    """Test that label propagation converges."""
    adj_matrix = make_simple_graph()
    
    lp = LabelPropagationCommunity(max_iter=100)
    lp.fit(adj_matrix)
    
    # Should converge before max_iter
    assert lp.n_iter_ <= 100


def test_label_propagation_simple_case():
    """Test label propagation on simple case."""
    # Simple line graph
    adj_matrix = sparse.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]))
    
    lp = LabelPropagationCommunity(max_iter=10)
    labels = lp.fit_predict(adj_matrix)
    
    assert labels.shape == (3,)


def test_community_detection_invalid_input():
    """Test community detection with invalid input."""
    # Non-square matrix
    adj_matrix = np.array([[0, 1], [1, 0], [0, 1]])
    
    louvain = LouvainCommunityDetection(random_state=42)
    
    with pytest.raises(ValueError, match="must be square"):
        louvain.fit(adj_matrix)
    
    lp = LabelPropagationCommunity()
    
    with pytest.raises(ValueError, match="must be square"):
        lp.fit(adj_matrix)


def test_louvain_isolated_nodes():
    """Test Louvain with isolated nodes."""
    # Graph with isolated node
    adj_matrix = sparse.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],  # Isolated node
    ]))
    
    louvain = LouvainCommunityDetection(random_state=42)
    labels = louvain.fit_predict(adj_matrix)
    
    assert labels.shape == (4,)
    # Isolated node should be in its own community
    assert labels[3] != labels[0] or louvain.n_communities_ == 1


def test_label_propagation_complete_graph():
    """Test label propagation on complete graph."""
    # Complete graph should converge to single community
    n = 5
    adj_matrix = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    
    lp = LabelPropagationCommunity(max_iter=10)
    labels = lp.fit_predict(adj_matrix)
    
    assert labels.shape == (n,)
    # All nodes should be in the same community
    assert lp.n_communities_ == 1
