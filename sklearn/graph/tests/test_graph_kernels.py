"""Tests for graph kernel algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import sparse

from sklearn.graph import RandomWalkKernel, WeisfeilerLehmanKernel


def make_test_graphs():
    """Create simple test graphs."""
    # Graph 1: Triangle
    graph1 = sparse.csr_matrix(np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]))
    
    # Graph 2: Square
    graph2 = sparse.csr_matrix(np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]))
    
    # Graph 3: Line
    graph3 = sparse.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]))
    
    return [graph1, graph2, graph3]


def test_weisfeiler_lehman_basic():
    """Test basic Weisfeiler-Lehman kernel functionality."""
    graphs = make_test_graphs()
    
    kernel = WeisfeilerLehmanKernel(n_iter=2)
    features = kernel.fit_transform(graphs)
    
    assert features.shape[0] == 3
    assert features.shape[1] > 0
    assert not np.isnan(features).any()


def test_weisfeiler_lehman_fit_transform():
    """Test fit and transform separately."""
    graphs = make_test_graphs()
    
    kernel = WeisfeilerLehmanKernel(n_iter=2)
    kernel.fit(graphs)
    features = kernel.transform(graphs)
    
    assert features.shape[0] == 3
    assert hasattr(kernel, "label_to_idx_")


def test_weisfeiler_lehman_single_graph():
    """Test with a single graph."""
    graph = make_test_graphs()[0]
    
    kernel = WeisfeilerLehmanKernel(n_iter=2)
    features = kernel.fit_transform([graph])
    
    assert features.shape == (1, kernel.n_features_out_)


def test_weisfeiler_lehman_normalization():
    """Test normalization option."""
    graphs = make_test_graphs()
    
    # With normalization
    kernel_norm = WeisfeilerLehmanKernel(n_iter=2, normalize=True)
    features_norm = kernel_norm.fit_transform(graphs)
    
    # Check that rows are normalized
    norms = np.linalg.norm(features_norm, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(3))
    
    # Without normalization
    kernel_no_norm = WeisfeilerLehmanKernel(n_iter=2, normalize=False)
    features_no_norm = kernel_no_norm.fit_transform(graphs)
    
    # Features should be different
    assert not np.allclose(features_norm, features_no_norm)


def test_weisfeiler_lehman_iterations():
    """Test effect of iteration parameter."""
    graphs = make_test_graphs()
    
    kernel1 = WeisfeilerLehmanKernel(n_iter=1)
    features1 = kernel1.fit_transform(graphs)
    
    kernel3 = WeisfeilerLehmanKernel(n_iter=3)
    features3 = kernel3.fit_transform(graphs)
    
    # More iterations should give more features
    assert kernel3.n_features_out_ >= kernel1.n_features_out_


def test_random_walk_kernel_basic():
    """Test basic random walk kernel functionality."""
    graphs = make_test_graphs()
    
    kernel = RandomWalkKernel(walk_length=3)
    features = kernel.fit_transform(graphs)
    
    assert features.shape == (3, 4)  # walk_length + 1
    assert not np.isnan(features).any()


def test_random_walk_kernel_fit_transform():
    """Test fit and transform separately."""
    graphs = make_test_graphs()
    
    kernel = RandomWalkKernel(walk_length=3)
    kernel.fit(graphs)
    features = kernel.transform(graphs)
    
    assert features.shape[0] == 3
    assert hasattr(kernel, "walk_counts_")


def test_random_walk_kernel_parameters():
    """Test random walk kernel with different parameters."""
    graphs = make_test_graphs()
    
    kernel = RandomWalkKernel(walk_length=5, lambda_decay=0.05)
    features = kernel.fit_transform(graphs)
    
    assert features.shape == (3, 6)  # walk_length + 1


def test_random_walk_kernel_normalization():
    """Test normalization option."""
    graphs = make_test_graphs()
    
    # With normalization
    kernel_norm = RandomWalkKernel(walk_length=3, normalize=True)
    features_norm = kernel_norm.fit_transform(graphs)
    
    # Check that rows are normalized
    norms = np.linalg.norm(features_norm, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(3))


def test_random_walk_kernel_single_graph():
    """Test with a single graph."""
    graph = make_test_graphs()[0]
    
    kernel = RandomWalkKernel(walk_length=3)
    features = kernel.fit_transform([graph])
    
    assert features.shape == (1, 4)


def test_graph_kernel_invalid_input():
    """Test graph kernels with invalid input."""
    # Non-square matrix
    graph = np.array([[0, 1], [1, 0], [0, 1]])
    
    kernel = WeisfeilerLehmanKernel()
    
    with pytest.raises(ValueError, match="must be square"):
        kernel.fit([graph])


def test_graph_kernel_transform_before_fit():
    """Test that transform fails before fit."""
    graphs = make_test_graphs()
    
    kernel = WeisfeilerLehmanKernel()
    
    with pytest.raises(ValueError, match="This .* instance is not fitted yet"):
        kernel.transform(graphs)
