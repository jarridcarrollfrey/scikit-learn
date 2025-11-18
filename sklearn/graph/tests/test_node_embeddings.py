"""Tests for node embedding algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import sparse

from sklearn.graph import DeepWalk, Node2Vec


def make_test_graph():
    """Create a simple test graph."""
    # Simple graph with 6 nodes and clear structure
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])
    return sparse.csr_matrix(adj_matrix)


def test_node2vec_basic():
    """Test basic Node2Vec functionality."""
    adj_matrix = make_test_graph()
    
    model = Node2Vec(
        n_components=8,
        walk_length=10,
        num_walks=5,
        random_state=42
    )
    embeddings = model.fit_transform(adj_matrix)
    
    assert embeddings.shape == (6, 8)
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


def test_node2vec_fit_transform():
    """Test fit and transform separately."""
    adj_matrix = make_test_graph()
    
    model = Node2Vec(n_components=4, num_walks=3, random_state=42)
    model.fit(adj_matrix)
    embeddings = model.transform()
    
    assert embeddings.shape == (6, 4)
    assert hasattr(model, "embeddings_")


def test_node2vec_parameters():
    """Test Node2Vec with different parameters."""
    adj_matrix = make_test_graph()
    
    # Test with different p and q values
    model = Node2Vec(
        n_components=4,
        p=0.5,
        q=2.0,
        walk_length=15,
        num_walks=5,
        random_state=42
    )
    embeddings = model.fit_transform(adj_matrix)
    
    assert embeddings.shape == (6, 4)


def test_node2vec_invalid_input():
    """Test Node2Vec with invalid input."""
    # Non-square matrix
    adj_matrix = np.array([[0, 1], [1, 0], [0, 1]])
    
    model = Node2Vec(random_state=42)
    
    with pytest.raises(ValueError, match="must be square"):
        model.fit(adj_matrix)


def test_deepwalk_basic():
    """Test basic DeepWalk functionality."""
    adj_matrix = make_test_graph()
    
    model = DeepWalk(
        n_components=8,
        walk_length=10,
        num_walks=5,
        random_state=42
    )
    embeddings = model.fit_transform(adj_matrix)
    
    assert embeddings.shape == (6, 8)
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


def test_deepwalk_fit_transform():
    """Test DeepWalk fit and transform separately."""
    adj_matrix = make_test_graph()
    
    model = DeepWalk(n_components=4, num_walks=3, random_state=42)
    model.fit(adj_matrix)
    embeddings = model.transform()
    
    assert embeddings.shape == (6, 4)
    assert hasattr(model, "embeddings_")


def test_deepwalk_deterministic():
    """Test that DeepWalk is deterministic with fixed random state."""
    adj_matrix = make_test_graph()
    
    model1 = DeepWalk(n_components=4, num_walks=3, random_state=42)
    embeddings1 = model1.fit_transform(adj_matrix)
    
    model2 = DeepWalk(n_components=4, num_walks=3, random_state=42)
    embeddings2 = model2.fit_transform(adj_matrix)
    
    np.testing.assert_array_almost_equal(embeddings1, embeddings2)


def test_node2vec_sparse_input():
    """Test Node2Vec with sparse matrix input."""
    adj_matrix = make_test_graph()
    
    model = Node2Vec(n_components=4, num_walks=3, random_state=42)
    embeddings = model.fit_transform(adj_matrix)
    
    assert embeddings.shape == (6, 4)


def test_node2vec_disconnected_graph():
    """Test Node2Vec with disconnected graph."""
    # Graph with isolated node
    adj_matrix = sparse.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]))
    
    model = Node2Vec(n_components=4, walk_length=5, num_walks=2, random_state=42)
    embeddings = model.fit_transform(adj_matrix)
    
    assert embeddings.shape == (4, 4)
