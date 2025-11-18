# Graph Machine Learning Implementation

This document describes the new graph machine learning capabilities added to scikit-learn.

## Overview

A comprehensive graph machine learning module has been added to scikit-learn at `sklearn.graph`. This module provides state-of-the-art algorithms for learning on graph-structured data, including node embeddings, graph kernels, and community detection.

## New Components

### 1. Node Embeddings (`sklearn.graph._node_embeddings.py`)

#### Node2Vec
- Implements the Node2Vec algorithm for learning node embeddings
- Supports biased random walks with configurable p and q parameters
- Uses skip-gram with negative sampling for training
- **Input**: Graph adjacency matrix (sparse)
- **Output**: Node embedding matrix (n_nodes × n_components)

#### DeepWalk
- Implements the DeepWalk algorithm (special case of Node2Vec)
- Uses unbiased random walks
- Simpler and faster than Node2Vec when no bias is needed
- **Input**: Graph adjacency matrix (sparse)
- **Output**: Node embedding matrix (n_nodes × n_components)

### 2. Graph Kernels (`sklearn.graph._graph_kernels.py`)

#### WeisfeilerLehmanKernel
- Implements the Weisfeiler-Lehman graph kernel
- Iteratively refines node labels based on neighborhood structure
- Creates feature vectors that capture multi-scale graph structure
- **Input**: List of graph adjacency matrices
- **Output**: Feature matrix (n_graphs × n_features)

#### RandomWalkKernel
- Implements the random walk graph kernel
- Compares graphs based on random walk patterns
- Supports configurable walk length and decay parameters
- **Input**: List of graph adjacency matrices
- **Output**: Feature matrix (n_graphs × n_features)

### 3. Community Detection (`sklearn.graph._community.py`)

#### LouvainCommunityDetection
- Implements the Louvain method for community detection
- Optimizes modularity through multi-level refinement
- Supports configurable resolution parameter
- **Input**: Graph adjacency matrix (sparse)
- **Output**: Community labels for each node

#### LabelPropagationCommunity
- Implements label propagation for community detection
- Fast algorithm suitable for large networks
- Propagates labels through network until convergence
- **Input**: Graph adjacency matrix (sparse)
- **Output**: Community labels for each node

## Module Structure

```
sklearn/graph/
├── __init__.py                  # Module initialization
├── _node_embeddings.py          # Node2Vec and DeepWalk
├── _graph_kernels.py            # Weisfeiler-Lehman and Random Walk kernels
├── _community.py                # Community detection algorithms
├── README.md                    # Module documentation
├── meson.build                  # Build configuration
└── tests/                       # Test suite
    ├── __init__.py
    ├── test_node_embeddings.py
    ├── test_graph_kernels.py
    ├── test_community.py
    └── meson.build
```

## Examples

Three comprehensive examples have been added to demonstrate the new capabilities:

### 1. `examples/graph/plot_node2vec_karate_club.py`
- Demonstrates Node2Vec on Zachary's Karate Club network
- Shows how to learn 2D embeddings for visualization
- Illustrates community structure discovery through embeddings

### 2. `examples/graph/plot_community_detection.py`
- Demonstrates Louvain community detection on synthetic networks
- Shows effect of resolution parameter
- Compares detected communities with ground truth

### 3. `examples/graph/plot_graph_kernels.py`
- Demonstrates graph classification using Weisfeiler-Lehman kernel
- Creates synthetic molecular-like graphs
- Shows complete ML pipeline: kernel → features → SVM classification

## Integration

The module has been integrated into scikit-learn:

1. **Added to module list**: `sklearn/__init__.py` updated to include "graph"
2. **Build system**: `sklearn/meson.build` updated to compile the graph submodule
3. **Tests**: Comprehensive test suite with 30+ test cases
4. **Documentation**: README with API documentation and usage examples

## API Design

All algorithms follow scikit-learn conventions:

- **Estimator API**: All classes inherit from `BaseEstimator`
- **Transformers**: Node embeddings and graph kernels use `TransformerMixin`
- **Clustering**: Community detection uses `ClusterMixin`
- **Parameter validation**: Uses `_parameter_constraints` for input validation
- **Random state**: Supports reproducibility through `random_state` parameter
- **Fitted attributes**: Follow scikit-learn convention (trailing underscore)

## Use Cases

### Node Embeddings
- **Node classification**: Classify nodes in a network
- **Link prediction**: Predict missing or future links
- **Visualization**: 2D/3D visualization of network structure
- **Transfer learning**: Pre-trained embeddings for downstream tasks

### Graph Kernels
- **Molecular property prediction**: Classify molecules based on structure
- **Protein function prediction**: Classify proteins by structure
- **Social network classification**: Categorize social networks
- **Program analysis**: Classify code by control flow graphs

### Community Detection
- **Social network analysis**: Find communities in social networks
- **Biological networks**: Identify functional modules in protein networks
- **Citation analysis**: Find research communities
- **Infrastructure networks**: Identify clusters in transportation/utility networks

## Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Scalability |
|-----------|----------------|------------------|-------------|
| Node2Vec | O(W×L×N×I) | O(N×D) | ~10K nodes |
| DeepWalk | O(W×L×N×I) | O(N×D) | ~10K nodes |
| WL Kernel | O(G×N²×I) | O(G×F) | <1K graphs |
| RW Kernel | O(G×N²×L) | O(G×L) | <1K graphs |
| Louvain | O(E×log N) | O(N) | Millions |
| LabelProp | O(E×I) | O(N) | Millions |

Where:
- W = number of walks, L = walk length, N = nodes, I = iterations
- G = number of graphs, F = features, E = edges, D = embedding dimension

## Testing

Comprehensive test suite with 30+ tests covering:
- Basic functionality for all algorithms
- Parameter validation and error handling
- Edge cases (isolated nodes, disconnected graphs, etc.)
- Determinism with fixed random state
- Different input formats (sparse matrices, dense arrays)
- Transform before fit error checking

Run tests with:
```bash
pytest sklearn/graph/tests/
```

## Future Enhancements

Potential additions to the module:
1. **Graph Neural Networks**: GCN, GAT, GraphSAGE implementations
2. **Additional kernels**: Shortest path kernel, subtree kernel
3. **More embeddings**: LINE, Struc2Vec, metapath2vec
4. **Graph generation**: Random graph models, generative models
5. **Temporal graphs**: Dynamic community detection, temporal walks
6. **Heterogeneous graphs**: Support for multiple node/edge types

## Dependencies

The module uses only standard scikit-learn dependencies:
- NumPy (for numerical operations)
- SciPy (for sparse matrix operations)
- scikit-learn base classes and utilities

No additional dependencies required.

## Compatibility

- Python: 3.11+
- NumPy: 1.19+
- SciPy: 1.5+
- Follows scikit-learn compatibility guidelines

## References

See `sklearn/graph/README.md` for complete list of academic references.

---

> This pull request includes code written with the assistance of AI.
> The code has **not yet been reviewed** by a human.
