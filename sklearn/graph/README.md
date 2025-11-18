# Graph Machine Learning Module

This module provides machine learning algorithms specifically designed for graph-structured data. It includes methods for node embeddings, graph kernels, and community detection.

## Features

### Node Embeddings

Convert nodes in a graph into low-dimensional vector representations that preserve graph structure.

#### Node2Vec
- **Description**: Learns continuous feature representations for nodes using biased random walks
- **Key Parameters**: 
  - `p`: Return parameter (controls likelihood of revisiting nodes)
  - `q`: In-out parameter (controls exploration vs exploitation)
  - `n_components`: Embedding dimension
- **Use Cases**: Node classification, link prediction, visualization

#### DeepWalk
- **Description**: Special case of Node2Vec with unbiased random walks (p=1, q=1)
- **Use Cases**: Simpler alternative to Node2Vec when no bias is needed

### Graph Kernels

Transform entire graphs into feature vectors for graph-level classification tasks.

#### Weisfeiler-Lehman Kernel
- **Description**: Captures graph structure through iterative node label refinement
- **Key Parameters**:
  - `n_iter`: Number of refinement iterations
  - `normalize`: Whether to normalize feature vectors
- **Use Cases**: Graph classification, graph similarity comparison

#### Random Walk Kernel
- **Description**: Compares graphs based on matching random walk patterns
- **Key Parameters**:
  - `walk_length`: Maximum walk length to consider
  - `lambda_decay`: Decay factor for longer walks
- **Use Cases**: Graph classification, molecular property prediction

### Community Detection

Identify groups of densely connected nodes in networks.

#### Louvain Community Detection
- **Description**: Greedy modularity optimization for finding communities
- **Key Parameters**:
  - `resolution`: Controls community size (higher = more communities)
  - `max_iter`: Maximum optimization iterations
- **Use Cases**: Social network analysis, biological network analysis

#### Label Propagation Community
- **Description**: Fast community detection through label diffusion
- **Key Parameters**:
  - `max_iter`: Maximum propagation iterations
- **Use Cases**: Large-scale networks where speed is critical

## Usage Examples

### Node Embeddings

```python
import numpy as np
from scipy import sparse
from sklearn.graph import Node2Vec

# Create a graph adjacency matrix
adj_matrix = sparse.csr_matrix(np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
]))

# Learn embeddings
model = Node2Vec(n_components=8, random_state=42)
embeddings = model.fit_transform(adj_matrix)
```

### Graph Classification

```python
from sklearn.graph import WeisfeilerLehmanKernel
from sklearn.svm import SVC

# List of graph adjacency matrices
graphs = [graph1, graph2, graph3, ...]
labels = [0, 1, 0, ...]

# Transform graphs to features
kernel = WeisfeilerLehmanKernel(n_iter=3)
features = kernel.fit_transform(graphs)

# Train classifier
clf = SVC()
clf.fit(features, labels)
```

### Community Detection

```python
from sklearn.graph import LouvainCommunityDetection

# Detect communities
louvain = LouvainCommunityDetection(resolution=1.0, random_state=42)
communities = louvain.fit_predict(adj_matrix)

print(f"Found {louvain.n_communities_} communities")
print(f"Modularity: {louvain.modularity_:.3f}")
```

## Input Format

All algorithms expect graphs as sparse adjacency matrices:
- **Shape**: (n_nodes, n_nodes) for single graphs
- **Format**: Scipy sparse matrix (CSR, CSC, or COO)
- **Values**: Binary (0/1) for unweighted graphs, positive for weighted graphs
- **Symmetry**: Should be symmetric for undirected graphs

For graph kernels, provide a list of adjacency matrices:
```python
graphs = [sparse.csr_matrix(adj1), sparse.csr_matrix(adj2), ...]
```

## API Design

All estimators follow scikit-learn conventions:
- **Transformers**: `fit()`, `transform()`, `fit_transform()`
- **Predictors**: `fit()`, `predict()`, `fit_predict()`
- **Attributes**: End with underscore (e.g., `embeddings_`, `labels_`)
- **Parameters**: Can be inspected and set via `get_params()` and `set_params()`

## Performance Considerations

### Node Embeddings
- **Time Complexity**: O(num_walks × walk_length × n_nodes × n_iter)
- **Space Complexity**: O(n_nodes × n_components)
- **Recommendation**: For large graphs (>10K nodes), reduce `num_walks` or `walk_length`

### Graph Kernels
- **Time Complexity**: O(n_graphs × n_nodes² × n_iter) for WL kernel
- **Space Complexity**: O(n_graphs × n_features)
- **Recommendation**: Use graph kernels for datasets with <1000 graphs

### Community Detection
- **Time Complexity**: O(n_edges × log(n_nodes)) for Louvain
- **Space Complexity**: O(n_nodes)
- **Recommendation**: Louvain scales to millions of nodes

## References

### Node2Vec
Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. 
In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery 
and data mining (pp. 855-864).

### DeepWalk
Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). Deepwalk: Online learning of social 
representations. In Proceedings of the 20th ACM SIGKDD international conference on 
Knowledge discovery and data mining (pp. 701-710).

### Weisfeiler-Lehman Kernel
Shervashidze, N., Schweitzer, P., Leeuwen, E. J. V., Mehlhorn, K., & Borgwardt, K. M. 
(2011). Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 
12(Sep), 2539-2561.

### Random Walk Kernel
Vishwanathan, S. V. N., Schraudolph, N. N., Kondor, R., & Borgwardt, K. M. (2010). 
Graph kernels. Journal of Machine Learning Research, 11(Apr), 1201-1242.

### Louvain Algorithm
Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding 
of communities in large networks. Journal of statistical mechanics: theory and 
experiment, 2008(10), P10008.

### Label Propagation
Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect 
community structures in large-scale networks. Physical review E, 76(3), 036106.

## See Also

- `sklearn.cluster.SpectralClustering`: Graph-based clustering using Laplacian eigenmaps
- `sklearn.semi_supervised.LabelPropagation`: Semi-supervised learning on graphs
- `sklearn.manifold`: Manifold learning algorithms (some graph-based)
