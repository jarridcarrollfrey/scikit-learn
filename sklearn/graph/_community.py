"""Community detection algorithms for graphs."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class LouvainCommunityDetection(BaseEstimator, ClusterMixin):
    """Louvain community detection algorithm.
    
    The Louvain method is a greedy optimization method for detecting communities
    in large networks. It optimizes the modularity measure, which quantifies
    the quality of an assignment of nodes to communities.
    
    Parameters
    ----------
    resolution : float, default=1.0
        Resolution parameter for modularity. Higher values lead to more
        communities, lower values to fewer communities.
        
    max_iter : int, default=100
        Maximum number of iterations at each level.
        
    tol : float, default=1e-7
        Tolerance for convergence. Algorithm stops when modularity improvement
        is below this threshold.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm.
        Pass an int for reproducible results across multiple function calls.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Community label for each node.
        
    modularity_ : float
        Final modularity value.
        
    n_communities_ : int
        Number of communities detected.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import LouvainCommunityDetection
    >>> # Create a graph with clear community structure
    >>> adj_matrix = sparse.csr_matrix(np.array([
    ...     [0, 1, 1, 0, 0, 0],
    ...     [1, 0, 1, 0, 0, 0],
    ...     [1, 1, 0, 1, 0, 0],
    ...     [0, 0, 1, 0, 1, 1],
    ...     [0, 0, 0, 1, 0, 1],
    ...     [0, 0, 0, 1, 1, 0]
    ... ]))
    >>> louvain = LouvainCommunityDetection(random_state=42)
    >>> labels = louvain.fit_predict(adj_matrix)
    >>> louvain.n_communities_
    2
    
    References
    ----------
    .. [1] Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E.
           (2008). Fast unfolding of communities in large networks. Journal of
           statistical mechanics: theory and experiment, 2008(10), P10008.
    """
    
    _parameter_constraints: dict = {
        "resolution": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(self, resolution=1.0, max_iter=100, tol=1e-7, random_state=None):
        self.resolution = resolution
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _compute_modularity(self, adj_matrix, communities, m):
        """Compute modularity of a partition."""
        modularity = 0.0
        
        # Convert to CSR for efficient access
        if not sparse.isspmatrix_csr(adj_matrix):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        # Compute degree for each node
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # Compute modularity
        for i in range(len(communities)):
            for j in range(len(communities)):
                if communities[i] == communities[j]:
                    A_ij = adj_matrix[i, j]
                    expected = (degrees[i] * degrees[j]) / (2 * m)
                    modularity += (A_ij - self.resolution * expected)
        
        return modularity / (2 * m)

    def _move_communities(self, adj_matrix, communities, random_state):
        """Perform one pass of moving nodes to communities."""
        n_nodes = adj_matrix.shape[0]
        
        # Convert to CSR format
        if not sparse.isspmatrix_csr(adj_matrix):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        # Compute total edge weight
        m = adj_matrix.sum() / 2.0
        
        # Compute degree for each node
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # Compute community weights
        community_weights = np.zeros(n_nodes)
        for i, comm in enumerate(communities):
            community_weights[comm] += degrees[i]
        
        improved = False
        nodes = np.arange(n_nodes)
        random_state.shuffle(nodes)
        
        for node in nodes:
            current_comm = communities[node]
            node_degree = degrees[node]
            
            # Remove node from its community
            community_weights[current_comm] -= node_degree
            
            # Compute modularity gain for each neighbor community
            neighbor_communities = set()
            neighbors = adj_matrix[node].nonzero()[1]
            for neighbor in neighbors:
                neighbor_communities.add(communities[neighbor])
            
            best_comm = current_comm
            best_gain = 0.0
            
            for comm in neighbor_communities:
                # Compute weight to community
                weight_to_comm = 0.0
                for neighbor in neighbors:
                    if communities[neighbor] == comm:
                        weight_to_comm += adj_matrix[node, neighbor]
                
                # Compute modularity gain
                gain = (weight_to_comm - self.resolution * node_degree * 
                       community_weights[comm] / (2 * m))
                
                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm
            
            # Move node to best community
            communities[node] = best_comm
            community_weights[best_comm] += node_degree
            
            if best_comm != current_comm:
                improved = True
        
        return communities, improved

    def _aggregate_graph(self, adj_matrix, communities):
        """Aggregate graph based on communities."""
        unique_comms = np.unique(communities)
        n_communities = len(unique_comms)
        
        # Create community mapping
        comm_mapping = {comm: i for i, comm in enumerate(unique_comms)}
        new_communities = np.array([comm_mapping[c] for c in communities])
        
        # Create aggregated adjacency matrix
        new_adj = np.zeros((n_communities, n_communities))
        
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    comm_i = new_communities[i]
                    comm_j = new_communities[j]
                    new_adj[comm_i, comm_j] += adj_matrix[i, j]
        
        return sparse.csr_matrix(new_adj), new_communities

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Detect communities in the graph.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse=True)
        
        if X.shape[0] != X.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        random_state = check_random_state(self.random_state)
        
        # Initialize each node in its own community
        n_nodes = X.shape[0]
        communities = np.arange(n_nodes)
        
        # Store original node to community mapping
        node_to_comm = np.arange(n_nodes)
        
        current_graph = X
        m = X.sum() / 2.0
        
        # Multi-level optimization
        for _ in range(self.max_iter):
            # Phase 1: Move nodes to communities
            old_modularity = self._compute_modularity(current_graph, communities, m)
            
            for _ in range(self.max_iter):
                communities, improved = self._move_communities(
                    current_graph, communities, random_state
                )
                
                if not improved:
                    break
            
            # Check for convergence
            new_modularity = self._compute_modularity(current_graph, communities, m)
            
            if new_modularity - old_modularity < self.tol:
                break
            
            # Update node to community mapping
            for i in range(len(node_to_comm)):
                node_to_comm[i] = communities[node_to_comm[i]]
            
            # Phase 2: Aggregate graph
            current_graph, communities = self._aggregate_graph(current_graph, communities)
            
            # If no more aggregation possible, stop
            if len(communities) == len(np.unique(communities)):
                break
        
        # Relabel communities to be contiguous
        unique_labels = np.unique(node_to_comm)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self.labels_ = np.array([label_mapping[label] for label in node_to_comm])
        
        self.n_communities_ = len(unique_labels)
        self.modularity_ = self._compute_modularity(X, self.labels_, m)
        
        return self

    def fit_predict(self, X, y=None):
        """Detect communities and return labels.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        labels : ndarray of shape (n_nodes,)
            Community label for each node.
        """
        self.fit(X, y)
        return self.labels_


class LabelPropagationCommunity(BaseEstimator, ClusterMixin):
    """Label propagation for community detection.
    
    This algorithm detects communities by propagating labels through the
    network. Each node is initialized with a unique label, and at each
    iteration, nodes adopt the label that is most frequent among their
    neighbors.
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Community label for each node.
        
    n_communities_ : int
        Number of communities detected.
        
    n_iter_ : int
        Number of iterations run.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import LabelPropagationCommunity
    >>> adj_matrix = sparse.csr_matrix(np.array([
    ...     [0, 1, 1, 0],
    ...     [1, 0, 1, 0],
    ...     [1, 1, 0, 1],
    ...     [0, 0, 1, 0]
    ... ]))
    >>> lp = LabelPropagationCommunity(max_iter=10)
    >>> labels = lp.fit_predict(adj_matrix)
    
    References
    ----------
    .. [1] Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time
           algorithm to detect community structures in large-scale networks.
           Physical review E, 76(3), 036106.
    """
    
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Detect communities using label propagation.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse=True)
        
        if X.shape[0] != X.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        # Convert to CSR for efficient access
        if not sparse.isspmatrix_csr(X):
            X = sparse.csr_matrix(X)
        
        n_nodes = X.shape[0]
        
        # Initialize each node with unique label
        labels = np.arange(n_nodes)
        
        # Iterate until convergence
        for iteration in range(self.max_iter):
            old_labels = labels.copy()
            
            # Update labels based on neighbor labels
            for node in range(n_nodes):
                neighbors = X[node].nonzero()[1]
                
                if len(neighbors) == 0:
                    continue
                
                # Get neighbor labels
                neighbor_labels = labels[neighbors]
                
                # Find most frequent label
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                max_count = counts.max()
                most_frequent = unique_labels[counts == max_count]
                
                # If tie, keep current label if it's in the tie
                if labels[node] in most_frequent:
                    continue
                else:
                    # Otherwise pick the smallest label (for determinism)
                    labels[node] = most_frequent.min()
            
            # Check for convergence
            if np.all(labels == old_labels):
                break
        
        # Relabel communities to be contiguous
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self.labels_ = np.array([label_mapping[label] for label in labels])
        
        self.n_communities_ = len(unique_labels)
        self.n_iter_ = iteration + 1
        
        return self

    def fit_predict(self, X, y=None):
        """Detect communities and return labels.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        labels : ndarray of shape (n_nodes,)
            Community label for each node.
        """
        self.fit(X, y)
        return self.labels_
