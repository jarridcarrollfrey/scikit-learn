"""Graph kernels for graph classification and comparison."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted


class WeisfeilerLehmanKernel(BaseEstimator, TransformerMixin):
    """Weisfeiler-Lehman graph kernel.
    
    The Weisfeiler-Lehman kernel is a powerful graph kernel based on the
    Weisfeiler-Lehman test of isomorphism. It iteratively augments node
    labels using the labels of neighboring nodes, creating a hierarchy
    of node labels that captures graph structure at multiple scales.
    
    Parameters
    ----------
    n_iter : int, default=3
        Number of Weisfeiler-Lehman iterations. Higher values capture
        more global graph structure.
        
    normalize : bool, default=True
        Whether to normalize the kernel matrix.
        
    Attributes
    ----------
    n_features_out_ : int
        Number of output features after transformation.
        
    label_sequences_ : list of dict
        Label sequences for each graph at each iteration.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import WeisfeilerLehmanKernel
    >>> # Create two simple graphs as adjacency matrices
    >>> graph1 = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
    >>> graph2 = sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    >>> kernel = WeisfeilerLehmanKernel(n_iter=2)
    >>> # For a list of graphs, we need to pass them differently
    >>> # This is a simplified example
    
    References
    ----------
    .. [1] Shervashidze, N., Schweitzer, P., Leeuwen, E. J. V., Mehlhorn, K.,
           & Borgwardt, K. M. (2011). Weisfeiler-lehman graph kernels.
           Journal of Machine Learning Research, 12(Sep), 2539-2561.
    """
    
    _parameter_constraints: dict = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "normalize": ["boolean"],
    }

    def __init__(self, n_iter=3, normalize=True):
        self.n_iter = n_iter
        self.normalize = normalize

    def _weisfeiler_lehman_step(self, adj_matrix, labels):
        """Perform one iteration of Weisfeiler-Lehman relabeling."""
        n_nodes = adj_matrix.shape[0]
        new_labels = np.zeros(n_nodes, dtype=object)
        
        # Convert to CSR for efficient row access
        if not sparse.isspmatrix_csr(adj_matrix):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        for node in range(n_nodes):
            # Get neighbors
            neighbors = adj_matrix[node].nonzero()[1]
            
            # Sort neighbor labels
            neighbor_labels = sorted([labels[n] for n in neighbors])
            
            # Create new label by concatenating current label with neighbor labels
            new_labels[node] = str(labels[node]) + "_" + "_".join(map(str, neighbor_labels))
        
        return new_labels

    def _compute_feature_vector(self, adj_matrix, initial_labels=None):
        """Compute Weisfeiler-Lehman feature vector for a graph."""
        n_nodes = adj_matrix.shape[0]
        
        # Initialize labels (all nodes have label 1 if not provided)
        if initial_labels is None:
            labels = np.ones(n_nodes, dtype=int)
        else:
            labels = initial_labels.copy()
        
        # Collect all labels across iterations
        all_labels = []
        
        # Add initial labels
        label_counts = {}
        for label in labels:
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        all_labels.append(label_counts)
        
        # Perform Weisfeiler-Lehman iterations
        for _ in range(self.n_iter):
            labels = self._weisfeiler_lehman_step(adj_matrix, labels)
            
            # Count labels
            label_counts = {}
            for label in labels:
                label_counts[str(label)] = label_counts.get(str(label), 0) + 1
            all_labels.append(label_counts)
        
        return all_labels

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Weisfeiler-Lehman kernel.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, list):
            X = [X]
        
        # Compute feature vectors for all graphs
        self.label_sequences_ = []
        all_labels_set = set()
        
        for graph in X:
            graph = check_array(graph, accept_sparse=True)
            if graph.shape[0] != graph.shape[1]:
                raise ValueError("Each adjacency matrix must be square")
            
            label_sequence = self._compute_feature_vector(graph)
            self.label_sequences_.append(label_sequence)
            
            # Collect all unique labels
            for label_counts in label_sequence:
                all_labels_set.update(label_counts.keys())
        
        # Create label to index mapping
        self.label_to_idx_ = {label: idx for idx, label in enumerate(sorted(all_labels_set))}
        self.n_features_out_ = len(self.label_to_idx_)
        
        return self

    def transform(self, X):
        """Transform graphs to feature vectors.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        Returns
        -------
        features : ndarray of shape (n_graphs, n_features_out_)
            Feature matrix where each row is the feature vector for a graph.
        """
        check_is_fitted(self, ["label_to_idx_", "n_features_out_"])
        
        if not isinstance(X, list):
            X = [X]
        
        features = np.zeros((len(X), self.n_features_out_))
        
        for i, graph in enumerate(X):
            graph = check_array(graph, accept_sparse=True)
            label_sequence = self._compute_feature_vector(graph)
            
            # Convert label sequence to feature vector
            for label_counts in label_sequence:
                for label, count in label_counts.items():
                    if label in self.label_to_idx_:
                        features[i, self.label_to_idx_[label]] += count
        
        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features = features / norms
        
        return features

    def fit_transform(self, X, y=None):
        """Fit and transform graphs to feature vectors.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        features : ndarray of shape (n_graphs, n_features_out_)
            Feature matrix where each row is the feature vector for a graph.
        """
        return self.fit(X, y).transform(X)


class RandomWalkKernel(BaseEstimator, TransformerMixin):
    """Random walk graph kernel.
    
    The random walk kernel compares graphs by counting matching walks.
    It captures graph structure by comparing the distribution of random
    walks in different graphs.
    
    Parameters
    ----------
    walk_length : int, default=4
        Maximum length of random walks to consider.
        
    lambda_decay : float, default=0.01
        Decay factor for walk length. Controls the importance of longer walks.
        
    normalize : bool, default=True
        Whether to normalize the kernel matrix.
        
    Attributes
    ----------
    n_features_out_ : int
        Number of output features after transformation.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import RandomWalkKernel
    >>> # Create two simple graphs
    >>> graph1 = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
    >>> graph2 = sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    >>> kernel = RandomWalkKernel(walk_length=3)
    
    References
    ----------
    .. [1] Vishwanathan, S. V. N., Schraudolph, N. N., Kondor, R., &
           Borgwardt, K. M. (2010). Graph kernels. Journal of Machine Learning
           Research, 11(Apr), 1201-1242.
    """
    
    _parameter_constraints: dict = {
        "walk_length": [Interval(Integral, 1, None, closed="left")],
        "lambda_decay": [Interval(Real, 0, 1, closed="both")],
        "normalize": ["boolean"],
    }

    def __init__(self, walk_length=4, lambda_decay=0.01, normalize=True):
        self.walk_length = walk_length
        self.lambda_decay = lambda_decay
        self.normalize = normalize

    def _compute_walk_counts(self, adj_matrix):
        """Compute walk counts for different lengths."""
        n_nodes = adj_matrix.shape[0]
        
        # Ensure sparse CSR format
        if not sparse.isspmatrix_csr(adj_matrix):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        # Normalize adjacency matrix by degree
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv = sparse.diags(1.0 / degrees)
        P = D_inv @ adj_matrix  # Transition matrix
        
        # Compute powers of transition matrix
        walk_counts = []
        P_power = sparse.eye(n_nodes, format='csr')
        
        for length in range(self.walk_length + 1):
            # Count walks by summing diagonal elements
            count = P_power.diagonal().sum() * (self.lambda_decay ** length)
            walk_counts.append(count)
            
            if length < self.walk_length:
                P_power = P_power @ P
        
        return np.array(walk_counts)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the random walk kernel.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, list):
            X = [X]
        
        # Compute walk counts for all graphs
        self.walk_counts_ = []
        
        for graph in X:
            graph = check_array(graph, accept_sparse=True)
            if graph.shape[0] != graph.shape[1]:
                raise ValueError("Each adjacency matrix must be square")
            
            walk_counts = self._compute_walk_counts(graph)
            self.walk_counts_.append(walk_counts)
        
        self.n_features_out_ = self.walk_length + 1
        
        return self

    def transform(self, X):
        """Transform graphs to feature vectors based on random walks.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        Returns
        -------
        features : ndarray of shape (n_graphs, n_features_out_)
            Feature matrix where each row contains walk counts for a graph.
        """
        check_is_fitted(self, ["walk_counts_", "n_features_out_"])
        
        if not isinstance(X, list):
            X = [X]
        
        features = np.zeros((len(X), self.n_features_out_))
        
        for i, graph in enumerate(X):
            graph = check_array(graph, accept_sparse=True)
            walk_counts = self._compute_walk_counts(graph)
            features[i] = walk_counts
        
        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features = features / norms
        
        return features

    def fit_transform(self, X, y=None):
        """Fit and transform graphs to feature vectors.
        
        Parameters
        ----------
        X : list of sparse matrices
            List of adjacency matrices representing graphs.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        features : ndarray of shape (n_graphs, n_features_out_)
            Feature matrix where each row contains walk counts for a graph.
        """
        return self.fit(X, y).transform(X)
