"""Node embedding algorithms for graph representation learning."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class Node2Vec(BaseEstimator, TransformerMixin):
    """Node2Vec graph embedding algorithm.
    
    Node2Vec learns continuous feature representations for nodes in a graph by
    optimizing a neighborhood preserving objective. The algorithm uses biased
    random walks to generate node sequences and then applies Skip-gram model
    to learn embeddings.
    
    Parameters
    ----------
    n_components : int, default=128
        Dimensionality of the embedding space.
        
    walk_length : int, default=80
        Length of random walk starting from each node.
        
    num_walks : int, default=10
        Number of random walks to generate from each node.
        
    p : float, default=1.0
        Return parameter. Controls the likelihood of immediately revisiting
        a node in the walk. Higher values make the walk less likely to sample
        an already visited node.
        
    q : float, default=1.0
        In-out parameter. Controls the likelihood of visiting nodes that are
        farther from the starting node. If q > 1, the random walk is biased
        towards nodes close to the starting node. If q < 1, the walk is biased
        towards nodes farther away.
        
    window_size : int, default=10
        Context window size for skip-gram model.
        
    n_iter : int, default=1
        Number of iterations (epochs) over the corpus.
        
    workers : int, default=1
        Number of parallel workers for generating random walks.
        
    negative_samples : int, default=5
        Number of negative samples for skip-gram model.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the random walks and embedding initialization.
        Pass an int for reproducible results across multiple function calls.
        
    Attributes
    ----------
    embeddings_ : ndarray of shape (n_nodes, n_components)
        The learned embeddings for each node in the graph.
        
    n_features_in_ : int
        Number of features (nodes) seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import Node2Vec
    >>> # Create a simple graph adjacency matrix
    >>> adj_matrix = sparse.csr_matrix(np.array([
    ...     [0, 1, 1, 0],
    ...     [1, 0, 1, 1],
    ...     [1, 1, 0, 1],
    ...     [0, 1, 1, 0]
    ... ]))
    >>> model = Node2Vec(n_components=2, random_state=42)
    >>> embeddings = model.fit_transform(adj_matrix)
    >>> embeddings.shape
    (4, 2)
    
    References
    ----------
    .. [1] Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature
           learning for networks. In Proceedings of the 22nd ACM SIGKDD
           international conference on Knowledge discovery and data mining
           (pp. 855-864).
    """
    
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "walk_length": [Interval(Integral, 1, None, closed="left")],
        "num_walks": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0, None, closed="neither")],
        "q": [Interval(Real, 0, None, closed="neither")],
        "window_size": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "workers": [Interval(Integral, 1, None, closed="left")],
        "negative_samples": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=128,
        walk_length=80,
        num_walks=10,
        p=1.0,
        q=1.0,
        window_size=10,
        n_iter=1,
        workers=1,
        negative_samples=5,
        random_state=None,
    ):
        self.n_components = n_components
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.n_iter = n_iter
        self.workers = workers
        self.negative_samples = negative_samples
        self.random_state = random_state

    def _generate_walks(self, adj_matrix, random_state):
        """Generate random walks starting from each node."""
        n_nodes = adj_matrix.shape[0]
        walks = []
        
        # Convert to CSR format for efficient row access
        if not sparse.isspmatrix_csr(adj_matrix):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        for _ in range(self.num_walks):
            random_state.shuffle(np.arange(n_nodes))
            for node in range(n_nodes):
                walks.append(self._node2vec_walk(adj_matrix, node, random_state))
        
        return walks

    def _node2vec_walk(self, adj_matrix, start_node, random_state):
        """Simulate a biased random walk starting from start_node."""
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_neighbors = adj_matrix[cur].nonzero()[1]
            
            if len(cur_neighbors) == 0:
                break
            
            if len(walk) == 1:
                # First step: uniform random choice
                walk.append(random_state.choice(cur_neighbors))
            else:
                prev = walk[-2]
                next_node = self._biased_walk_step(
                    adj_matrix, prev, cur, cur_neighbors, random_state
                )
                walk.append(next_node)
        
        return walk

    def _biased_walk_step(self, adj_matrix, prev, cur, neighbors, random_state):
        """Choose next node with bias based on p and q parameters."""
        # Calculate unnormalized probabilities
        probs = []
        for neighbor in neighbors:
            if neighbor == prev:
                # Return to previous node
                probs.append(1.0 / self.p)
            elif adj_matrix[prev, neighbor] > 0:
                # Neighbor is also neighbor of previous (DFS)
                probs.append(1.0)
            else:
                # Neighbor is farther away (BFS)
                probs.append(1.0 / self.q)
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Sample next node
        return random_state.choice(neighbors, p=probs)

    def _train_embeddings(self, walks, random_state):
        """Train embeddings using skip-gram with negative sampling."""
        n_nodes = max(max(walk) for walk in walks) + 1
        
        # Initialize embeddings
        embeddings = random_state.randn(n_nodes, self.n_components) * 0.01
        context_embeddings = random_state.randn(n_nodes, self.n_components) * 0.01
        
        # Learning rate
        alpha = 0.025
        min_alpha = 0.0001
        
        for epoch in range(self.n_iter):
            # Anneal learning rate
            current_alpha = max(min_alpha, alpha * (1.0 - epoch / self.n_iter))
            
            for walk in walks:
                for i, center_node in enumerate(walk):
                    # Context window
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i == j:
                            continue
                        
                        context_node = walk[j]
                        
                        # Positive sample
                        center_vec = embeddings[center_node]
                        context_vec = context_embeddings[context_node]
                        score = np.dot(center_vec, context_vec)
                        pred = 1.0 / (1.0 + np.exp(-score))
                        grad = (1.0 - pred) * current_alpha
                        
                        embeddings[center_node] += grad * context_vec
                        context_embeddings[context_node] += grad * center_vec
                        
                        # Negative samples
                        for _ in range(self.negative_samples):
                            neg_node = random_state.randint(0, n_nodes)
                            if neg_node == context_node:
                                continue
                            
                            neg_vec = context_embeddings[neg_node]
                            score = np.dot(center_vec, neg_vec)
                            pred = 1.0 / (1.0 + np.exp(-score))
                            grad = -pred * current_alpha
                            
                            embeddings[center_node] += grad * neg_vec
                            context_embeddings[neg_node] += grad * center_vec
        
        return embeddings

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Node2Vec model.
        
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
        # Validate input
        X = check_array(X, accept_sparse=True)
        
        if X.shape[0] != X.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        
        self.n_features_in_ = X.shape[0]
        
        # Set random state
        random_state = check_random_state(self.random_state)
        
        # Generate walks
        walks = self._generate_walks(X, random_state)
        
        # Train embeddings
        self.embeddings_ = self._train_embeddings(walks, random_state)
        
        return self

    def transform(self, X=None):
        """Return the learned embeddings.
        
        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        embeddings : ndarray of shape (n_nodes, n_components)
            The learned node embeddings.
        """
        check_is_fitted(self, ["embeddings_"])
        return self.embeddings_

    def fit_transform(self, X, y=None):
        """Fit the model and return embeddings.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        embeddings : ndarray of shape (n_nodes, n_components)
            The learned node embeddings.
        """
        return self.fit(X, y).transform()


class DeepWalk(BaseEstimator, TransformerMixin):
    """DeepWalk graph embedding algorithm.
    
    DeepWalk learns continuous feature representations for nodes by treating
    random walks as sentences and applying language modeling techniques
    (Skip-gram). It is a special case of Node2Vec where p=1 and q=1.
    
    Parameters
    ----------
    n_components : int, default=128
        Dimensionality of the embedding space.
        
    walk_length : int, default=80
        Length of random walk starting from each node.
        
    num_walks : int, default=10
        Number of random walks to generate from each node.
        
    window_size : int, default=10
        Context window size for skip-gram model.
        
    n_iter : int, default=1
        Number of iterations (epochs) over the corpus.
        
    workers : int, default=1
        Number of parallel workers for generating random walks.
        
    negative_samples : int, default=5
        Number of negative samples for skip-gram model.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the random walks and embedding initialization.
        Pass an int for reproducible results across multiple function calls.
        
    Attributes
    ----------
    embeddings_ : ndarray of shape (n_nodes, n_components)
        The learned embeddings for each node in the graph.
        
    n_features_in_ : int
        Number of features (nodes) seen during fit.
        
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sklearn.graph import DeepWalk
    >>> # Create a simple graph adjacency matrix
    >>> adj_matrix = sparse.csr_matrix(np.array([
    ...     [0, 1, 1, 0],
    ...     [1, 0, 1, 1],
    ...     [1, 1, 0, 1],
    ...     [0, 1, 1, 0]
    ... ]))
    >>> model = DeepWalk(n_components=2, random_state=42)
    >>> embeddings = model.fit_transform(adj_matrix)
    >>> embeddings.shape
    (4, 2)
    
    References
    ----------
    .. [1] Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). Deepwalk: Online
           learning of social representations. In Proceedings of the 20th ACM
           SIGKDD international conference on Knowledge discovery and data
           mining (pp. 701-710).
    """
    
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "walk_length": [Interval(Integral, 1, None, closed="left")],
        "num_walks": [Interval(Integral, 1, None, closed="left")],
        "window_size": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "workers": [Interval(Integral, 1, None, closed="left")],
        "negative_samples": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=128,
        walk_length=80,
        num_walks=10,
        window_size=10,
        n_iter=1,
        workers=1,
        negative_samples=5,
        random_state=None,
    ):
        self.n_components = n_components
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.n_iter = n_iter
        self.workers = workers
        self.negative_samples = negative_samples
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the DeepWalk model.
        
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
        # DeepWalk is Node2Vec with p=1, q=1
        self._node2vec = Node2Vec(
            n_components=self.n_components,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=1.0,
            q=1.0,
            window_size=self.window_size,
            n_iter=self.n_iter,
            workers=self.workers,
            negative_samples=self.negative_samples,
            random_state=self.random_state,
        )
        
        self._node2vec.fit(X, y)
        self.embeddings_ = self._node2vec.embeddings_
        self.n_features_in_ = X.shape[0]
        
        return self

    def transform(self, X=None):
        """Return the learned embeddings.
        
        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        embeddings : ndarray of shape (n_nodes, n_components)
            The learned node embeddings.
        """
        check_is_fitted(self, ["embeddings_"])
        return self.embeddings_

    def fit_transform(self, X, y=None):
        """Fit the model and return embeddings.
        
        Parameters
        ----------
        X : sparse matrix of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        embeddings : ndarray of shape (n_nodes, n_components)
            The learned node embeddings.
        """
        return self.fit(X, y).transform()
