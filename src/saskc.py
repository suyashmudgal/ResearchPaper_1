import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from typing import Optional, Union, Tuple

class SASKC(BaseEstimator, ClassifierMixin):
    """
    Supervised Adaptive Statistical Kernel Classifier (SASKC).

    A non-parametric classifier that constructs a Riemannian manifold based on 
    global feature variance. It employs a statistical kernel to dynamically 
    weight features and a rank-based voting mechanism for robust prediction.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification.
    
    use_weights : bool, default=True
        Whether to apply variance-based feature weighting.
    
    use_rank_voting : bool, default=True
        Whether to apply rank-decay weighting to votes.
        
    epsilon : float, default=1e-6
        Smoothing parameter for weight calculation to prevent division by zero.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    
    y_ : ndarray of shape (n_samples,)
        The training target values.
    
    feature_weights_ : ndarray of shape (n_features,)
        Weights assigned to each feature based on variance.
        w_f = 1 / (1 + sigma_f^2 + epsilon)
    """

    def __init__(self, 
                 n_neighbors: int = 5, 
                 use_weights: bool = True, 
                 use_rank_voting: bool = True,
                 epsilon: float = 1e-6):
        self.n_neighbors = n_neighbors
        self.use_weights = use_weights
        self.use_rank_voting = use_rank_voting
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SASKC':
        """
        Fit the SASKC model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector.
        
        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store unique classes
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        
        # Compute Feature Weights (Stage 3 of Pipeline)
        if self.use_weights:
            # Global variance of each feature
            variances = np.var(X, axis=0)
            # Adaptive Weighting: w_f = 1 / (1 + sigma^2 + epsilon)
            # This penalizes high-variance (noisy) features.
            self.feature_weights_ = 1.0 / (1.0 + variances + self.epsilon)
        else:
            self.feature_weights_ = np.ones(X.shape[1])
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_query = X.shape[0]
        n_train = self.X_.shape[0]
        n_classes = len(self.classes_)
        
        # 1. Weighted Distance Calculation
        # We use scipy's cdist with 'seuclidean'. 
        # 'seuclidean' computes sqrt(sum((x-y)^2 / V)).
        # We want sqrt(sum(w * (x-y)^2)).
        # So we set V = 1/w.
        
        # Handle case where weights are effectively zero (avoid div by zero)
        # Weights are bounded (0, 1], so 1/w is safe.
        V = 1.0 / (self.feature_weights_ + 1e-10)
        
        # D matrix: (n_query, n_train)
        # This is O(N_query * N_train * F) but highly optimized in C
        distances = cdist(X, self.X_, metric='seuclidean', V=V)
        
        # 2. Statistical Kernel Transformation
        # K(x, y) = 1 / (1 + d_W(x, y))
        kernel_matrix = 1.0 / (1.0 + distances)
        
        # 3. Neighbor Retrieval (Top-k)
        k = min(self.n_neighbors, n_train)
        
        # argpartition is O(N) average case, faster than full sort O(N log N)
        # Indices of k nearest neighbors for each query point
        if k < n_train:
            # (n_query, k) indices
            top_k_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        else:
            top_k_indices = np.tile(np.arange(n_train), (n_query, 1))
            
        # We need to sort these top k to apply rank weights correctly
        # Gather distances for top k
        # Advanced indexing: row indices broadcast against col indices
        row_indices = np.arange(n_query)[:, None]
        top_k_dists = distances[row_indices, top_k_indices]
        
        # Sort within the top k
        sorted_k_order = np.argsort(top_k_dists, axis=1)
        
        # Apply sorting to indices and kernel values
        final_indices = top_k_indices[row_indices, sorted_k_order]
        final_kernel_vals = kernel_matrix[row_indices, final_indices]
        final_labels = self.y_[final_indices]
        
        # 4. Rank-Based Voting
        # Vote = Kernel_Similarity * Rank_Decay
        if self.use_rank_voting:
            ranks = np.arange(1, k + 1) # [1, 2, ..., k]
            rank_weights = 1.0 / ranks  # [1, 0.5, 0.33, ...]
            # Broadcast rank weights across all queries
            votes = final_kernel_vals * rank_weights[None, :]
        else:
            votes = final_kernel_vals
            
        # 5. Aggregate Votes
        probabilities = np.zeros((n_query, n_classes))
        
        # Vectorized aggregation is tricky with different labels per row.
        # We iterate over classes or use scatter_add equivalent.
        # Since n_classes is small, iterating classes is efficient.
        
        for cls_idx, cls_label in enumerate(self.classes_):
            # Mask where neighbor label matches current class
            # (n_query, k) boolean mask
            mask = (final_labels == cls_label)
            
            # Sum votes where mask is True
            probabilities[:, cls_idx] = np.sum(votes * mask, axis=1)
            
        # Normalize to probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # Safety
        probabilities /= row_sums
        
        return probabilities
