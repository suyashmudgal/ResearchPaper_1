import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Union

class SASKC(BaseEstimator, ClassifierMixin):
    """
    Supervised Adaptive Statistical Kernel Classifier (SASKC).
    
    A novel supervised classifier that combines feature variance weighting,
    weighted Euclidean distance, and rank-based voting.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification.
        
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
    """
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SASKC':
        """
        Fit the SASKC model according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values (class labels).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        
        # Step 1: Compute feature-wise statistics
        # variance per feature: σ_f² = Var(X[:, f])
        # define feature weight: w_f = 1 / (1 + σ_f²)
        # We use ddof=0 for population variance or ddof=1 for sample variance. 
        # Spec doesn't specify, but usually sample variance (ddof=1) is safer. 
        # However, numpy var default is ddof=0. Let's use ddof=0 (population) as it's standard in ML unless specified.
        # Actually, let's use ddof=0 to match standard "Var(X)" notation.
        variances = np.var(X, axis=0)
        self.feature_weights_ = 1.0 / (1.0 + variances)
        
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
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        # Step 2: Weighted distance
        # Dᵢⱼ = sqrt( Σ_f [ w_f * (xᵢ,f - xⱼ,f)² ] )
        # We can compute this efficiently using broadcasting or cdist with custom metric.
        # But for clarity and to match spec exactly, let's do it explicitly or semi-vectorized.
        # Since dataset is small (Wine ~178 samples), we can do it per test sample or fully vectorized.
        # Let's do fully vectorized for efficiency if possible, but memory might be an issue if X is huge.
        # Wine is tiny, so fully vectorized is fine.
        
        # Expand dims for broadcasting:
        # X: (n_test, 1, n_features)
        # self.X_: (1, n_train, n_features)
        # weights: (1, 1, n_features)
        
        # However, for very large X_test, this might OOM. Let's loop over X_test samples for safety and clarity.
        
        for i in range(n_samples):
            # Difference: (n_train, n_features)
            diff = X[i] - self.X_
            
            # Weighted squared difference
            weighted_sq_diff = self.feature_weights_ * (diff ** 2)
            
            # Sum over features
            sum_sq_diff = np.sum(weighted_sq_diff, axis=1)
            
            # Sqrt -> Distances Dᵢⱼ (n_train,)
            distances = np.sqrt(sum_sq_diff)
            
            # Step 3: Nearest neighbors
            # Sort distances ascending
            # We only need top k
            k = min(self.n_neighbors, len(self.X_))
            
            # Get indices of k nearest neighbors
            # argpartition is faster than sort for top k
            if k < len(self.X_):
                nearest_indices = np.argpartition(distances, k)[:k]
            else:
                nearest_indices = np.arange(len(self.X_))
                
            # We need to sort these k indices by distance to get correct ranks
            nearest_dists = distances[nearest_indices]
            sorted_k_indices = np.argsort(nearest_dists)
            
            final_indices = nearest_indices[sorted_k_indices]
            final_dists = nearest_dists[sorted_k_indices]
            final_labels = self.y_[final_indices]
            
            # Step 4: Similarity + rank weights
            # similarity: Sⱼ = 1 / (1 + Dᵢⱼ)
            similarities = 1.0 / (1.0 + final_dists)
            
            # rank weight: Rⱼ = 1 / rankⱼ (1, 1/2, 1/3, ...)
            # ranks are 1-based: 1, 2, ..., k
            ranks = np.arange(1, k + 1)
            rank_weights = 1.0 / ranks
            
            # Step 5: Class support score
            # C_c = Σ_j ( Sⱼ * Rⱼ * 1[yⱼ == c] )
            
            # Calculate contribution for each neighbor: Sⱼ * Rⱼ
            contributions = similarities * rank_weights
            
            for idx, label in enumerate(final_labels):
                # Find index of label in self.classes_
                class_idx = np.searchsorted(self.classes_, label)
                # Note: searchsorted works if classes_ is sorted. unique_labels returns sorted.
                probabilities[i, class_idx] += contributions[idx]
                
        # Step 6: Prediction + probabilities
        # class probabilities = C_c normalized across classes
        
        # Handle case where sum is 0 (shouldn't happen unless k=0 or something weird)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        
        probabilities /= row_sums
        
        return probabilities
