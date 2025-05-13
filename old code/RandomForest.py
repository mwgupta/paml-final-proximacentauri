import numpy as np
import streamlit as st
import pandas as pd
from sklearn.utils import resample

# 3.4 Random Forest Class
class RandomForest(object):
    def __init__(self, num_trees=10, max_depth=1, max_features='sqrt'):
        self.model_name = 'Random Forest'
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    class DecisionStump:
        def __init__(self):
            self.feature_index = None
            self.threshold = None
            self.polarity = 1
        
        def fit(self, X, y):
            n_samples, n_features = X.shape
            min_error = float('inf')
            
            for feature_i in range(n_features):
                thresholds = np.unique(X[:, feature_i])
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        predictions[polarity * X[:, feature_i] < polarity * threshold] = -1
                        error = np.sum(predictions != y)
                        
                        if error < min_error:
                            min_error = error
                            self.polarity = polarity
                            self.threshold = threshold
                            self.feature_index = feature_i
        
        def predict(self, X):
            n_samples = X.shape[0]
            predictions = np.ones(n_samples)
            feature_values = X[:, self.feature_index]
            predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
            return predictions
    
    def fit(self, X, y):
        self.trees = []
        
        # Ensure numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy().flatten()
        
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            n_selected_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            n_selected_features = min(self.max_features, n_features)
        else:
            n_selected_features = n_features
        
        for i in range(self.num_trees):
            # Bootstrap sample
            X_sample, y_sample = resample(X, y, random_state=i)
            
            # Random feature subset
            feature_indices = np.random.choice(n_features, n_selected_features, replace=False)
            
            # Extract subset using a loop to avoid fancy indexing issues
            X_sample_subset = np.zeros((X_sample.shape[0], len(feature_indices)))
            for j, feat_idx in enumerate(feature_indices):
                X_sample_subset[:, j] = X_sample[:, feat_idx]
            
            # Fit decision stump
            stump = self.DecisionStump()
            stump.fit(X_sample_subset, y_sample)
            
            self.trees.append((stump, feature_indices))
        
        return self
    
    def predict(self, X):
        # Ensure NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        stump_preds = []
        for stump, feat_ids in self.trees:
            # Extract subset using a loop to avoid fancy indexing issues
            X_subset = np.zeros((X.shape[0], len(feat_ids)))
            for j, feat_idx in enumerate(feat_ids):
                X_subset[:, j] = X[:, feat_idx]
                
            pred = stump.predict(X_subset)
            stump_preds.append(pred)
        
        # Majority vote
        stump_preds = np.array(stump_preds)
        y_pred = np.sign(np.sum(stump_preds, axis=0))
        y_pred[y_pred == 0] = 1  # break ties in favor of +1
        return y_pred