"""
Getting preliminary results for Midpoint Check-In
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def rmse(y_true, y_pred):
    num_examples = y_true.shape[0]
    return np.sqrt(np.sum(np.power(y_true - y_pred, 2)) / num_examples)

def mae(y_true, y_pred):
    num_examples = y_true.shape[0]
    return np.sum(np.abs(y_true - y_pred)) / num_examples

def r2(y_true, y_pred):
    tss = np.sum(np.power(y_true - np.mean(y_true), 2))
    rss = np.sum(np.power(y_true - y_pred, 2))
    return 1 - (rss / tss)

METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

def compute_eval_metrics(X, y_true, model, metrics):
    metric_dict = {}
    y_pred = model.predict(X).reshape(-1)
    for metric in metrics:
        metric_dict[metric] = METRICS_MAP[metric](y_true, y_pred)
    return metric_dict

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_history = []

    def normalize(self, X):
        means = np.mean(X[:, 1:], axis=0)
        stds = np.std(X[:, 1:], axis=0) + 1e-7
        X[:, 1:] = (X[:, 1:] - means) / stds
        return X

    def predict(self, X):
        num_examples = X.shape[0]
        X_bias = np.append(np.ones((num_examples, 1)), X, axis=1)
        X_norm = self.normalize(X_bias.copy())
        return X_norm.dot(self.W).reshape(-1, 1)

    def update_weights(self):
        num_examples = self.X.shape[0]
        X_bias = np.append(np.ones((num_examples, 1)), self.X, axis=1)
        X_norm = self.normalize(X_bias.copy())
        Y_pred = self.predict(self.X)

        dW = - (2 * (X_norm.T).dot(self.Y - Y_pred)) / num_examples
        cost = np.sqrt(np.sum(np.power(self.Y - Y_pred, 2)) / num_examples)
        self.cost_history.append(cost)

        self.W -= self.learning_rate * dW
        return cost

    def fit(self, X, Y):
        self.X = X
        self.Y = Y.reshape(-1, 1)
        self.W = np.zeros((X.shape[1] + 1, 1))

        for i in range(self.num_iterations):
            cost = self.update_weights()
            if i % 10 == 0 or i == self.num_iterations - 1:
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {cost:.4f}")
        return self

df = pd.read_csv('mortgage_3000_onehot.csv').dropna()

target = 'propensity_score'
features = [col for col in df.columns if col != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Data Loaded Successfully!")

model = LinearRegression(learning_rate=0.01, num_iterations=100)
model.fit(X_train, y_train)

print("\nModel Training Completed!\n")

metrics = ['root_mean_squared_error', 'mean_absolute_error', 'r2_score']
train_metrics = compute_eval_metrics(X_train, y_train, model, metrics)
test_metrics = compute_eval_metrics(X_test, y_test, model, metrics)

print("Train Set Metrics:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
