"""
Getting preliminary results for Midpoint Check-In
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pages.B_Train_Model import LinearRegression
from pages.C_Test_Model import rmse, mae, r2

df = pd.read_csv('mortgage_3000_onehot.csv').dropna()

target = 'propensity_score'
features = [col for col in df.columns if col != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = LinearRegression(learning_rate=0.01, num_iterations=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test).reshape(-1)

print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")
print(f"Mean Absolute Error (MAE): {mae(y_test, y_pred):.4f}")
print(f"R² Score: {r2(y_test, y_pred):.4f}")
