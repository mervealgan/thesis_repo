import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your prepped files
X = pd.read_csv("X_features.csv")
y = pd.read_csv("y_labels.csv").squeeze()  # convert to Series

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train your MLP model
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    random_state=42
)

mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 MSE: {mse:.4f}")
print(f"📈 R² score: {r2:.4f}")
