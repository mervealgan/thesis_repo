import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the readability features + score_original
X_readability = pd.read_csv("X_features.csv")

# Step 2: Load the CamemBERT embedding differences
X_embeddings = pd.read_csv("embedding_diff.csv")

# Step 3: Combine both into one full input matrix
X_full = pd.concat([X_embeddings, X_readability], axis=1)

# Step 4: Load your target variable
y = pd.read_csv("y_labels.csv").squeeze()

# Step 5: Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Step 6: Define and train your MLP model
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

# Step 7: Evaluate
y_pred = mlp.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("✅ Full model trained successfully!")
print(f"📉 MSE: {mse:.4f}")
print(f"📈 R² score: {r2:.4f}")

import joblib
joblib.dump(mlp, "final_model_mlp.pkl")
print("✅ Model saved.")


X_full.to_csv("X_full_with_embeddings.csv", index=False)
print("✅ Final input features saved.")

import pandas as pd

results = pd.DataFrame({
    "true_gain": y_val,
    "predicted_gain": y_pred
})

results.to_csv("predictions_val.csv", index=False)
print("✅ Predictions saved for validation set.")