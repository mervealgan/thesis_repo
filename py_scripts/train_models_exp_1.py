import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import joblib
from sklearn.decomposition import PCA

# Data
X_read = pd.read_csv("X_read.csv")
X_embeddings = pd.read_csv("embedding_diff_mean_pool.csv")
y = pd.read_csv("y_target_gain.csv").squeeze()

# Combining all input
X = pd.concat([X_embeddings, X_read], axis=1)

# Splitting for test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    'MLP': MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='tanh', solver='adam',
        alpha=0.5, learning_rate_init=0.001, max_iter=300,
        early_stopping=True, random_state=42
    ),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGB': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = {}

# Training - Evaluation
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error')
    print(f"{name} CV MAE: {-np.mean(cv_scores):.4f}")

    # Train on full training set
    model.fit(X_train, y_train)

    # Saving model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_exp1_model.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred
    })
    predictions_df.to_csv(f"{name.replace(' ', '_').lower()}_exp1_predictions.csv", index=False)


    # Evaluate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson = pearsonr(y_test, y_pred)[0]
    spearman = spearmanr(y_test, y_pred)[0]

    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'Pearson': pearson,
        'Spearman': spearman
    }

    print(f"{name} Test Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pearson correlation: {pearson:.4f}")
    print(f"  Spearman correlation: {spearman:.4f}")

# Save results
pd.DataFrame(results).T.to_csv("exp1_results.csv")