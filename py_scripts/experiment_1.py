import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import joblib

#
X_readabilityfeatures = pd.read_csv("X_features_mlp_01.csv")
X_embeddings = pd.read_csv("embedding_diff_pool.csv")

#
X = pd.concat([X_embeddings, X_readabilityfeatures], axis=1)
y = pd.read_csv("y_labels.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# joblib.dump(pca, "pca_model_03.pkl") / # TODO : Is this necessary?

#
models = {
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam',
                        alpha=0.5, learning_rate_init=0.001, max_iter=300,
                        early_stopping=True, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

#
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error')
    print(f"{name} CV MAE: {-np.mean(cv_scores):.4f}")

    # Train on full training set # TODO
    model.fit(X_train, y_train)

    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl") # TODO : The name.

    # Predict on test set # TODO
    y_pred = model.predict(X_test)

    # Calculate metrics # TODO
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

# For the best performing model, analyze feature importance # TODO French and is it necessary or accurate?
best_model_name = min(results, key=lambda x: results[x]['MAE'])
print(f"Best model by MAE: {best_model_name}")