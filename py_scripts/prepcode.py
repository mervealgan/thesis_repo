import pandas as pd

# Load dataset
df = pd.read_csv("orimossimmos_with_features.csv")

# Step 1: Select all diff-based features
diff_columns = [col for col in df.columns if col.startswith("diff_")]
X = df[diff_columns]

# Optional: add score_original for more context
X["score_original"] = df["avg_rating_original"]

# Step 2: Select target
y = df["readability_gain"]

# Step 3: Optional — save as files for future use
X.to_csv("X_features.csv", index=False)
y.to_csv("y_labels.csv", index=False)

print(f"✅ Input matrix and target ready. Features: {X.shape[1]} | Examples: {X.shape[0]}")