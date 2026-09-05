import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/moves.csv")
data.columns = data.columns.str.strip()

# Check if dataset supports 5 moves or fallback to available move columns
cols = [c for c in data.columns if c in ["Move1", "Move2", "Move3", "Move4", "Move5", "NextMove"]]
for col in cols:
    data[col] = data[col].astype(str).str.strip().str.lower()

move_map = {"rock": 0, "paper": 1, "scissors": 2}
for col in cols:
    data[col] = data[col].map(move_map)

data = data.dropna()

# Features and target
feature_cols = [c for c in cols if c != "NextMove"]
X = data[feature_cols]
y = data["NextMove"]

# Train stronger Random Forest with balanced class weights
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(X, y)

# Save model
joblib.dump(model, "models/rps_model.pkl")
print(f"Strong model trained successfully using {len(feature_cols)}-move history!")