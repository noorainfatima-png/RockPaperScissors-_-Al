import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/moves.csv")
data.columns = data.columns.str.strip()

for col in ["Move1", "Move2", "Move3", "NextMove"]:
    data[col] = data[col].astype(str).str.strip().str.lower()

move_map = {"rock": 0, "paper": 1, "scissors": 2}
data["Move1"] = data["Move1"].map(move_map)
data["Move2"] = data["Move2"].map(move_map)
data["Move3"] = data["Move3"].map(move_map)
data["NextMove"] = data["NextMove"].map(move_map)

data = data.dropna()

X = data[["Move1", "Move2", "Move3"]]
y = data["NextMove"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Random Forest Accuracy: {accuracy:.2f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")