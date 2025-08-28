import pandas as pd
import yaml
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import json

# ======================
# Load Config
# ======================
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# ======================
# Load Model
# ======================
model = joblib.load("models/model.pkl")

# ======================
# Load Test Data
# ======================
df = pd.read_csv("data/processed/test.csv")

X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]

# ======================
# Predictions & Metrics
# ======================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()  # convert numpy array to list for JSON

print("✅ Evaluation Results:")
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# ======================
# Save Metrics to JSON
# ======================
os.makedirs("experiments", exist_ok=True)
results_file = "experiments/results.json"

# Create results dictionary
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": config["model"]["type"],
    "params": config["model"]["params"],
    "accuracy": acc,
    "f1_score": f1,
    "confusion_matrix": cm
}

# Append results if file exists, otherwise start fresh
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
else:
    data = []

data.append(results)

with open(results_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"Results logged in {results_file}")
