import pandas as pd
import yaml
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load Config
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Training Data
df = pd.read_csv("data/processed/train.csv")

# Last column is assumed to be target ("loan_status")
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

# Model Selection
if config["model"]["type"] == "RandomForestClassifier":
    model = RandomForestClassifier(**config["model"]["params"])
elif config["model"]["type"] == "LogisticRegression":
    model = LogisticRegression(**config["model"]["params"])
elif config["model"]["type"] == "DecisionTreeClassifier":
    model = DecisionTreeClassifier(**config["model"]["params"])
else:
    raise ValueError(f"Unsupported model type: {config['model']['type']}")

# Train Model
model.fit(X_train, y_train)

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("✅ Model trained and saved at models/model.pkl")
