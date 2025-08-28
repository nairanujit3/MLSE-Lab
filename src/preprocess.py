import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ======================
# Load Config
# ======================
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# ======================
# Load Raw Data
# ======================
df = pd.read_csv(config["dataset"])

# 'loan_status' is the target column
target_col = "loan_status"
X = df.drop(columns=[target_col])
y = df[target_col]

# ======================
# Identify Columns
# ======================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

transformers = []
if config["preprocessing"].get("scale", False) and numerical_cols:
    transformers.append(("num", StandardScaler(), numerical_cols))

if config["preprocessing"].get("encode", False) and categorical_cols:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

if not transformers:
    X_processed = X.copy()
else:
    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)

    # Convert back to DataFrame with reset index
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = pd.DataFrame(X_processed, index=X.index)


X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)

# Reset indices to avoid misalignment
train_df = pd.concat([pd.DataFrame(X_train).reset_index(drop=True),
                      y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([pd.DataFrame(X_test).reset_index(drop=True),
                     y_test.reset_index(drop=True)], axis=1)


os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Preprocessing complete: train/test saved to data/processed/")
