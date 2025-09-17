import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_config(config_path="params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    raw_data_path = config["data"]["raw"]
    train_path = config["data"]["train"]
    test_path = config["data"]["test"]
    target_col = config["data"]["target_col"]

    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    
    df = pd.read_csv(raw_data_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset. "
                       f"Available columns: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    
    if config["preprocessing"]["scale"]:
        scaler = StandardScaler()
        X[X.select_dtypes(include=["number"]).columns] = scaler.fit_transform(
            X.select_dtypes(include=["number"])
        )

    if config["preprocessing"]["encode"]:
        encoder = LabelEncoder()
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = encoder.fit_transform(X[col])

    df_processed = pd.concat([X, y], axis=1)
    
    train_df, test_df = train_test_split(
        df_processed,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

   
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train data saved at: {train_path}")
    print(f"✅ Test data saved at: {test_path}")

if __name__ == "__main__":
    main()
