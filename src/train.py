import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def load_config(config_path='params.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config):
    train_df = pd.read_csv('data/processed/train.csv')
    X_train = train_df.drop(train_df.columns[-1], axis=1)
    y_train = train_df[train_df.columns[-1]].astype(int)

    model_params = config['model']['params']
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    Path('models').mkdir(exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    config = load_config()
    train_model(config)