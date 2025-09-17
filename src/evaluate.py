import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

def evaluate_model():
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop(test_df.columns[-1], axis=1)
    y_test = test_df[test_df.columns[-1]].astype(int)

    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    model_name = type(model).__name__
    params = model.get_params()

    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "n_estimators": params.get("n_estimators")
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    Path("experiments").mkdir(exist_ok=True)
    results_file = Path("experiments/results.json")

    experiment_log = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "n_estimators": params.get("n_estimators")
    }

    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    results.append(experiment_log)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete")
    print("Metrics saved to metrics.json (for DVC exp show)")
    print("Full log saved/appended to experiments/results.json")

if __name__ == "__main__":
    evaluate_model()
