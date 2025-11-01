import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score,precision_score,accuracy_score,confusion_matrix,recall_score


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#setting file paths
DATA_PATH = "data/final"
MODEL_PATH = "artifacts/models/model.joblib"
REPORT_PATH = "artifacts/results/optimization_results.csv"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


# ---------------- Load Data ----------------
def load_data():
    X = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    y = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    return X, y

# ---------------- Define Model Search Space ----------------
def get_search_spaces():
    return{
        "RandomForest":{
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200, 250, 300],
                    "max_depth": [5, 10, 15 ,20],
                    "min_samples_split": [2, 3, 5, 8]
                }
        }
    }


# ---------------- Optimization Pipeline ----------------
def optimize_models(X, y):
    results = []
    best_score = 0
    best_model = None
    best_name = None

    search_spaces = get_search_spaces()

    for name, config in search_spaces.items():
        logging.info(f"🔍 Tuning {name}...")
        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        grid.fit(X, y)

        best_params = grid.best_params_
        best_f1 = grid.best_score_
        results.append({
            "Model": name,
            "BestParams": best_params,
            "CV_F1": best_f1
        })

        if best_f1 > best_score:
            best_score = best_f1
            best_model = grid.best_estimator_
            best_name = name

        logging.info(f"{name}: Best F1 = {best_f1:.4f}, Params = {best_params}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_PATH, index=False)
    logging.info(f"Saved optimization report at {REPORT_PATH}")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    logging.info(f"✅ Best model ({best_name}) saved at {MODEL_PATH}")

# ---------------- Main ----------------
if __name__ == "__main__":
    X, y = load_data()
    optimize_models(X, y)
    