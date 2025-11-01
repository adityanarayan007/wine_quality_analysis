import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pandas as pd

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#setting file paths
DATA_PATH = "data/final"
MODEL_PATH = "models/final_model.joblib"


# ---------------- Load Data ----------------
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    #X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    #y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))
    return X_train, y_train


#----------------- Train Model --------------
def train(X_train,y_train):
    logging.info("Started Model Training")
    model = RandomForestClassifier(max_depth=15,min_samples_split=5,n_estimators=250)
    model.fit(X_train,y_train)
    joblib.dump(model,MODEL_PATH)
    logging.info(f"Model Trainingf Complete and saved in {MODEL_PATH}")


#------------RUN TRAIN MODEL-----------------
if __name__ == "__main__":
    X_train,y_train = load_data()
    train(
        X_train=X_train,
        y_train=y_train
    )