# --- src/features/build_features.py ---
import pandas as pd
import os
import logging
import sys


# This allows imports like 'from src.features...' to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.features.feature_utilities import create_total_column_and_clean, traget_remodeling_util # 🌟 New Import


# Setup logging..
logging.basicConfig(level=logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')


# Configure file Paths and values (These remain global constants here)
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
FINAL_DATA_PATH = "data/final/"
X_TRAIN_PATH = os.path.join(FINAL_DATA_PATH,"X_train.csv")
X_TEST_PATH = os.path.join(FINAL_DATA_PATH,"X_test.csv")
Y_TRAIN_PATH = os.path.join(FINAL_DATA_PATH,"y_train.csv")
Y_TEST_PATH = os.path.join(FINAL_DATA_PATH,"y_test.csv")
ACIDITY_COLS = ["fixed acidity","volatile acidity"]
SULFUR_COLS = ["free sulfur dioxide","total sulfur dioxide"]
TARGET = 'quality' # 🌟 Changed to single string for cleaner usage

# Load data (No change needed)
def load_data(train_path:str, test_path:str):
    logging.info(f"Loading train and test data..")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    logging.info(f"Train data shape : {train_data.shape}")
    logging.info(f"Test data shape : {test_data.shape}")
    logging.info("Data Loaded Successfully...")
    return train_data,test_data

# Separate the target variable (Modified for string TARGET)
def separate_target(train_data,test_data,target_col:str):
    # Pass target as a list for drop, but use string for selection
    X_train = train_data.drop(columns=[target_col],axis=1) 
    X_test = test_data.drop(columns=[target_col],axis=1)

    y_train = train_data[[target_col]] # Keep as DataFrame for remodeling function
    y_test = test_data[[target_col]]

    logging.info("Target Feature has been separated from train and test Data...")
    return X_train,X_test,y_train,y_test

# prepare features (Uses imported utility functions)
def prepare_features(X_train,X_test):
    logging.info("Starting feature Preparation..")
    
    # 🌟 Uses imported utility function
    X_train = create_total_column_and_clean(X_train, ACIDITY_COLS, "total acidity")
    X_train = create_total_column_and_clean(X_train, SULFUR_COLS, "sulphur bound")
    X_test = create_total_column_and_clean(X_test, ACIDITY_COLS, "total acidity")
    X_test = create_total_column_and_clean(X_test, SULFUR_COLS, "sulphur bound")

    logging.info(f"Feature preparation complete, new shape for X_train : {X_train.shape}")
    logging.info(f"Feature preparation complete, new shape for X_test : {X_test.shape}")
    return X_train,X_test

# traget_remodeling (Uses imported utility functions)
def traget_remodeling(y_train,y_test, target_col:str):
    # 🌟 Uses imported utility function
    y_train_remodelled = traget_remodeling_util(y_train, target_col)
    y_test_remodelled = traget_remodeling_util(y_test, target_col)
    
    logging.info("Remodelled target data from multiclass to binary.")
    return y_train_remodelled, y_test_remodelled

# Save Data to data/final (No change needed)
def save_data(df:pd.DataFrame, path:str):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    df.to_csv(path,index=False)
    logging.info(f"Saved the final data into {path}")


# Driver Function
def feature_engineering():
    # 🌟 Simplified target usage by changing TARGET to a single string
    train_data,test_data = load_data(train_path=TRAIN_PATH,test_path=TEST_PATH)
    X_train,X_test,y_train,y_test = separate_target(train_data=train_data,test_data=test_data,target_col=TARGET)
    X_train,X_test = prepare_features(X_train=X_train,X_test=X_test)
    y_train,y_test = traget_remodeling(y_train=y_train,y_test=y_test, target_col=TARGET)
    save_data(X_train,X_TRAIN_PATH)
    save_data(X_test,X_TEST_PATH)
    save_data(y_train,Y_TRAIN_PATH)
    save_data(y_test,Y_TEST_PATH)
    logging.info("Feature Engineering Completed...")


# Run feature Engineering
if __name__ == "__main__":
    feature_engineering()