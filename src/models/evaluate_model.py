import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix
)


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


#setting file paths and other parameters
DATA_PATH = "data/final"
MODEL_PATH = "models/final_model.joblib"
REPORT_PATH = "artifacts/results/final_results.csv"
# Set the average mode based on successful binary classification (Quality >= 6 vs < 6)
AVERAGE_MODE = 'binary' 
# If the problem was multi-class, would use 'weighted' or 'macro'.


#----------------Load test data-------------
def load_test_data():
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))
    return X_test, y_test


#--------------model evaluation---------------
def model_evaluation(model, X_test: pd.DataFrame, y_test: pd.DataFrame, model_name: str, path: str):
    """
    Performs classification model evaluation, calculates all key metrics, 
    and saves the results to a structured CSV file.
    
    Args:
        model: The trained scikit-learn model object.
        X_test: Test features.
        y_test: True test labels (DataFrame or Series).
        model_name: Name of the model (e.g., "RandomForest").
        path: Full file path to save the results CSV.
    """
    logging.info(f"Model evaluation started for {model_name}...")
    
    # 1. Generate Predictions
    # Get hard predictions (0 or 1)
    predictions = model.predict(X_test)
    
    # Get probability scores for AUC-ROC
    try:
        # Most classifiers have predict_proba; we take the score for the positive class (index 1)
        probabilities = model.predict_proba(X_test)[:, 1] 
        has_proba = True
    except AttributeError:
        logging.warning("Model does not have predict_proba method. AUC-ROC will be skipped.")
        has_proba = False

    # Ensure y_test is a flat array/Series for metric functions
    y_true = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test
    
    # 2. Calculate Metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, predictions),
        'F1_Score': f1_score(y_true, predictions, average=AVERAGE_MODE, zero_division=0),
        'Precision': precision_score(y_true, predictions, average=AVERAGE_MODE, zero_division=0),
        'Recall': recall_score(y_true, predictions, average=AVERAGE_MODE, zero_division=0),
        'Confusion_Matrix': confusion_matrix(y_true, predictions).tolist(), # Convert to list for easy storage
        # Add a placeholder for AUC-ROC
        'AUC_ROC': np.nan 
    }
    
    if has_proba:
        metrics['AUC_ROC'] = roc_auc_score(y_true, probabilities)
        logging.info(f"AUC-ROC calculated: {metrics['AUC_ROC']:.4f}")

    # 3. Create and Save Results DataFrame
    
    # Create a DataFrame from the single metrics dictionary
    results_df = pd.DataFrame([metrics])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # check: Append results if the file already exists, otherwise create a new one
    if os.path.exists(path):
        # Load existing results
        existing_df = pd.read_csv(path)
        # Check if the current model result already exists to avoid duplication
        if model_name in existing_df['Model'].values:
            logging.warning(f"Model {model_name} already exists in results file. Overwriting...")
            existing_df = existing_df[existing_df['Model'] != model_name]

        # Combine old and new results
        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    # Save the updated DataFrame
    final_df.to_csv(path, index=False)
    
    logging.info(f"✅ Model evaluation complete. Results saved to {path}")
    logging.info("\n--- Final Results Snippet ---")
    logging.info(final_df.sort_values(by='F1_Score', ascending=False).to_markdown(index=False, floatfmt=".4f"))



#--------------Runner function for evaluation--------------
def evaluate():
    model = joblib.load(MODEL_PATH)
    X_test,y_test = load_test_data()
    model_evaluation(model=model,X_test=X_test,y_test=y_test,model_name="RandonForest",path = REPORT_PATH)

if __name__ == "__main__":
    evaluate()
    

