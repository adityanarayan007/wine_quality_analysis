import pandas as pd
import joblib
import os
import logging
from typing import Dict, Any, List
import sys


# This allows imports like 'from src.features...' to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 🌟 IMPORT THE PURE FEATURE UTILITY
# Assumes feature_utilities.py is in src/features/
from src.features.feature_utilities import create_total_column_and_clean 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (Must match your pipeline's global constants) ---
MODEL_PATH = os.path.join('models', 'final_model.joblib')

# Columns used for feature engineering
ACIDITY_COLS: List[str] = ["fixed acidity", "volatile acidity"]
SULFUR_COLS: List[str] = ["free sulfur dioxide", "total sulfur dioxide"]

# The list of FINAL features expected by your trained model (Order is CRUCIAL!)
# You must generate this list after your build_features.py pipeline runs successfully
FINAL_FEATURE_COLS: List[str] = [
    'citric acid', 'residual sugar', 'chlorides', 'density', 'pH', 
    'sulphates', 'alcohol', 'total acidity', 'sulphur bound' 
    # Example order: Original columns that remain, followed by your two new calculated columns
]


class ModelPredictor:
    """
    Manages loading artifacts and running the end-to-end prediction pipeline.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the trained model and the fitted scaler using joblib."""
        try:
            self.model = joblib.load(MODEL_PATH)
            logging.info("Model and Scaler artifacts loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Failed to load artifact: {e}")
            raise RuntimeError("Required model or scaler file not found. Deployment cannot proceed.")

    def _feature_engineer(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the exact same feature engineering steps used during training 
        by calling the imported utility function.
        """
        data_df = data_df.copy()
        
        # 1. Create Total Acidity
        data_df = create_total_column_and_clean(data_df, ACIDITY_COLS, "total acidity")
        
        # 2. Create Sulphur Bound
        data_df = create_total_column_and_clean(data_df, SULFUR_COLS, "sulphur bound")
        
        # 3. Final Step: Select and reorder columns to match the model's expected input
        try:
            return data_df[FINAL_FEATURE_COLS]
        except KeyError as e:
             logging.error(f"Missing feature in input data: {e}")
             raise ValueError(f"Input data is missing expected feature: {e}")

    def predict(self, raw_data: Dict[str, Any]) -> int:
        """
        Takes raw input (from the API), preprocesses it, and returns the prediction.
        """
        # 1. Convert raw input dictionary to a pandas DataFrame (1 row)
        input_df = pd.DataFrame([raw_data])
        
        # 2. Feature Engineering (using the reusable logic)
        engineered_df = self._feature_engineer(input_df)
        
        # 3. Scaling (Using the fitted scaler's .transform())
        # The output is a NumPy array
        #scaled_features = self.scaler.transform(engineered_df)
        
        # 4. Prediction
        prediction = self.model.predict(engineered_df)
        
        # Return the prediction as a standard Python integer
        return int(prediction[0])

# --- Instantiate the Predictor Globally ---
# This loads artifacts ONLY ONCE when the service starts
try:
    predictor = ModelPredictor()
except RuntimeError:
    predictor = None 

# --- Example Usage (If running script directly) ---
if __name__ == '__main__':
    # NOTE: All 11 original columns must be present in the input dictionary!
    sample_input = {
        "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00, 
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0, 
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51, 
        "sulphates": 0.56, "alcohol": 9.4
    }
    
    if predictor:
        result = predictor.predict(sample_input)
        print(f"\nPrediction Result: {result} (0=Poor/Average, 1=Good/Average)")