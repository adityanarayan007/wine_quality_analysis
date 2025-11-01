import os
import sys
from flask import Flask, render_template_string, request, redirect, url_for
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FIX for ModuleNotFoundError: No module named 'src' ---
# This ensures that Python can find modules inside the 'src' directory 
# when the script is run from the project root.
# It adds the project root (two levels up) to the system path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    # 🌟 Import the globally instantiated predictor object from your service file
    from src.service.predictor import predictor, ModelPredictor
except ImportError as e:
    logging.error(f"Failed to import ModelPredictor: {e}")
    predictor = None # Set to None if loading fails

app = Flask(__name__)

# --- Model Status Check ---
if predictor is None or not isinstance(predictor, ModelPredictor):
    # If the predictor failed to load, disable the prediction route and log.
    MODEL_LOAD_ERROR = True
    logging.error("Model predictor failed to initialize. Prediction service disabled.")
else:
    MODEL_LOAD_ERROR = False
    logging.info("Flask app initialized and ModelPredictor is ready.")

# --- HTML TEMPLATE (Includes Tailwind CSS for styling) ---
# NOTE: The HTML is served as a string for simplicity in deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wine Quality Classifier</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f7fafc; }
    </style>
</head>
<body>
    <div class="min-h-screen flex items-center justify-center p-4">
        <div class="bg-white shadow-2xl rounded-xl p-8 max-w-lg w-full">
            <h1 class="text-3xl font-bold text-gray-800 mb-6 border-b pb-3">
                Wine Quality Prediction 🍷
            </h1>
            
            {% if error_message %}
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded-md" role="alert">
                    <p class="font-bold">Error</p>
                    <p>{{ error_message }}</p>
                </div>
            {% endif %}

            {% if result is not none %}
                <div class="p-6 mb-6 rounded-lg shadow-inner 
                    {% if result == 1 %} bg-green-100 border border-green-400 {% else %} bg-yellow-100 border border-yellow-400 {% endif %}">
                    <h2 class="text-xl font-semibold mb-2 
                        {% if result == 1 %} text-green-700 {% else %} text-yellow-700 {% endif %}">
                        Prediction Result:
                    </h2>
                    <p class="text-3xl font-extrabold 
                        {% if result == 1 %} text-green-900 {% else %} text-yellow-900 {% endif %}">
                        {% if result == 1 %} HIGH Quality (&#x1F377;) {% else %} LOW Quality (&#x1F97A;) {% endif %}
                    </p>
                    <p class="text-sm text-gray-600 mt-2">
                        (Quality score >= 6 is classified as HIGH)
                    </p>
                </div>
            {% endif %}

            <form method="POST" action="/predict" class="space-y-4">
                <p class="text-gray-600 mb-4">Enter the 11 chemical properties of the wine:</p>
                
                <div class="grid grid-cols-2 gap-4">
                    {% for name, default_val in features %}
                    <div>
                        <label for="{{ name }}" class="block text-sm font-medium text-gray-700 capitalize">{{ name.replace('_', ' ') }}:</label>
                        <input type="number" step="0.001" id="{{ name }}" name="{{ name }}" 
                               value="{{ default_val }}" required
                               class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm">
                    </div>
                    {% endfor %}
                </div>

                <button type="submit" 
                        class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-lg font-medium text-white 
                               bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition duration-150 ease-in-out">
                    Get Prediction
                </button>
            </form>
            
            {% if model_error %}
                <p class="text-center text-red-500 mt-4 text-xs font-semibold">
                    [ERROR] Model failed to load. Please check artifacts/models/final_model.pkl path.
                </p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# Define the 11 input features and default values (Red Wine sample)
INPUT_FEATURES = [
    ("fixed acidity", 7.4),
    ("volatile acidity", 0.70),
    ("citric acid", 0.00),
    ("residual sugar", 1.9),
    ("chlorides", 0.076),
    ("free sulfur dioxide", 11.0),
    ("total sulfur dioxide", 34.0),
    ("density", 0.9978),
    ("pH", 3.51),
    ("sulphates", 0.56),
    ("alcohol", 9.4)
]

@app.route('/', methods=['GET'])
def index():
    """Renders the prediction form."""
    return render_template_string(
        HTML_TEMPLATE, 
        features=INPUT_FEATURES, 
        result=None, 
        error_message=None,
        model_error=MODEL_LOAD_ERROR
    )

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles form submission, processes data, and returns prediction."""
    
    if MODEL_LOAD_ERROR:
        return render_template_string(
            HTML_TEMPLATE, 
            features=INPUT_FEATURES, 
            result=None, 
            error_message="Prediction service is currently down (Model Artifacts Missing/Failed to Load).",
            model_error=MODEL_LOAD_ERROR
        )

    # Convert form data to a dictionary of floats
    input_data: Dict[str, Any] = {}
    current_feature_name = None  # 🌟 FIX: Initialize current_feature_name outside the try block

    try:
        for name, _ in INPUT_FEATURES:
            current_feature_name = name  # 🌟 FIX: Assign the name before processing the value
            
            # We are using the original name (with spaces) as the dictionary key
            value = request.form.get(name)
            
            if value is None or value == '':
                # Treat empty or None as a ValueError for required fields
                raise ValueError(f"Input for {name} cannot be empty.")
                
            input_data[name] = float(value)
            
    except ValueError as ve:
        # Use the safely assigned variable for informative feedback
        feature_name_display = current_feature_name if current_feature_name else "an input field"
        
        logging.error(f"Input conversion failed for '{feature_name_display}': {ve}")

        return render_template_string(
            HTML_TEMPLATE, 
            features=INPUT_FEATURES, 
            result=None, 
            # Provide specific feedback to the user
            error_message=f"Invalid input for '{feature_name_display}'. Please ensure all values are valid numbers and are not left blank.",
            model_error=MODEL_LOAD_ERROR
        )

    logging.info(f"Received prediction request with data: {json.dumps(input_data)}")
    
    try:
        # Call the globally loaded predictor object
        prediction_result = predictor.predict(input_data)
        
        logging.info(f"Prediction made: {prediction_result}")

        # Redirect back to the index page with the result in the query string
        return render_template_string(
            HTML_TEMPLATE, 
            features=INPUT_FEATURES, 
            result=prediction_result, 
            error_message=None,
            model_error=MODEL_LOAD_ERROR
        )
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return render_template_string(
            HTML_TEMPLATE, 
            features=INPUT_FEATURES, 
            result=None, 
            error_message=f"An unexpected error occurred during prediction: {e}",
            model_error=MODEL_LOAD_ERROR
        )


if __name__ == '__main__':
    # To run the development server:
    # Navigate to the project root directory
    # python src/api/app.py 
    app.run(debug=True, port=8080)
