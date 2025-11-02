🍷 Wine Quality Prediction – End-to-End Machine Learning Project
🚀 Overview

This project predicts the quality of red wine based on its physicochemical properties such as acidity, sugar level, pH, alcohol content, and more.
It’s built as a complete MLOps-style pipeline, including model training, optimization, deployment with Flask, and containerization via Docker.

🧩 Project Architecture
Wine Quality Prediction
│
├── data/
│   ├── raw/                   # Original dataset
│   ├── processed/             # Cleaned and transformed data
│
├── notebooks/
│   ├── EDA_and_Model_Comparison.ipynb  # Exploratory data analysis and initial model experiments
│
├── src/
│   ├── data/                  # Data loading and preprocessing scripts
│   ├── features/              # Feature engineering logic
│   ├── models/                # Model training, tuning, and evaluation scripts
│   ├── app/                   # Flask web application
│
├── artifacts/
│   ├── models/                # Saved model, encoders, scalers
│
├── templates/                 # HTML templates for the web app
├── static/                    # CSS / JS / Images for UI
│
├── Dockerfile                 # Container configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── app.py                     # Flask app entry point

⚙️ Tech Stack

Language: Python 3.10

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Framework: Flask

Model: Random Forest Classifier

Containerization: Docker

Deployment: Render

🧠 Machine Learning Workflow

Data Ingestion: Loaded and cleaned the Wine Quality dataset (UCI Machine Learning Repository).

EDA: Identified key correlations between features and wine quality (alcohol, acidity, sulfur compounds, etc.).

Feature Engineering: Scaled numerical features and handled outliers. combined acidity columns to total acidity and sulfur columns to sulfur bound

Model Selection: Tested multiple algorithms — Logistic Regression, SVM, Decision Tree, Random Forest and Gradient Boosting.

Optimization: Tuned hyperparameters using GridSearchCV.

Evaluation: Assessed using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Deployment: Built Flask app and containerized using Docker.

Render Hosting: Deployed publicly accessible prediction web app.
🎯 Results
Model         Accuracy    F1_Score              Precision             Recall
RandForrest   0.796875,   0.8209366391184573,   0.8097826086956522,   0.8324022346368715
SVC           0.75,       0.7687861271676301,   0.7964071856287425,   0.7430167597765364
LogReg        0.7375,     0.76,                 0.7777777777777778,   0.7430167597765364
GradBoost     0.734375,   0.7578347578347578,   0.7732558139534884,   0.7430167597765364
DecTree       0.73125,    0.7570621468926554,   0.7657142857142857,   0.7486033519553073

✅ The Random Forest Classifier achieved the best performance with balanced precision and recall.
Following parametrs were best for RandomForest which was determined from GridSearchCV : 
RandomForest,"{'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 250}"

# 1️⃣ Clone the repository
git clone https://github.com/adityanarayan007/wine_quality_analysis

# 2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Flask app
python app.py

# 5️⃣ Visit the app
Open http://127.0.0.1:5000/ in your browser



# Build the Docker image
docker build -t wine-quality-app .

# Run the container
docker run -p 5000:5000 wine-quality-app


🧭 Understanding the Features and Their Impact on Wine Quality
| Feature                       | Description                                                 | Typical Range | Impact on Quality                                                                |
| ----------------------------- | ----------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------- |
| **Fixed Acidity**             | Acids that don’t evaporate easily (tartaric, malic, citric) | 4.0 – 15.0    | Moderate acidity contributes to freshness; too high makes wine sour.             |
| **Volatile Acidity**          | Acetic acid (vinegar-like)                                  | 0.1 – 1.6     | High values reduce quality; creates unpleasant smell/taste.                      |
| **Citric Acid**               | Adds flavor and stability                                   | 0.0 – 1.0     | Higher citric acid often improves quality slightly.                              |
| **Residual Sugar**            | Sugar left after fermentation                               | 0.9 – 15.5    | Sweetness increases drinkability but too much reduces quality.                   |
| **Chlorides**                 | Salt content                                                | 0.01 – 0.2    | High chloride = salty taste → lower quality.                                     |
| **Free Sulfur Dioxide (SO₂)** | Prevents microbial growth                                   | 1 – 75        | Moderate SO₂ protects wine; too high creates off-flavors.                        |
| **Total Sulfur Dioxide**      | Sum of free and bound SO₂                                   | 6 – 300       | Very high levels indicate poor handling → lower quality.                         |
| **Density**                   | Measure of sugar + alcohol                                  | 0.990 – 1.004 | Lower density (more alcohol, less sugar) → better quality.                       |
| **pH**                        | Acidity level                                               | 2.8 – 4.0     | Ideal wines have balanced pH (~3.2–3.5). Too high = dull, too low = overly sour. |
| **Sulphates**                 | Adds antioxidant property                                   | 0.3 – 1.6     | Higher sulphates usually correlate with better preservation and quality.         |
| **Alcohol**                   | Percentage of ethanol                                       | 8.0 – 14.9    | Strong positive correlation: higher alcohol = higher perceived quality.          |

Then open: http://localhost:5000

🌐 Live Demo

🚀 Try the Web App on Render

✅ Tip for Users:
When using the app, enter realistic values within these ranges.
The prediction assumes the input follows the same structure and scaling as the dataset.

🧾 Future Improvements

Integrate CI/CD with GitHub Actions

Add automated model retraining pipeline

Improve UI with better input validation

Enable batch predictions via CSV upload

✨ Author

Aditya Narayan Mishra
📧 [Your Email]
🔗 [LinkedIn Profile] | [Portfolio Website]