🍷 Wine Quality Prediction Service

1. Project Overview

This project implements an end-to-end Machine Learning solution to classify red wine samples based on their physiochemical properties. The goal is to predict whether a wine is considered "Good Quality" (score ≥ 7) or "Poor Quality" (score < 7) using a robust and deployable model.

This is a demonstration of a complete MLOps pipeline, from training and artifact management to containerized deployment.

2. Project Goals

ML Performance Objectives

Target Metric: Achieve a minimum of 85% F1-Score on the validation set for the "Good Quality" class (minority class).

Model Selection: Utilize a robust ensemble method (Random Forest or Gradient Boosting) for high predictive power and interpretability.

Data Strategy: Implement Feature Engineering to derive better predictive features (e.g., pH to fixed acidity ratios) to improve model generalization.

MLOps and Deployment Objectives

Code Quality: Maintain a modular structure with separate directories for source code (src/), API (src/api/), and services (src/services/).

Containerization: Successfully package the model and prediction service into a small, production-ready Docker image using Gunicorn.

Scalable Deployment: Deploy the containerized service to a scalable cloud platform (Render/GCP Cloud Run) for public access and demonstration.

3. Technology Stack

Category

Component

Description

Language

Python 3.10

Core programming language.

ML/Data

Scikit-learn, Pandas

Model training, feature engineering, and data handling.

Web Service

Flask, Gunicorn

Lightweight API and production-grade web server.

Artifacts

Joblib

Serialization and loading of the trained model (final_model.joblib).

Deployment

Docker

Containerization for reproducible builds.

Cloud Hosting

Render / Google Cloud Run

Serverless platform for hosting the public API.

4. Quick Analysis: How Values Impact Wine Quality

For users testing the prediction service, understanding how the input features affect the model's output is critical. The model has learned specific thresholds, but here are the general trends:

Feature

Trend for Good Quality

Reasoning

Alcohol

Higher values (e.g., > 11.5)

Often the strongest predictor; higher alcohol content is correlated with higher perceived quality.

Volatile Acidity

Lower values (e.g., < 0.40)

Too much volatile acidity makes wine taste like vinegar; lower values are highly desired.

Sulphates

Higher values (e.g., > 0.65)

Sulphates (SO2) act as a preservative. Higher levels often indicate better stability and quality control.

Fixed Acidity

Varies, but Mid-Range

Too low is flat; too high is sour. Needs to be balanced with pH for the best result.

Pro-Tip: To quickly test for a "Good Quality" result, try a combination of High Alcohol (12.0), Low Volatile Acidity (0.35), and High Sulphates (0.80), keeping other values near the average.

5. MLOps / Deployment Guide

The entire service is packaged in a single Docker image, making it easy to build and run locally or deploy to any cloud platform (ECS, Cloud Run, Azure Container Apps).

Prerequisites

Docker Desktop installed and running.

The trained model artifact (final_model.joblib) must be present in the models/ directory.

Local Run Commands

Stop/Remove Previous Container (if necessary):

docker stop wine_service
docker rm wine_service


Build the Docker Image:
This command packages the source code, dependencies, and model into an image tagged wine-predictor.

docker build -t wine-predictor .


Run the Container:
This starts the service in detached mode (-d), exposing it on your local port 5000.

docker run -d -p 5000:8080 --name wine_service wine-predictor


Access the Application:
Open your browser to: http://localhost:5000

Live Demo Available Here: [INSERT PUBLIC RENDER / GCP CLOUD RUN URL]