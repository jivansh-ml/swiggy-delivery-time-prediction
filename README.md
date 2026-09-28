# Delivery Time Prediction

An end-to-end machine learning pipeline that predicts food delivery time using the Swiggy delivery dataset. Built with a focus on reproducibility, experiment tracking, and cloud deployment.

🔗 **Live Demo:** [delivery-time-prediction.onrender.com](https://delivery-time-prediction-vgk0.onrender.com/)
📊 **Experiment Tracking:** [DagsHub Project](https://dagshub.com/jivanshs51/swiggy-delivery-time-prediction)

---

## Overview

This project predicts how long a food delivery will take based on rider, weather, traffic, and order-level features from the Swiggy dataset. It covers the full ML lifecycle — from raw data cleaning to a model served through a Flask web app on Render.

**Key results:**
- **45,593** delivery records cleaned and processed
- **15 engineered features** used for prediction
- **3.02-minute MAE**, **0.84 R²** on the held-out test set
- **50+ MLflow-tracked experiment runs** across model families (KNN, SVM, Random Forest, XGBoost, LightGBM)
- Final model: a **stacking regressor** (Random Forest + LightGBM base learners, Linear Regression meta-model)

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Modeling | Scikit-learn, LightGBM, Optuna |
| Experiment Tracking | MLflow (hosted on DagsHub) |
| Data & Model Versioning | DVC (DagsHub remote storage) |
| Web Framework | Flask |
| Serving | Gunicorn |
| Deployment | Render |

---

## Pipeline Architecture

The project follows a DVC-orchestrated pipeline with six stages:

```
data_cleaning → data_preparation → data_preprocessing → train → evaluation → register_model
```

- **data_cleaning** — drops invalid records (e.g. underage riders, erroneous ratings), parses timestamps, fixes coordinate signs, and derives base time/location fields
- **data_preparation** — engineers features such as `pickup_time_minutes`, `distance`, `order_time_of_day`, `is_weekend`, and cleans categorical fields (weather, traffic, vehicle type, festival, city type)
- **data_preprocessing** — encodes categorical variables, scales numerical features, and applies a power transform to the target variable
- **train** — trains and tunes candidate models using Optuna, then fits the final stacking regressor
- **evaluation** — computes MAE, R², and cross-validated scores on the test set
- **register_model** — logs and registers the final model artifact via MLflow on DagsHub

Each run is reproducible with:

```bash
dvc repro
```

---

## Model Development

Model selection was run as an Optuna study across six algorithms (KNN, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM), with all trials logged to MLflow.

| Model | Avg. MAE (Optuna trials) |
|---|---|
| **LightGBM** | **3.09 min** |
| XGBoost | 3.32 min |
| Gradient Boosting | 3.42 min |
| Random Forest | 3.96 min |
| KNN | 4.76 min |
| SVM | 5.92 min |

The final model is a **stacking regressor** combining tuned Random Forest and LightGBM base estimators with a Linear Regression meta-model, achieving:

- **Test MAE:** 3.02 minutes
- **Test R²:** 0.84
- **5-fold CV MAE:** ~3.06 minutes

---

## Project Structure

```
├── data/                   # DVC-tracked raw and processed data
├── src/
│   ├── data/                # data_cleaning, data_preparation, data_preprocessing scripts
│   ├── model/                # train, evaluate, register scripts
│   └── ...
├── models/
│   ├── model.joblib
│   ├── stacking_regressor.joblib
│   ├── power_transformer.joblib
│   └── preprocessor.joblib
├── app.py                   # Flask application
├── dvc.yaml                 # Pipeline definition
├── params.yaml               # Pipeline parameters
├── requirements.txt
└── README.md
```

> Update this section to match your actual repo layout.

---

## Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/jivansh-ml/swiggy-delivery-time-prediction.git
   cd swiggy-delivery-time-prediction
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Pull DVC-tracked data and model artifacts
   ```bash
   dvc pull
   ```

4. Reproduce the pipeline (optional)
   ```bash
   dvc repro
   ```

---

## Running the App Locally

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

The app loads `model.joblib` and `preprocessor.joblib` directly from disk for inference — no MLflow/DagsHub calls happen at request time.

---

## Deployment

The application is deployed on **Render**: https://delivery-time-prediction-vgk0.onrender.com/

---


your license here]
