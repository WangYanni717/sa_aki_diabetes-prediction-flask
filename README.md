# SA_AKI_Diabetes Clinical Prediction Model

## Overview

A Flask-based web application for clinical risk prediction using the CatBoost machine learning algorithm. The model predicts in-hospital mortality in ICU patients with Sepsis-Associated Acute Kidney Injury and Diabetes comorbidity based on 19 clinical indicators assessed within 24 hours of ICU admission.

## Quick Start

**Website Deployment:** https://sa-aki-diabetes-prediction.onrender.com

## Local Development

### Requirements

- Python 3.10
- See `requirements.txt` for detailed dependencies

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python app_flask.py
```

Access the application at `http://localhost:5001` in your web browser.

## Model Specifications

- **Algorithm**: CatBoost (Categorical Boosting)
- **Input Features**: 19 clinical indicators including:
  - Vital signs (heart rate, blood oxygen saturation, body temperature, diastolic pressure)
  - Laboratory values (platelets, white blood cells, red cell distribution width, lactate, glucose, blood urea nitrogen)
  - Hemostasis markers (prothrombin time, partial thromboplastin time, pH)
  - Fluid and renal parameters (urine output, fluid balance)
  - Vasopressor support (norepinephrine rate)
  - Organ dysfunction score (SOFA)
  - Patient demographics (age, weight)
- **Output**: Probability of in-hospital mortality
- **Model Evaluation**: The model performance was evaluated using metrics such as AUC (Area Under the Curve), accuracy, and precision. These metrics indicate how well the model predicts mortality risk based on clinical indicators.
- **Data Source**: MIMIC-IV dataset (70% for training, 30% for validation); eICU dataset as external test set
- **Data Preprocessing**: Outlier detection and handling, multiple imputation for missing values.

## Data Format

Input clinical values are validated against physiologically reasonable ranges. The model provides real-time risk stratification with accompanying clinical guidance based on detected risk factors.

## Repository Structure

```text
.
├── app_flask.py           # Flask web application
├── cat_model.pkl          # Trained CatBoost model
├── requirements.txt       # Python dependencies
├── procfile               # Deployment configuration
├── runtime.txt            # Python runtime version
├── README.md              # This file
└── .gitignore             # Git ignore file
```
