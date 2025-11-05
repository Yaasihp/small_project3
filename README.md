# small_project3
# Distributed Logistic Regression for Diabetes Prediction

## Overview
This project uses PySpark and multiple virtual machines (VMs) to perform
statistical machine learning (binary classification) on a large public health
dataset (>250,000 records), predicting whether individuals have diabetes.

The workflow includes:
- Preprocessing & feature engineering
- Weighted Logistic Regression model
- Accuracy, AUC, precision, recall evaluation
- Runtime comparison using 1 VM vs. 2 VMs

---

## Dataset
BRFSS 2015 Diabetes Health Indicators  
Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset  

Target variable: `Diabetes_binary`  
Class balance: ~15% positive, ~85% negative

The dataset is numeric, clean, and contains binary + continuous features.

---

## Files
- `main_spark_ml.py` — PySpark model pipeline + evaluation
- `screenshots/` — Spark UI + terminal output
- `report/` — Project write-up (optional)

---

## How to Run

### Distributed (2-VM cluster)
