# Logistic Regression on Mini Clinical Dataset

This project demonstrates binary classification using logistic regression on a small, synthetic clinical dataset.

## Dataset

- Manually created with 5 features:
  - symptom (categorical)
  - temperature (numeric with NaN & outliers)
  - blood_pressure (numeric with NaN & outliers)
  - smoker (categorical)
  - visit_doctor (target: yes/no)

## Preprocessing Steps

- Fill missing values with median
- Remove outliers using IQR
- One-hot encode categorical features
- Standard scale numerical features

## Model

- Logistic Regression (`sklearn`)
- Train-test split: 70-30

## Evaluation

- Confusion Matrix
- ROC Curve

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python clinic_logistic_regression.py
```
