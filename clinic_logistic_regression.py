import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Sample mini clinical dataset
data = {
    "symptom":       ["cough", "headache", "none", "cough", "fever", "fever",
                      "none", "cough", "cough", "none", "headache", "fever",
                      "none", "cough", "cough"],

    "temperature":   [37.2, 36.8, 36.5, 38.5, 39.1, 40.2, 36.4, 37.9, np.nan, 36.7, 36.6, 38.8, 36.2, 37.3, 45.0],  # NaN & outlier

    "blood_pressure": [120, 130, 115, 140, 150, 160, 118, 135, 128, np.nan, 110, 145, 125, 132, 90],  # NaN & outlier

    "smoker":        ["no", "yes", "no", "yes", "no", "no", "yes", "yes", "no", "no", "yes", "yes", "no", "yes", "no"],

    "visit_doctor":  ["no", "yes", "no", "yes", "yes", "yes", "no", "yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
}

df = pd.DataFrame(data)
print("Raw data:"); display(df)

# 2. Fill missing values
df["temperature"].fillna(df["temperature"].median(), inplace=True)
df["blood_pressure"].fillna(df["blood_pressure"].median(), inplace=True)

# 3. Remove outliers using IQR
for col in ["temperature", "blood_pressure"]:
    q1, q3 = df[col].quantile([.25, .75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

# 4. One-hot encode categorical features
df_enc = pd.get_dummies(df, columns=["symptom", "smoker"], drop_first=True)

# 5. Scale numeric features
features = ["temperature", "blood_pressure"]
df_enc[features] = StandardScaler().fit_transform(df_enc[features])

# 6. Logistic Regression model
X = df_enc.drop("visit_doctor", axis=1)
y = df_enc["visit_doctor"].map({"no": 0, "yes": 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression().fit(X_train, y_train)

# 7. Evaluation: Confusion Matrix and ROC Curve
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()
