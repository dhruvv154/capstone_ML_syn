import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np


# ✅ LOAD FIRST
df = pd.read_csv("data/final_dataset.csv")

# ✅ THEN MODIFY
df["packet_rate"] *= np.random.uniform(0.9, 1.1, len(df))

# Load dataset
df = pd.read_csv("data/final_dataset.csv")

# Encode protocol
le = LabelEncoder()
df["protocol"] = le.fit_transform(df["protocol"])

# Features
X = df.drop(columns=["anomaly_label"])
y = df["anomaly_label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

importances = model.feature_importances_
feature_names = X.columns

# Sort
sorted_idx = importances.argsort()

plt.figure(figsize=(8, 5))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig("feature_importance.png")

print("Feature importance saved as feature_importance.png")

# ----------------------------
# SAVE MODEL
# ----------------------------

joblib.dump({
    "model": model,
    "encoder": le,
    "features": X.columns.tolist()
}, "final_model.pkl")

print("Model saved successfully")