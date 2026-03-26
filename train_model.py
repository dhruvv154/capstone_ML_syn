import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("sdn_dataset.csv")

print("Dataset loaded successfully")
print("Total samples:", len(df))

# Encode protocol
le = LabelEncoder()
df["protocol"] = le.fit_transform(df["protocol"])

# Features & labels
X = df.drop(columns=["anomaly_label"])
y = df["anomaly_label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training complete")

# Evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "sdn_model.pkl")

print("\nModel saved as sdn_model.pkl ✅")
