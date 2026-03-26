import pandas as pd
import joblib
import time
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Configuration Parameters
# ----------------------------

THRESHOLD = 0.8            # Anomaly probability threshold
CONSECUTIVE_LIMIT = 3      # Number of consecutive anomalies required
COOLDOWN_SECONDS = 10      # Time between alerts

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("sdn_model.pkl")

# Load dataset (simulating streaming input)
df = pd.read_csv("sdn_dataset.csv")

# Encode protocol
le = LabelEncoder()
df["protocol"] = le.fit_transform(df["protocol"])

X = df.drop(columns=["anomaly_label"])

print("🚀 Starting Real-Time SDN Monitoring...\n")

anomaly_counter = 0
last_alert_time = 0

# ----------------------------
# Real-Time Simulation Loop
# ----------------------------

for i in range(len(X)):

    sample = X.iloc[i:i+1]

    # Get anomaly probability (class 1)
    probability = model.predict_proba(sample)[0][1]

    # Check if above threshold
    if probability > THRESHOLD:
        anomaly_counter += 1
        print(f"⚠️ Suspicious Activity Detected | Prob: {probability:.2f} | Count: {anomaly_counter}")
    else:
        anomaly_counter = 0
        print("✅ Traffic Normal")

    # Confirmed alert condition
    if anomaly_counter >= CONSECUTIVE_LIMIT:
        current_time = time.time()

        # Cooldown check
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            print("\n🚨 CONFIRMED NETWORK FAILURE DETECTED 🚨")
            print(f"Anomaly Probability: {probability:.2f}")
            print("Taking preventive action / sending notification...")
            print("--------------------------------------------------\n")

            last_alert_time = current_time

    time.sleep(0.2)  # simulate real-time packet arrival
