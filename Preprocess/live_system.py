import pandas as pd
import joblib
import time
import numpy as np

bundle = joblib.load("final_model.pkl")
model = bundle["model"]
le = bundle["encoder"]
features = bundle["features"]

df = pd.read_csv("data/final_dataset.csv").sample(200)

threshold = 0.75

def explain(row):
    reasons = []
    if row["packet_loss_ratio"] > 0.2:
        reasons.append("High packet loss")
    if row["packet_rate"] > 300:
        reasons.append("Traffic spike")
    if row["port_utilization"] > 80:
        reasons.append("Port congestion")
    return ", ".join(reasons) if reasons else "Unknown"

print("Real-time system started...\n")

for _, row in df.iterrows():

    row["protocol"] = le.transform([row["protocol"]])[0]

    X = row[features].values.reshape(1, -1)
    prob = model.predict_proba(X)[0][1]

    if prob > 0.9:
        level = "CRITICAL"
    elif prob > threshold:
        level = "DEGRADED"
    else:
        level = "NORMAL"

    print(f"{level} | Prob: {prob:.2f}")

    if prob > threshold:
        print("Cause:", explain(row))

    print("-" * 50)

    time.sleep(0.3)