import pandas as pd
import numpy as np

# ----------------------------
# LOAD DATA
# ----------------------------

df1 = pd.read_csv("data/sdn_dataset.csv")
df2 = pd.read_csv("data/dataset_sdn.csv")

print("DF1:", df1.shape)
print("DF2:", df2.shape)

# ----------------------------
# STANDARDIZE DF1 (synthetic)
# ----------------------------

df1_clean = pd.DataFrame({
    "protocol": df1.get("protocol", "TCP"),
    "packet_rate": df1.get("packet_rate", 0),
    "byte_count": df1.get("byte_count", 0),
    "flow_duration_sec": df1.get("flow_duration_sec", 1),
    "packet_loss_ratio": df1.get("packet_loss_ratio", 0)
})

# ----------------------------
# STANDARDIZE DF2 (telemetry)
# ----------------------------

df2_clean = pd.DataFrame({
    "protocol": df2.get("Protocol", "TCP"),
    "packet_rate": df2.get("throughput_rate", 0),
    "byte_count": df2.get("throughput_bytes", 0),
    "flow_duration_sec": df2.get("time_diff", 1),
    "packet_loss_ratio": df2.get("drop_rate", 0)
})

# ----------------------------
# MERGE SAFELY
# ----------------------------

df = pd.concat([df1_clean, df2_clean], ignore_index=True)

print("Merged shape:", df.shape)

# ----------------------------
# CLEAN DATA
# ----------------------------

df = df.replace([np.inf, -np.inf], 0)
df = df.dropna()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------

df["port_utilization"] = np.clip(df["packet_rate"] / 500 * 100, 0, 100)

# ----------------------------
# ANOMALY INJECTION
# ----------------------------

df["anomaly_label"] = 0

# Inject anomalies ONLY if dataset is big enough
if len(df) > 50:
    idx = np.random.choice(df.index, int(0.2 * len(df)), replace=False)

    df.loc[idx, "packet_rate"] *= np.random.uniform(2, 5, len(idx))
    df.loc[idx, "packet_loss_ratio"] = np.random.uniform(0.1, 0.5, len(idx))
    df.loc[idx, "port_utilization"] = np.random.uniform(70, 100, len(idx))

    df.loc[idx, "anomaly_label"] = 1

print("Label distribution:")
print(df["anomaly_label"].value_counts())

# ----------------------------
# HANDLE IMBALANCE SAFELY
# ----------------------------

normal = df[df["anomaly_label"] == 0]
anomaly = df[df["anomaly_label"] == 1]

if len(anomaly) == 0:
    print("⚠️ No anomalies found — injecting fallback anomalies")

    idx = np.random.choice(df.index, int(0.1 * len(df)), replace=False)
    df.loc[idx, "anomaly_label"] = 1

    anomaly = df[df["anomaly_label"] == 1]

# Balance
anomaly = anomaly.sample(len(normal), replace=True)
df_balanced = pd.concat([normal, anomaly])

print("Final shape:", df_balanced.shape)

# ----------------------------
# SAVE
# ----------------------------

df_balanced.to_csv("data/final_dataset.csv", index=False)

print("Final dataset saved successfully")