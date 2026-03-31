import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/final_dataset.csv")

print("Dataset shape:", df.shape)

# ----------------------------
# 1. CLASS DISTRIBUTION
# ----------------------------

plt.figure()
df["anomaly_label"].value_counts().plot(kind="bar")
plt.title("Class Distribution (Normal vs Anomaly)")
plt.xlabel("Class (0 = Normal, 1 = Anomaly)")
plt.ylabel("Count")
plt.savefig("viz_class_distribution.png")

# ----------------------------
# 2. FEATURE DISTRIBUTION
# ----------------------------

plt.figure()
df[df["anomaly_label"] == 0]["packet_rate"].hist(bins=50, alpha=0.5)
df[df["anomaly_label"] == 1]["packet_rate"].hist(bins=50, alpha=0.5)
plt.title("Packet Rate Distribution")
plt.xlabel("Packet Rate")
plt.ylabel("Frequency")
plt.legend(["Normal", "Anomaly"])
plt.savefig("viz_packet_rate.png")

# ----------------------------
# 3. PACKET LOSS VS TRAFFIC
# ----------------------------

plt.figure()
plt.scatter(df["packet_rate"], df["packet_loss_ratio"], alpha=0.3)
plt.title("Packet Rate vs Packet Loss")
plt.xlabel("Packet Rate")
plt.ylabel("Packet Loss Ratio")
plt.savefig("viz_scatter.png")

# ----------------------------
# 4. CORRELATION HEATMAP
# ----------------------------

plt.figure()
corr = df.corr(numeric_only=True)

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("viz_correlation.png")

print("All visualizations saved!")