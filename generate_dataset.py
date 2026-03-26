import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)

protocols = ["TCP", "UDP", "ICMP"]
rows = []
start_time = datetime.now()

for i in range(30000):

    timestamp = start_time + timedelta(seconds=i)
    proto = random.choice(protocols)

    regime = np.random.choice(
        ["normal", "congestion", "link_failure", "storm"],
        p=[0.65, 0.2, 0.1, 0.05]
    )

    if regime == "normal":
        packet_rate = abs(np.random.normal(25, 10))
        duration = np.random.exponential(60)
        packet_loss = np.random.uniform(0, 0.01)
        jitter = np.random.uniform(0, 3)

    elif regime == "congestion":
        packet_rate = np.random.uniform(150, 350)
        duration = np.random.uniform(20, 100)
        packet_loss = np.random.uniform(0.02, 0.08)
        jitter = np.random.uniform(5, 25)

    elif regime == "link_failure":
        packet_rate = np.random.uniform(5, 30)
        duration = np.random.uniform(1, 5)
        packet_loss = np.random.uniform(0.4, 0.8)
        jitter = np.random.uniform(15, 60)

    else:
        packet_rate = np.random.uniform(300, 600)
        duration = np.random.uniform(1, 10)
        packet_loss = np.random.uniform(0.05, 0.2)
        jitter = np.random.uniform(10, 40)

    packet_count = int(packet_rate * duration)
    byte_count = packet_count * np.random.uniform(400, 1300)
    byte_rate = byte_count / max(duration, 1)
    port_util = min(packet_rate / 600 * 100, 100)

    label = 0 if regime == "normal" else 1

    rows.append([
        proto,
        random.randint(1024, 65535),
        random.choice([80, 443, 53, 0]),
        packet_count,
        byte_count,
        duration,
        packet_rate,
        byte_rate,
        port_util,
        packet_loss,
        jitter,
        label
    ])

df = pd.DataFrame(rows, columns=[
    "protocol", "src_port", "dst_port",
    "packet_count", "byte_count",
    "flow_duration_sec",
    "packet_rate", "byte_rate",
    "port_utilization",
    "packet_loss_ratio",
    "jitter_ms",
    "anomaly_label"
])

df.to_csv("sdn_dataset.csv", index=False)
print("Dataset generated successfully ✅")
