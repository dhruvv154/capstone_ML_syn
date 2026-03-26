import pandas as pd
import joblib
import time
import random
import math
import tkinter as tk
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from collections import deque
import numpy as np

# ----------------------------
# Configuration
# ----------------------------

BASE_THRESHOLD = 0.75
MAX_THRESHOLD = 0.9
CONSECUTIVE_LIMIT = 3
COOLDOWN_SECONDS = 10
DELAY_MS = 200
WINDOW_SIZE = 100
ANOMALY_RATE_TRIGGER = 0.4  # 40%

# ----------------------------
# Load Model & Data
# ----------------------------

model = joblib.load("sdn_model.pkl")
df = pd.read_csv("sdn_dataset.csv")

le = LabelEncoder()
df["protocol"] = le.fit_transform(df["protocol"])

X = df.drop(columns=["anomaly_label"])

# ----------------------------
# GUI Setup
# ----------------------------

root = tk.Tk()
root.title("AI-Based SDN Failure Detection System")
root.geometry("900x600")
root.configure(bg="#121212")

title = tk.Label(root, text="Intelligent SDN Monitoring Dashboard",
                 font=("Arial", 20, "bold"),
                 fg="white", bg="#121212")
title.pack(pady=15)

status_label = tk.Label(root, text="Status: Starting...",
                        font=("Arial", 14),
                        fg="white", bg="#121212")
status_label.pack()

prob_label = tk.Label(root, text="Anomaly Probability: 0.00",
                      font=("Arial", 12),
                      fg="white", bg="#121212")
prob_label.pack()

threshold_label = tk.Label(root, text=f"Current Threshold: {BASE_THRESHOLD}",
                           font=("Arial", 12),
                           fg="cyan", bg="#121212")
threshold_label.pack()

window_label = tk.Label(root, text="Window Anomaly Rate: 0%",
                        font=("Arial", 12),
                        fg="white", bg="#121212")
window_label.pack()

# Layout: left canvas for host visualization, right alert box
main_frame = tk.Frame(root, bg="#121212")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

canvas = tk.Canvas(main_frame, width=700, height=320, bg="#0b0b0b", highlightthickness=0)
canvas.pack(side="left", padx=10, pady=10)

alert_box = tk.Text(main_frame, width=40, bg="black",
                    fg="red", insertbackground="white")
alert_box.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# ----------------------------
# Host visualization setup
# ----------------------------
HOSTS = {
    "Host A": (100, 80),
    "Host B": (300, 80),
    "Host C": (500, 80),
    "Host D": (700, 80),
    "Switch": (400, 240)
}

NODE_RADIUS = 22
active_packets = []  # list of dicts: {id, x1,y1,x2,y2,progress,color}

def draw_nodes():
    canvas.delete("node")
    for name, (x, y) in HOSTS.items():
        canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS,
                           fill="#1f6feb" if name != "Switch" else "#444444",
                           outline="white", tags=("node",))
        canvas.create_text(x, y, text=name, fill="white", font=("Arial", 9), tags=("node",))

draw_nodes()

def spawn_packet(src_name, dst_name, color):
    sx, sy = HOSTS[src_name]
    dx, dy = HOSTS[dst_name]
    pkt = {
        "x1": sx,
        "y1": sy,
        "x2": dx,
        "y2": dy,
        "progress": 0.0,
        "color": color,
        "id": None
    }
    pkt["id"] = canvas.create_oval(sx-6, sy-6, sx+6, sy+6, fill=color, outline="")
    active_packets.append(pkt)

def animate_packets():
    to_remove = []
    for pkt in active_packets:
        pkt["progress"] += 0.06
        t = pkt["progress"]
        if t >= 1.0:
            canvas.delete(pkt["id"])
            to_remove.append(pkt)
            continue
        # simple linear interpolation
        nx = pkt["x1"] + (pkt["x2"] - pkt["x1"]) * t
        ny = pkt["y1"] + (pkt["y2"] - pkt["y1"]) * t
        canvas.coords(pkt["id"], nx-6, ny-6, nx+6, ny+6)
    for pkt in to_remove:
        active_packets.remove(pkt)
    root.after(40, animate_packets)

animate_packets()

# ----------------------------
# Monitoring Variables
# ----------------------------

current_index = 0
anomaly_counter = 0
last_alert_time = 0
current_threshold = BASE_THRESHOLD
prediction_window = deque(maxlen=WINDOW_SIZE)

# ----------------------------
# Monitoring Logic
# ----------------------------

def update_monitor():
    global current_index, anomaly_counter
    global last_alert_time, current_threshold

    if current_index >= len(X):
        current_index = 0

    sample = X.iloc[current_index:current_index+1]
    probability = model.predict_proba(sample)[0][1]

    # Add to rolling window
    prediction_window.append(1 if probability > current_threshold else 0)

    # ----------------------------
    # Spawn a visual packet between random hosts
    # color from green (low prob) to red (high prob)
    try:
        hosts_only = [h for h in HOSTS.keys() if h != "Switch"]
        src = random.choice(hosts_only)
        dst = random.choice([h for h in hosts_only if h != src])
        # color gradient
        r = int(min(255, 255 * (probability)))
        g = int(min(255, 255 * (1 - probability)))
        color = f"#{r:02x}{g:02x}00"
        spawn_packet(src, dst, color)
    except Exception:
        pass

    # Calculate rolling anomaly rate
    if len(prediction_window) > 0:
        anomaly_rate = np.mean(prediction_window)
    else:
        anomaly_rate = 0

    window_label.config(
        text=f"Window Anomaly Rate: {anomaly_rate*100:.1f}%"
    )

    # ----------------------------
    # Adaptive Threshold Logic
    # ----------------------------

    if anomaly_rate > 0.5:
        current_threshold = min(MAX_THRESHOLD, current_threshold + 0.01)
    elif anomaly_rate < 0.2:
        current_threshold = max(BASE_THRESHOLD, current_threshold - 0.01)

    threshold_label.config(
        text=f"Current Threshold: {current_threshold:.2f}"
    )

    # ----------------------------
    # Consecutive Alert Logic
    # ----------------------------

    if probability > current_threshold:
        anomaly_counter += 1
        status_label.config(text="Status: Suspicious Traffic", fg="orange")
    else:
        anomaly_counter = 0
        status_label.config(text="Status: Normal Traffic", fg="lightgreen")

    # Confirmed alert
    if anomaly_counter >= CONSECUTIVE_LIMIT:
        current_time = time.time()
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%H:%M:%S")
            alert_box.insert(
                "end",
                f"[{timestamp}] 🚨 Confirmed Failure | Prob: {probability:.2f}\n"
            )
            alert_box.see("end")
            last_alert_time = current_time

    # ----------------------------
    # Window-Level Alert
    # ----------------------------

    if anomaly_rate > ANOMALY_RATE_TRIGGER:
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_box.insert(
            "end",
            f"[{timestamp}] ⚠️ High Network Instability Detected (Window Rate: {anomaly_rate:.2f})\n"
        )
        alert_box.see("end")

    prob_label.config(
        text=f"Anomaly Probability: {probability:.2f}"
    )

    current_index += 1
    root.after(DELAY_MS, update_monitor)

update_monitor()
root.mainloop()
