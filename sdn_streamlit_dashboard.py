import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from collections import deque
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# CONFIGURATION
# ----------------------------

THRESHOLD = 0.75
WINDOW_SIZE = 100
REFRESH_INTERVAL = 150  # milliseconds

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(layout="wide")
st.title("🚀 AI-Based SDN Real-Time Monitoring Dashboard")

# ----------------------------
# LOAD MODEL & DATA (Cached)
# ----------------------------

@st.cache_resource
def load_model():
    return joblib.load("sdn_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("sdn_dataset.csv")
    le = LabelEncoder()
    df["protocol"] = le.fit_transform(df["protocol"])
    return df

model = load_model()
df = load_data()
X = df.drop(columns=["anomaly_label"])

# ----------------------------
# SESSION STATE INIT
# ----------------------------

if "index" not in st.session_state:
    st.session_state.index = 0

if "prob_history" not in st.session_state:
    st.session_state.prob_history = []

if "window" not in st.session_state:
    st.session_state.window = deque(maxlen=WINDOW_SIZE)

# ----------------------------
# GET CURRENT SAMPLE
# ----------------------------

i = st.session_state.index % len(X)
sample = X.iloc[i:i+1]

probability = model.predict_proba(sample)[0][1]
packet_rate = sample["packet_rate"].values[0]

st.session_state.prob_history.append(probability)
st.session_state.window.append(probability > THRESHOLD)

anomaly_rate = np.mean(st.session_state.window)

# ----------------------------
# RISK LOGIC
# ----------------------------

if probability < 0.5:
    status = "🟢 NORMAL"
    color = "green"
elif probability < THRESHOLD:
    status = "🟡 MEDIUM RISK"
    color = "orange"
else:
    status = "🔴 HIGH RISK"
    color = "red"

thickness = min(10, max(2, int(packet_rate / 50)))

# ----------------------------
# TOPOLOGY GRAPH FUNCTION
# ----------------------------

def create_topology(color, thickness):

    G = nx.Graph()
    G.add_node("S1")
    hosts = ["h1", "h2", "h3", "h4", "h5"]

    for h in hosts:
        G.add_edge("S1", h)

    pos = {
        "S1": (0, 1),
        "h1": (-2, 0),
        "h2": (-1, -0.5),
        "h3": (0, -1),
        "h4": (1, -0.5),
        "h5": (2, 0)
    }

    fig = go.Figure()

    # Draw edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=thickness, color=color),
            hoverinfo="none"
        ))

    # Draw nodes
    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=40, color="blue")
    ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=450
    )

    return fig

# ----------------------------
# LAYOUT
# ----------------------------

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_topology(color, thickness)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"### Status: {status}")
    st.metric("Anomaly Probability", f"{probability:.2f}")
    st.metric("Window Anomaly Rate", f"{anomaly_rate*100:.1f}%")
    st.metric("Packet Rate", f"{packet_rate:.2f}")

    st.subheader("Probability Trend")
    st.line_chart(st.session_state.prob_history)

# ----------------------------
# AUTO REFRESH
# ----------------------------

st.session_state.index += 1

st_autorefresh(interval=REFRESH_INTERVAL, key="refresh")
