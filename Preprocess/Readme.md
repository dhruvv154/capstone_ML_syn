🚀 AI-Based SDN Anomaly Detection System

This project implements a real-time AI-based system for detecting anomalies in Software Defined Networks (SDN) using machine learning (XGBoost).

📁 Project Structure
Preprocess/
│
├── data/
│   ├── sdn_dataset.csv          # Synthetic dataset
│   ├── dataset_sdn.csv          # Telemetry dataset
│   └── final_dataset.csv        # Processed dataset
│
├── merge_and_preprocess.py      # Data merging + preprocessing
├── train_model.py               # Model training
├── live_system.py               # Real-time simulation
├── visualize.py                 # Dataset visualization
├── final_model.pkl              # Trained model
⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
2️⃣ Install Dependencies
pip install pandas numpy scikit-learn xgboost matplotlib joblib

👉 For Mac (required for XGBoost):

brew install libomp
▶️ How to Run
Step 1: Merge & Preprocess Data
python merge_and_preprocess.py

👉 Generates:

data/final_dataset.csv
Step 2: Train Model
python train_model.py

👉 Outputs:

models/final_model.pkl
feature_importance.png
Step 3: Run Real-Time Monitoring
python live_system.py

👉 Displays:

Real-time predictions
Status (NORMAL / DEGRADED / CRITICAL)
Anomaly probability
Explanation
Step 4: Visualize Dataset (Optional)
python visualize.py

👉 Generates:

Class distribution graph
Feature distribution plots
Correlation heatmap
🧠 Features
Real-time anomaly detection
Explainable predictions
Feature importance analysis
Hybrid dataset (synthetic + telemetry)
Low-latency monitoring system
