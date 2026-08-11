import sys
import os

# 1. Resolve project root path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from src.predict import FraudPredictor

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1e222d;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
    /* Section Divider Line */
    hr {
        margin: 1.5rem 0;
        border-color: #2e3440;
    }
    </style>
""", unsafe_allow_html=True)

# Resource Loading
@st.cache_resource
def load_resources():
    model_path = os.path.join(ROOT_DIR, "saved_models", "pytorch_smote_model.pth")
    scaler_path = os.path.join(ROOT_DIR, "saved_models", "scaler.joblib")
    test_data_path = os.path.join(ROOT_DIR, "saved_models", "test_data.joblib")
    
    predictor = FraudPredictor(model_path=model_path, scaler_path=scaler_path)
    X_test_scaled, y_test = joblib.load(test_data_path)
    
    return predictor, X_test_scaled, y_test

predictor, X_test_scaled, y_test = load_resources()

# Sidebar Configuration
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Decision Threshold", 
    min_value=0.05, 
    max_value=0.95, 
    value=0.50, 
    step=0.05,
    help="Adjust threshold to balance Precision vs. Recall based on business risk tolerance."
)

st.sidebar.info(
    f"**Current Threshold:** `{threshold:.2f}`\n\n"
    "• **Higher Threshold:** Reduces False Positives (Fewer genuine transactions flagged).\n"
    "• **Lower Threshold:** Higher Recall (Catches more fraud, but increases False Alerts)."
)

# Main Header
st.title("💳 Real-Time Credit Card Fraud Detection Engine")
st.caption("Production-Ready PyTorch Deep Learning Model Trained with SMOTE Resampling")
st.markdown("---")

# Tabbed Interface
tab1, tab2 = st.tabs(["🧪 Interactive Simulator", "📊 Model Performance Artifacts"])

# TAB 1: Live Simulator
with tab1:
    st.subheader("Simulate Incoming Real-Time Transactions")
    st.write("Sample test set transactions to test PyTorch model inference and decision logic.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Sample Genuine Transaction", use_container_width=True):
            legit_indices = np.where(y_test == 0)[0]
            idx = np.random.choice(legit_indices)
            st.session_state["sample"] = X_test_scaled[idx]
            st.session_state["true_label"] = 0
            
    with col2:
        if st.button("🔴 Sample Fraudulent Transaction", use_container_width=True):
            fraud_indices = np.where(y_test == 1)[0]
            idx = np.random.choice(fraud_indices)
            st.session_state["sample"] = X_test_scaled[idx]
            st.session_state["true_label"] = 1

    if "sample" in st.session_state:
        sample = st.session_state["sample"]
        true_label = st.session_state["true_label"]
        
        # Inference calculation
        import torch
        tensor_in = torch.tensor(sample.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            prob = float(torch.sigmoid(predictor.model(tensor_in)).squeeze())
        
        is_flagged = prob >= threshold
        
        st.markdown("### Inference Results")
        m1, m2, m3 = st.columns(3)
        
        m1.metric(
            label="Actual Ground Truth", 
            value="🚨 Fraud" if true_label == 1 else "✅ Genuine"
        )
        
        m2.metric(
            label="Predicted Fraud Probability", 
            value=f"{prob * 100:.2f}%"
        )
        
        if is_flagged:
            m3.metric(
                label="Action Required", 
                value="🚨 FLAGGED", 
                delta="High Risk Warning", 
                delta_color="inverse"
            )
        else:
            m3.metric(
                label="Action Required", 
                value="✅ APPROVED", 
                delta="Low Risk", 
                delta_color="normal"
            )

# TAB 2: Model Artifacts & Evaluation
with tab2:
    st.subheader("Evaluation Artifacts & Diagnostic Metrics")
    st.caption("Visual proof of model stability and class-imbalance treatment.")
    
    col_a, col_b = st.columns(2, gap="large")
    
    with col_a:
        st.markdown("#### 1. Precision-Recall Curve Comparison")
        st.write("Demonstrates SMOTE's improvement over class-weighting alone.")
        pr_path = os.path.join(ROOT_DIR, "saved_models", "pr_curve_comparison.png")
        if os.path.exists(pr_path):
            st.image(Image.open(pr_path), use_container_width=True)

    with col_b:
        st.markdown("#### 2. Confusion Matrices")
        st.write("Visualizing the reduction in False Positives under SMOTE.")
        cm_path = os.path.join(ROOT_DIR, "saved_models", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Threshold Trade-Off Analysis (SMOTE)")
    st.write("Shows Precision and Recall intersection across all possible classification cutoffs.")
    thresh_path = os.path.join(ROOT_DIR, "saved_models", "threshold_tradeoff_smote.png")
    if os.path.exists(thresh_path):
        st.image(Image.open(thresh_path), use_container_width=True)