"""
QuantumShield – Streamlit Real-Time Monitoring Dashboard
Run: streamlit run deployment/dashboard.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt

API_URL = os.environ.get("QUANTUMSHIELD_API", "http://localhost:5000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantumShield IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ QuantumShield — Intrusion Detection Dashboard")
st.caption("Hybrid Quantum-Classical IDS with Explainable AI | NCI Dublin MSc AI Practicum")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 10)
show_shap    = st.sidebar.checkbox("Show SHAP explanations", value=True)
sample_idx   = st.sidebar.number_input("SHAP sample index", min_value=0, value=0, step=1)

# ---------------------------------------------------------------------------
# Live prediction demo
# ---------------------------------------------------------------------------

st.header("🔍 Live Flow Classification")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Features (8 PCA components)")
    feature_vals = [st.number_input(f"PC{i+1}", value=float(np.random.randn()), key=f"pc{i}")
                    for i in range(8)]

with col2:
    if st.button("Classify Flow", type="primary"):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": feature_vals},
                timeout=10,
            )
            result = resp.json()
            label = result.get("prediction", "N/A")
            conf  = result.get("confidence", 0.0)
            color = "🔴 ATTACK" if label == "ATTACK" else "🟢 BENIGN"
            st.metric("Prediction", color)
            st.metric("Confidence", f"{conf:.2%}")
            st.json({
                "Benign prob":  f"{result['probabilities']['benign']:.4f}",
                "Attack prob":  f"{result['probabilities']['attack']:.4f}",
            })
        except Exception as e:
            st.error(f"API error: {e}")
            st.info("Make sure the Flask API is running: python deployment/api.py")

# ---------------------------------------------------------------------------
# SHAP explanation panel
# ---------------------------------------------------------------------------

if show_shap:
    st.header("🔎 SHAP Explanation")
    shap_plot_path = "results/shap_plots/shap_summary.png"
    waterfall_path = f"results/shap_plots/shap_waterfall_sample{int(sample_idx)}.png"

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(shap_plot_path):
            st.image(shap_plot_path, caption="Global SHAP Feature Importance", use_column_width=True)
        else:
            st.warning("SHAP summary not found. Run: python -m explainability.shap_hybrid")
    with c2:
        try:
            resp = requests.get(f"{API_URL}/shap/{int(sample_idx)}", timeout=5)
            if resp.status_code == 200:
                shap_data = resp.json()["shap_attack_class"]
                st.subheader(f"Sample {sample_idx} — Feature Contributions")
                shap_df = pd.DataFrame.from_dict(shap_data, orient="index", columns=["SHAP value"])
                shap_df = shap_df.sort_values("SHAP value", ascending=False)
                st.bar_chart(shap_df)
            else:
                st.info(resp.json().get("error", "SHAP not available"))
        except Exception:
            if os.path.exists(waterfall_path):
                st.image(waterfall_path, caption=f"Waterfall – sample {sample_idx}", use_column_width=True)

# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------

st.header("📊 Model Comparison")
comp_path = "results/model_comparison.png"
if os.path.exists(comp_path):
    st.image(comp_path, use_column_width=True)
else:
    st.info("Run evaluation/hypotheses.py to generate the comparison chart.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

st.caption(f"Auto-refreshing every {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()
