"""
app.py - Streamlit Dashboard for Predictive Maintenance
=========================================================
Provides a user-friendly UI to:
  • Input real-time sensor values
  • Call the FastAPI /predict endpoint
  • Display Failure / No Failure result with confidence gauge
  • Show optional feature importance visualisations
"""

import streamlit as st
import requests
import json
import os
from pathlib import Path

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown(
    "Enter real-time sensor readings to predict whether a machine is likely to **fail**."
)
st.divider()

# ─────────────────────────────────────────
# Sidebar – API health check
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API Base URL", value=API_URL)
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/", timeout=5)
            info = r.json()
            st.success(f"API Online ✅\nModel: **{info.get('model', 'N/A')}**")
        except Exception as e:
            st.error(f"API Offline ❌\n{e}")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This dashboard uses a machine learning model trained on the "
        "[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) "
        "to predict equipment failures."
    )

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
st.subheader("📥 Sensor Input")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L (Low)", "M (Medium)", "H (High)"],
        index=1,
        help="L=0, M=1, H=2 (quality variant)"
    )
    type_map = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}
    type_encoded = type_map[machine_type]

    air_temp = st.slider(
        "Air Temperature (K)",
        min_value=295.0, max_value=304.0,
        value=300.0, step=0.1,
        help="Typical range: 295–304 K"
    )

with col2:
    process_temp = st.slider(
        "Process Temperature (K)",
        min_value=305.0, max_value=314.0,
        value=310.0, step=0.1,
        help="Typical range: 305–314 K"
    )

    rot_speed = st.slider(
        "Rotational Speed (rpm)",
        min_value=1168, max_value=2886,
        value=1500, step=10,
        help="Typical range: 1168–2886 rpm"
    )

with col3:
    torque = st.slider(
        "Torque (Nm)",
        min_value=3.8, max_value=76.6,
        value=40.0, step=0.5,
        help="Typical range: 3.8–76.6 Nm"
    )

    tool_wear = st.slider(
        "Tool Wear (min)",
        min_value=0, max_value=253,
        value=100, step=1,
        help="Cumulative tool wear in minutes"
    )

# ─────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────
st.divider()

if st.button("🚀 Predict Machine Status", use_container_width=True, type="primary"):
    payload = {
        "Type": type_encoded,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    with st.spinner("Calling prediction API …"):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            # ── Result Display ────────────────────────────────
            st.divider()
            st.subheader("🔮 Prediction Result")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if result["prediction"] == 1:
                    st.error(f"### {result['label']}", icon="⚠️")
                    st.error(
                        f"**Failure Probability:** {result['probability_failure']*100:.1f}%"
                    )
                    st.markdown(
                        "> ⚠️ Immediate inspection recommended. "
                        "Schedule maintenance to prevent downtime."
                    )
                else:
                    st.success(f"### {result['label']}", icon="✅")
                    st.success(
                        f"**Failure Probability:** {result['probability_failure']*100:.1f}%"
                    )
                    st.markdown(
                        "> ✅ Machine operating within normal parameters."
                    )
                st.caption(f"Model used: **{result['model_used']}**")

            with res_col2:
                # Probability gauge using a progress bar
                prob_pct = result["probability_failure"]
                color = "🔴" if prob_pct > 0.5 else ("🟡" if prob_pct > 0.2 else "🟢")
                st.metric(
                    label="Failure Probability",
                    value=f"{prob_pct * 100:.1f}%",
                    delta=f"{color} {'HIGH RISK' if prob_pct > 0.5 else 'LOW RISK'}"
                )
                st.progress(min(prob_pct, 1.0))

            # ── Input Summary ─────────────────────────────────
            with st.expander("📋 Input Summary (sent to API)"):
                st.json(payload)
            with st.expander("📦 Raw API Response"):
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the API. "
                f"Make sure FastAPI is running at `{api_url}`.\n\n"
                "Run: `python main.py`"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─────────────────────────────────────────
# Feature Importance Visualisations
# ─────────────────────────────────────────
st.divider()
st.subheader("📊 Feature Importance Visualisations")
st.markdown("Generated after model training (`python train.py`).")

img_col1, img_col2 = st.columns(2)

for img_path, title, col in [
    ("feature_importance_random_forest.png", "Random Forest", img_col1),
    ("feature_importance_xgboost.png",        "XGBoost",        img_col2),
]:
    with col:
        if Path(img_path).exists():
            st.image(img_path, caption=f"Feature Importance – {title}", use_column_width=True)
        else:
            st.info(f"Run `python train.py` to generate the {title} chart.")

cm_col1, cm_col2 = st.columns(2)
for img_path, title, col in [
    ("confusion_matrix_random_forest.png", "Random Forest", cm_col1),
    ("confusion_matrix_xgboost.png",        "XGBoost",        cm_col2),
]:
    with col:
        if Path(img_path).exists():
            st.image(img_path, caption=f"Confusion Matrix – {title}", use_column_width=True)
        else:
            st.info(f"Run `python train.py` to generate the {title} confusion matrix.")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption(
    "Built with ❤️ using Streamlit · FastAPI · XGBoost · Scikit-learn  |  "
    "Dataset: AI4I 2020 Predictive Maintenance (UCI ML Repository)"
)
