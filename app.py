import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
st.set_page_config(
    page_title="PCOS Pro - Risk Assessment",
    page_icon="💊",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")

model = load_model()

# -------------------------------------------
# GLOBAL STYLING
# -------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f9fafc;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 {
    color: #2d3748;
}
.card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}
.metric {
    text-align: center;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 1rem;
}
.result-high {
    background: linear-gradient(135deg, #ff6b6b, #fa5252);
    color: white;
    border-radius: 15px;
    text-align: center;
    padding: 2rem;
}
.result-low {
    background: linear-gradient(135deg, #51cf66, #40c057);
    color: white;
    border-radius: 15px;
    text-align: center;
    padding: 2rem;
}
.stButton button {
    background: #4c6ef5;
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.3s;
}
.stButton button:hover {
    background: #364fc7;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.markdown("""
<div style='text-align:center; margin-bottom: 2rem;'>
    <h1>💊 PCOS Pro</h1>
    <p style='color:#718096; font-size:1.1rem;'>Comprehensive Polycystic Ovary Syndrome Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------
# INPUT SECTION
# -------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Clinical Data")
    weight = st.number_input("Weight (kg)", 30, 150, 65)
    cycle = st.selectbox("Menstrual Cycle", ["Regular", "Irregular"])
    ratio = st.number_input("Waist:Hip Ratio", 0.5, 1.2, 0.85)
    weight_gain = st.radio("Weight Gain", ["No", "Yes"], horizontal=True)
    hair_growth = st.radio("Hair Growth", ["No", "Yes"], horizontal=True)
    skin_dark = st.radio("Skin Darkening", ["No", "Yes"], horizontal=True)

with col2:
    st.markdown("### 🔬 Hormonal & Ultrasound Data")
    fsh = st.slider("FSH (mIU/mL)", 0.1, 20.0, 6.0, 0.1)
    lh = st.slider("LH (mIU/mL)", 0.1, 20.0, 8.0, 0.1)
    amh = st.slider("AMH (ng/mL)", 0.1, 15.0, 4.5, 0.1)
    prl = st.slider("Prolactin (ng/mL)", 1.0, 100.0, 18.0, 0.1)
    follicle_L = st.slider("Left Follicle Count", 0, 50, 12)
    follicle_R = st.slider("Right Follicle Count", 0, 50, 10)

# Derived values
cycle_num = 1 if cycle == "Irregular" else 0
weight_gain_num = 1 if weight_gain == "Yes" else 0
hair_growth_num = 1 if hair_growth == "Yes" else 0
skin_dark_num = 1 if skin_dark == "Yes" else 0
fsh_lh = fsh / lh if lh > 0 else 0
avg_size = 0.0

features = np.array([[weight, cycle_num, fsh, lh, fsh_lh, ratio, amh, prl,
                     weight_gain_num, hair_growth_num, skin_dark_num,
                     follicle_L, follicle_R, avg_size]], dtype=float)

st.markdown("---")

# -------------------------------------------
# PREDICTION
# -------------------------------------------
if st.button("🎯 Assess PCOS Risk", use_container_width=True):

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.markdown("## 🩺 Assessment Result")

    if pred == 1:
        st.markdown(f"""
        <div class="result-high">
            <h2>⚠️ High PCOS Risk Detected</h2>
            <p style='font-size:1.3rem;'>Risk Probability: <b>{prob:.1%}</b></p>
            <p>Immediate consultation with a gynecologist is recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <h2>✅ Low PCOS Risk</h2>
            <p style='font-size:1.3rem;'>Risk Probability: <b>{prob:.1%}</b></p>
            <p>Continue regular health monitoring and a balanced lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)


    # -------------------------------
    # 🧠 SHAP Explainability (Improved)
    # -------------------------------
# 🧠 SHAP Explainability with Human-friendly Medical Summary
# -------------------------------
# -------------------------------
# 🧠 SHAP Explainability (with readable medical phrases)
# -------------------------------
import shap

st.subheader("🧠 Feature Contribution (SHAP Explanation)")

try:
    # Convert to numeric safely
    features_clean = np.array(features, dtype=float)
    input_df = pd.DataFrame(features_clean, columns=[
        'Weight (Kg)', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
        'FSH/LH', 'Waist:Hip Ratio', 'AMH(ng/mL)', 'PRL(ng/mL)',
        'Weight gain(Y/N)', 'hair growth(Y/N)',
        'Skin darkening (Y/N)', 'Follicle No. (L)',
        'Follicle No. (R)', 'Avg. F size (L) (mm)'
    ])

    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(input_df)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP Value': shap_values[0],
        'Input Value': features_clean[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    st.write("### 🔝 Top Features Influencing Prediction:")
    st.dataframe(feature_importance.head(5))

    # -------------------------
    # Generate Explanation Phrases
    # -------------------------
    st.markdown("### 🧩 PCOS Risk Interpretation Summary:")

    explanation_list = []
    for _, row in feature_importance.head(5).iterrows():
        direction = "increased" if row['SHAP Value'] > 0 else "decreased"
        feature = row['Feature']
        value = row['Input Value']

        # Domain-based reason mapping
        reasons = {
            "AMH": "High AMH levels are linked with greater ovarian activity.",
            "LH": "Elevated LH indicates hormonal imbalance in PCOS.",
            "FSH": "Low FSH can affect ovulation and follicle growth.",
            "FSH/LH": "A low FSH/LH ratio is a typical hormonal indicator of PCOS.",
            "Weight": "Higher weight influences insulin resistance and hormones.",
            "Waist": "A high waist-to-hip ratio suggests fat distribution linked with PCOS.",
            "PRL": "Prolactin imbalance can affect reproductive hormones.",
            "hair": "Increased hair growth reflects androgen excess (common in PCOS).",
            "Skin": "Skin darkening suggests insulin-related hormonal imbalance.",
            "Follicle": "Follicle count changes reflect altered ovarian function.",
            "F size": "Irregular follicle size affects ovulation patterns."
        }

        reason = next((reasons[k] for k in reasons if k in feature), 
                      "This feature influences PCOS risk through hormonal or physical factors.")

        text = f"💡 **{feature}** (`{value:.2f}`) **{direction}** PCOS risk — {reason}"
        st.markdown(text)
        explanation_list.append(text)

    # -------------------------
    # Final Summary Sentence
    # -------------------------
    positives = [f for f, v in zip(feature_importance['Feature'], feature_importance['SHAP Value']) if v > 0]
    negatives = [f for f, v in zip(feature_importance['Feature'], feature_importance['SHAP Value']) if v < 0]

    summary = "Based on your inputs, "
    if positives:
        summary += f"**{', '.join(positives[:2])}** contributed most to the *increased PCOS likelihood*"
    if negatives:
        summary += f", while **{', '.join(negatives[:2])}** helped *reduce* the risk."
    summary += " 🩺"

    st.markdown("### 🧠 Overall Summary:")
    st.success(summary)

    # -------------------------
    # SHAP Visualization
    # -------------------------
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, max_display=5)
    st.pyplot(fig)

except Exception as e:
    st.warning("⚠️ SHAP explanation not supported in this environment.")
    st.info("Using fallback explanation based on feature importance.")
    importance_df = pd.DataFrame({
        'Feature': model.get_booster().feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Fallback: generate phrases anyway
    st.write("### 🔝 Top Important Features:")
    for f in importance_df['Feature'].head(5):
        st.markdown(f"💡 **{f}** likely influences PCOS risk based on model training.")

    st.dataframe(importance_df.head(5))



    











