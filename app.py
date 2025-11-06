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

    # -------------------------------------------
    # SHAP EXPLANATION
    # -------------------------------------------
    st.markdown("### 🧠 Feature Impact Analysis")
    try:
        feature_names = [
            'Weight', 'Menstrual Cycle', 'FSH', 'LH', 'FSH/LH Ratio', 
            'Waist:Hip Ratio', 'AMH', 'Prolactin', 'Weight Gain', 
            'Hair Growth', 'Skin Darkening', 'Left Follicles', 
            'Right Follicles', 'Avg Follicle Size'
        ]
        input_df = pd.DataFrame(features, columns=feature_names)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values[0],
            'Value': features[0]
        }).sort_values('Impact', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        top = feature_imp.head(6).sort_values('Impact', ascending=True)
        colors = ['#51cf66' if v < 0 else '#ff6b6b' for v in top['Impact']]
        ax.barh(top['Feature'], top['Impact'], color=colors)
        ax.set_xlabel("Impact on PCOS Risk")
        ax.set_title("Top Contributing Factors", pad=10)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig)

        st.markdown("#### 💡 Key Influences")
        for _, row in top.iterrows():
            effect = "increased" if row["Impact"] > 0 else "reduced"
            st.markdown(f"- **{row['Feature']}** ({row['Value']:.2f}) — {effect} PCOS risk")

    except Exception as e:
        st.warning("Feature impact visualization not available in this environment.")

    # -------------------------------------------
    # RECOMMENDATIONS
    # -------------------------------------------
    st.markdown("### 💡 Recommended Next Steps")

    col1, col2 = st.columns(2)
    if pred == 1:
        with col1:
            st.markdown("""
            **🏥 Medical Recommendations**
            - Consult a gynecologist
            - Perform hormonal and ultrasound tests
            - Check glucose and insulin levels
            """)
        with col2:
            st.markdown("""
            **🌿 Lifestyle Recommendations**
            - Maintain healthy weight
            - Exercise regularly
            - Eat a balanced diet
            - Manage stress
            """)
    else:
        with col1:
            st.markdown("""
            **✅ Maintenance Tips**
            - Keep track of cycle regularity
            - Continue balanced nutrition
            - Regular physical activity
            """)
        with col2:
            st.markdown("""
            **🔍 Monitoring**
            - Watch for new symptoms
            - Annual metabolic check
            """)
st.markdown("---")
st.caption("⚠️ This tool is for screening only and does not replace a professional medical diagnosis.")
