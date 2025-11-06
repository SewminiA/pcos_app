import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained XGBoost model
# -----------------------------
model = joblib.load("best_pcos_model.pkl")

st.set_page_config(page_title="PCOS Risk Explanation", layout="wide")
st.title("🌸 PCOS Risk Explainability (XAI) Dashboard")

# -----------------------------
# Input Section
# -----------------------------
st.sidebar.header("Enter your data")
weight = st.sidebar.number_input("Weight (kg)", 40, 120, 60)
cycle = st.sidebar.slider("Cycle length (days)", 15, 45, 28)
fsh = st.sidebar.number_input("FSH (mIU/mL)", 1.0, 15.0, 6.0)
lh = st.sidebar.number_input("LH (mIU/mL)", 1.0, 25.0, 8.0)
fsh_lh = st.sidebar.number_input("FSH/LH Ratio", 0.1, 5.0, 0.8)
amh = st.sidebar.number_input("AMH (ng/mL)", 0.1, 15.0, 3.0)
prl = st.sidebar.number_input("Prolactin (ng/mL)", 2.0, 40.0, 10.0)
weight_gain = st.sidebar.selectbox("Weight Gain", [0, 1])
hair_growth = st.sidebar.selectbox("Hair Growth", [0, 1])
skin_dark = st.sidebar.selectbox("Skin Darkening", [0, 1])
follicle_L = st.sidebar.slider("Follicle Left", 0, 30, 10)
follicle_R = st.sidebar.slider("Follicle Right", 0, 30, 10)

# Create input dataframe
input_df = pd.DataFrame([[
    weight, cycle, fsh, lh, fsh_lh, amh, prl,
    weight_gain, hair_growth, skin_dark,
    follicle_L, follicle_R
]], columns=[
    'Weight', 'Cycle', 'FSH', 'LH', 'FSH/LH', 'AMH', 'PRL',
    'Weight_Gain', 'Hair_Growth', 'Skin_Dark',
    'Follicle_L', 'Follicle_R'
])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Analyze PCOS Risk"):
    pred_prob = model.predict_proba(input_df)[0][1]
    risk = "High Risk" if pred_prob > 0.5 else "Low Risk"

    st.subheader("🧬 PCOS Risk Result:")
    st.metric(label="Predicted Risk", value=risk, delta=f"{pred_prob*100:.2f}%")

    # -----------------------------
    # XAI Section (Explainability)
    # -----------------------------
    st.subheader("📊 Explainable AI (XAI) - SHAP Feature Impact")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_values[0]
        }).sort_values(by='SHAP Value', ascending=False)

        st.markdown("### 🔍 Feature Impact Explanation:")

        reasons = {
            "Weight": "Higher body weight increases insulin resistance and hormone imbalance.",
            "Cycle": "Irregular or long cycles indicate hormonal imbalance linked with PCOS.",
            "FSH": "Lower FSH disrupts follicle growth, raising PCOS risk.",
            "LH": "Higher LH levels increase androgen production.",
            "FSH/LH": "Low ratio indicates hormonal imbalance typical in PCOS.",
            "AMH": "High AMH levels are often found in PCOS patients.",
            "PRL": "Abnormal prolactin can affect menstrual cycles.",
            "Weight_Gain": "Weight gain contributes to insulin resistance.",
            "Hair_Growth": "Excessive hair growth suggests androgen excess.",
            "Skin_Dark": "Skin darkening is linked to insulin resistance.",
            "Follicle_L": "Higher follicle count on ovaries is a PCOS marker.",
            "Follicle_R": "High follicle count on both ovaries suggests PCOS."
        }

        explanation_list = []
        for feature, value in zip(feature_importance['Feature'], feature_importance['SHAP Value']):
            direction = "increases" if value > 0 else "decreases"
            reason = next((reasons[k] for k in reasons if k in feature),
                          "This feature influences PCOS risk through hormonal or physical factors.")
            text = f"💡 **{feature}** (`{value:.2f}`) **{direction}** PCOS risk — {reason}"
            st.markdown(text)
            explanation_list.append(text)

        # -----------------------------
        # Final Summary
        # -----------------------------
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

        # -----------------------------
        # SHAP Visualization
        # -----------------------------
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, max_display=5)
        st.pyplot(fig)

    except Exception as e:
        # -----------------------------
        # Fallback Feature Importance
        # -----------------------------
        st.warning("⚠️ SHAP visualization not supported in this environment.")
        st.info("Fallback mode: showing feature importance from model instead.")

        importance_df = pd.DataFrame({
            'Feature': model.get_booster().feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.markdown("### 🔝 Top Important Features (Fallback Mode):")
        for f in importance_df['Feature'].head(5):
            st.markdown(f"💡 **{f}** likely has strong influence on PCOS risk.")
        st.dataframe(importance_df.head(5))

