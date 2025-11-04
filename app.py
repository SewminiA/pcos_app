import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("best_xgb_model.pkl")

# App title
st.title("💊 PCOS Screening Tool")

st.write("Enter your clinical and hormonal data below to assess PCOS risk:")

# User input
weight = st.number_input("Weight (Kg)", 30, 150)
cycle = st.selectbox("Cycle (R/I)", [0, 1])  # 0=Regular, 1=Irregular
fsh = st.number_input("FSH (mIU/mL)", 0.1, 20.0)
lh = st.number_input("LH (mIU/mL)", 0.1, 20.0)
fsh_lh = st.number_input("FSH/LH", 0.1, 10.0)
ratio = st.number_input("Waist:Hip Ratio", 0.5, 2.0)
amh = st.number_input("AMH (ng/mL)", 0.1, 15.0)
prl = st.number_input("PRL (ng/mL)", 1.0, 100.0)
weight_gain = st.selectbox("Weight gain (Y/N)", [0, 1])
hair_growth = st.selectbox("Hair growth (Y/N)", [0, 1])
skin_dark = st.selectbox("Skin darkening (Y/N)", [0, 1])
follicle_L = st.number_input("Follicle No. (L)", 0, 50)
follicle_R = st.number_input("Follicle No. (R)", 0, 50)
avg_size = st.number_input("Avg. F size (L) (mm)", 0.0, 30.0)

# Arrange input in same order as model trained
features = np.array([[weight, cycle, fsh, lh, fsh_lh, ratio, amh, prl,
                      weight_gain, hair_growth, skin_dark,
                      follicle_L, follicle_R, avg_size]], dtype=float)

if st.button("Predict PCOS Risk"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of PCOS (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of PCOS (Probability: {prob:.2f})")

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



    








