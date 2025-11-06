import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_xgb_model.pkl")

# -----------------------------
# App layout & title
# -----------------------------
st.set_page_config(page_title="PCOS Screening Tool", page_icon="💊", layout="centered")
st.title("💊 PCOS Screening Tool")
st.markdown("Provide your clinical details below to assess potential PCOS risk.")

# -----------------------------
# Input form
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Clinical Data")
    weight = st.number_input("Weight (Kg)", 30, 150, 60)
    cycle = st.selectbox("Menstrual Cycle", ["Regular", "Irregular"])
    ratio = st.number_input("Waist:Hip Ratio", 0.5, 2.0, 0.8)
    weight_gain = st.selectbox("Weight Gain", ["No", "Yes"])
    hair_growth = st.selectbox("Excess Hair Growth", ["No", "Yes"])
    skin_dark = st.selectbox("Skin Darkening", ["No", "Yes"])

with col2:
    st.subheader("Hormonal & Ultrasound Data")
    fsh = st.number_input("FSH (mIU/mL)", 0.1, 20.0, 5.0)
    lh = st.number_input("LH (mIU/mL)", 0.1, 20.0, 5.0)
    amh = st.number_input("AMH (ng/mL)", 0.1, 15.0, 3.0)
    prl = st.number_input("Prolactin (ng/mL)", 1.0, 100.0, 15.0)
    follicle_L = st.number_input("Left Follicle Count", 0, 50, 5)
    follicle_R = st.number_input("Right Follicle Count", 0, 50, 5)

# -----------------------------
# Data preparation
# -----------------------------
cycle = 1 if cycle == "Irregular" else 0
weight_gain = 1 if weight_gain == "Yes" else 0
hair_growth = 1 if hair_growth == "Yes" else 0
skin_dark = 1 if skin_dark == "Yes" else 0
fsh_lh = fsh / lh if lh > 0 else 0
avg_size = 0.0  # not used in dataset

features = np.array([[weight, cycle, fsh, lh, fsh_lh, ratio, amh, prl,
                      weight_gain, hair_growth, skin_dark,
                      follicle_L, follicle_R, avg_size]], dtype=float)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict PCOS Risk"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("📈 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of PCOS (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of PCOS (Probability: {prob:.2f})")

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader("🧠 Feature Contribution (Explainability)")
    try:
        input_df = pd.DataFrame(features, columns=[
            'Weight (Kg)', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
            'FSH/LH', 'Waist:Hip Ratio', 'AMH(ng/mL)', 'PRL(ng/mL)',
            'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Follicle No. (L)',
            'Follicle No. (R)', 'Avg. F size (L) (mm)'
        ])

        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(input_df)

        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_values[0],
            'Input Value': features[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)

        st.write("### 🔝 Top Features Influencing Prediction:")
        st.dataframe(feature_importance.head(5))

        # -------------------------
        # Explanation Phrases
        # -------------------------
        st.markdown("### 🧬 PCOS Risk Interpretation Summary:")

        explanation_list = []
        for _, row in feature_importance.head(5).iterrows():
            direction = "increased" if row['SHAP Value'] > 0 else "decreased"
            feature = row['Feature']
            value = row['Input Value']

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
        # Final Summary
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
        try:
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, max_display=5)
            st.pyplot(fig)
        except:
            st.info("📊 Feature impact visualization not supported. Showing top feature importance instead.")
            st.dataframe(feature_importance[['Feature', 'SHAP Value']].head(5))

    except Exception as e:
        st.warning("⚠️ SHAP explanation not supported in this environment.")
        st.info("Using fallback explanation based on feature importance.")

        importance_df = pd.DataFrame({
            'Feature': model.get_booster().feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.write("### 🔝 Top Important Features:")
        for f in importance_df['Feature'].head(5):
            st.markdown(f"💡 **{f}** likely influences PCOS risk based on model training.")
        st.dataframe(importance_df.head(5))

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.markdown("---")
    st.subheader("💡 Next Steps")
    if prediction == 1:
        st.write("""
        - Schedule appointment with a gynecologist  
        - Consider hormonal and ultrasound confirmation  
        - Maintain a healthy weight and balanced diet  
        - Regular exercise and stress management  
        """)
    else:
        st.write("""
        - Continue routine checkups  
        - Maintain a healthy lifestyle  
        - Monitor any menstrual or hormonal changes  
        """)

st.markdown("---")
st.caption("Note: This tool is for preliminary screening. Always consult healthcare professionals for a confirmed diagnosis.")
