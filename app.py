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
    st.subheader("🧠 Feature Contribution (SHAP Explanation)")

        # -------------------------------
    # 🧠 SHAP Explainability (Fixed)
    # -------------------------------
        # -------------------------------
    # 🧠 SHAP Explainability (Final Fixed Version)
    # -------------------------------
    import shap

    st.subheader("🧠 Feature Contribution (SHAP Explanation)")

    try:
        # ✅ Convert all values safely to numeric
        features_clean = np.array(features, dtype=float)

        # Create dataframe for SHAP
        input_df = pd.DataFrame(features_clean, columns=[
            'Weight (Kg)', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
            'FSH/LH', 'Waist:Hip Ratio', 'AMH(ng/mL)', 'PRL(ng/mL)',
            'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Follicle No. (L)',
            'Follicle No. (R)', 'Avg. F size (L) (mm)'
        ])

        # ✅ Use booster directly for SHAP TreeExplainer
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)

        # Compute SHAP values
        shap_values = explainer.shap_values(input_df)

        # Prepare feature importance
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_values[0],
            'Input Value': features_clean[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)

        # Display as table
        st.write("### Top Features Influencing This Prediction:")
        st.dataframe(feature_importance.head(5))

        # Human-readable text explanation
        explanation_text = []
        for _, row in feature_importance.head(5).iterrows():
            direction = "increased" if row['SHAP Value'] > 0 else "decreased"
            explanation_text.append(
                f"💡 **{row['Feature']}** (value: {row['Input Value']:.2f}) "
                f"{direction} PCOS risk."
            )

        st.markdown("### 🧩 Model Interpretation Summary:")
        for text in explanation_text:
            st.markdown(text)

        # Bar chart (instead of waterfall for single input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, max_display=5)
        st.pyplot(fig)

    except Exception as e:
        # Backup explanation when SHAP fails
        st.warning(f"⚠️ SHAP explanation not supported here. Showing fallback insights.")

        # Basic feature importance (model’s internal feature_importances_)
        importance_df = pd.DataFrame({
            'Feature': model.get_booster().feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.write("### Top Contributing Features (Approximation):")
        st.dataframe(importance_df.head(5))

        st.markdown("""
        _SHAP could not run in this environment (likely numeric format issue). 
        The above shows approximate feature importance based on model’s internal calculations._
        """)






