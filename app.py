import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="PCOS Screening Tool", page_icon="💊", layout="centered")

st.title("💊 PCOS Screening Tool (AI-powered)")
st.markdown("This tool uses a trained machine learning model and SHAP explainability to predict and interpret PCOS risk.")

# -------------------------------
# 2. Load Trained Model
# -------------------------------
import os


@st.cache_resource
def load_model():
    # Get absolute path to ensure Streamlit finds the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_xgb_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ 'best_xgb_model.pkl' not found! Please keep it in the same folder as app.py.")
        st.stop()

    model = joblib.load(model_path)
    return model

model = load_model()


# -------------------------------
# 3. Define Input Features
# -------------------------------
selected_features = [
    'Weight (Kg)', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
    'FSH/LH', 'Waist:Hip Ratio', 'AMH(ng/mL)', 'PRL(ng/mL)',
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
    'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)'
]

# -------------------------------
# 4. Collect User Input
# -------------------------------
st.header("🩺 Enter Your Medical Details")

def user_input():
    weight = st.number_input("Weight (Kg)", 30.0, 150.0, 60.0)
    cycle = st.number_input("Cycle (R/I)", 15.0, 60.0, 28.0)
    fsh = st.number_input("FSH (mIU/mL)", 0.1, 30.0, 6.0)
    lh = st.number_input("LH (mIU/mL)", 0.1, 30.0, 8.0)
    fsh_lh = fsh / lh if lh != 0 else 0
    whr = st.number_input("Waist:Hip Ratio", 0.5, 1.5, 0.8)
    amh = st.number_input("AMH (ng/mL)", 0.0, 20.0, 4.0)
    prl = st.number_input("PRL (ng/mL)", 0.0, 100.0, 20.0)
    wg = st.selectbox("Weight Gain (Y/N)", ["No", "Yes"])
    hg = st.selectbox("Hair Growth (Y/N)", ["No", "Yes"])
    sd = st.selectbox("Skin Darkening (Y/N)", ["No", "Yes"])
    fol_l = st.number_input("Follicle No. (L)", 0, 30, 10)
    fol_r = st.number_input("Follicle No. (R)", 0, 30, 10)
    avg_fsize = st.number_input("Avg. F size (L) (mm)", 0.0, 25.0, 10.0)

    data = {
        'Weight (Kg)': weight,
        'Cycle(R/I)': cycle,
        'FSH(mIU/mL)': fsh,
        'LH(mIU/mL)': lh,
        'FSH/LH': fsh_lh,
        'Waist:Hip Ratio': whr,
        'AMH(ng/mL)': amh,
        'PRL(ng/mL)': prl,
        'Weight gain(Y/N)': 1 if wg == "Yes" else 0,
        'hair growth(Y/N)': 1 if hg == "Yes" else 0,
        'Skin darkening (Y/N)': 1 if sd == "Yes" else 0,
        'Follicle No. (L)': fol_l,
        'Follicle No. (R)': fol_r,
        'Avg. F size (L) (mm)': avg_fsize
    }
    return pd.DataFrame([data])

input_df = user_input()

st.write("### Input Summary", input_df)

# -------------------------------
# 5. Predict Risk
# -------------------------------
if st.button("🔍 Predict PCOS Risk"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of PCOS (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of PCOS (Confidence: {(1-probability)*100:.2f}%)")

    # -------------------------------
    # 6. SHAP Explainability
    # -------------------------------
    st.subheader("🧠 Model Interpretation using SHAP")

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # Bar chart of top features
    st.write("### Most Influential Features for this Prediction:")
    shap_df = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP Value': shap_values.values[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    top_features = shap_df.head(5)
    st.table(top_features)

    # Optional SHAP plot
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=5, show=False)
    st.pyplot(fig)

