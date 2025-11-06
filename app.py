import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="PCOS Pro - Risk Assessment",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")

model = load_model()

# ------------------- CUSTOM STYLES -------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .section-title {
        font-size: 1.4rem;
        color: #2d3748;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.5rem;">🔬 PCOS Pro</h1>
    <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Advanced Polycystic Ovary Syndrome Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# ------------------- INPUT SECTION -------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card"><h3>📋 Patient Information</h3>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👤 Clinical", "🔬 Hormonal", "📊 Ultrasound"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=65)
            cycle = st.selectbox("Menstrual Cycle", ["Regular", "Irregular"])
        with colB:
            ratio = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.2, value=0.85, step=0.01)
            weight_gain = st.radio("Weight Gain", ["No", "Yes"], horizontal=True)

    with tab2:
        colA, colB = st.columns(2)
        with colA:
            fsh = st.slider("FSH (mIU/mL)", 0.1, 20.0, 6.0, 0.1)
            lh = st.slider("LH (mIU/mL)", 0.1, 20.0, 8.0, 0.1)
        with colB:
            amh = st.slider("AMH (ng/mL)", 0.1, 15.0, 4.5, 0.1)
            prl = st.slider("Prolactin (ng/mL)", 1.0, 100.0, 18.0, 0.1)

    with tab3:
        colA, colB = st.columns(2)
        with colA:
            follicle_L = st.slider("Left Follicle Count", 0, 50, 12)
            follicle_R = st.slider("Right Follicle Count", 0, 50, 10)
        with colB:
            hair_growth = st.radio("Hair Growth", ["No", "Yes"], horizontal=True)
            skin_dark = st.radio("Skin Darkening", ["No", "Yes"], horizontal=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- INSIGHTS SECTION -------------------
with col2:
    st.markdown('<div class="card"><h3>📈 Quick Insights</h3>', unsafe_allow_html=True)

    fsh_lh_ratio = fsh / lh if lh > 0 else 0
    total_follicles = follicle_L + follicle_R

    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.metric("FSH/LH Ratio", f"{fsh_lh_ratio:.2f}")
        st.metric("Total Follicles", total_follicles)
    with insight_col2:
        bmi_category = "Normal" if weight < 80 else "Elevated"
        st.metric("Weight Status", bmi_category)
        st.metric("Cycle Pattern", cycle)

    st.markdown("**Risk Indicators:**")
    risk_factors = []
    if cycle == "Irregular": risk_factors.append("Irregular cycles")
    if hair_growth == "Yes": risk_factors.append("Hair growth")
    if total_follicles > 20: risk_factors.append("High follicle count")
    if skin_dark == "Yes": risk_factors.append("Skin darkening")
    if weight_gain == "Yes": risk_factors.append("Weight gain")

    for factor in (risk_factors or ["No major risk factors detected"]):
        st.write(f"• {factor}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- FEATURE PREPARATION -------------------
cycle_num = 1 if cycle == "Irregular" else 0
weight_gain_num = 1 if weight_gain == "Yes" else 0
hair_growth_num = 1 if hair_growth == "Yes" else 0
skin_dark_num = 1 if skin_dark == "Yes" else 0
fsh_lh_val = fsh / lh if lh > 0 else 0
avg_size = 0.0

features = np.array([[weight, cycle_num, fsh, lh, fsh_lh_val, ratio, amh, prl,
                     weight_gain_num, hair_growth_num, skin_dark_num,
                     follicle_L, follicle_R, avg_size]], dtype=float)

# ------------------- ASSESSMENT BUTTON -------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if st.button("🎯 Start Comprehensive Assessment", use_container_width=True):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            <h2>⚠️ High PCOS Risk Detected</h2>
            <p style="font-size:1.4rem;">Assessment Score: <strong>{probability:.1%}</strong></p>
            <p>Immediate medical consultation recommended.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h2>✅ Low PCOS Risk</h2>
            <p style="font-size:1.4rem;">Assessment Score: <strong>{probability:.1%}</strong></p>
            <p>Continue with regular health monitoring.</p>
        </div>""", unsafe_allow_html=True)

    # ------------------- FEATURE IMPACT / FALLBACK -------------------
    st.markdown('<div class="card"><h3 class="section-title">📊 Detailed Analysis</h3>', unsafe_allow_html=True)

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

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values[0],
            'Value': features[0]
        }).sort_values('Impact', key=abs, ascending=False)

        st.subheader("🎯 Key Contributing Factors")
        top_factors = feature_importance.head(8)
        colA, colB = st.columns(2)

        for idx, row in top_factors.iterrows():
            color = "#c92a2a" if row["Impact"] > 0 else "#2b8a3e"
            icon = "📈" if row["Impact"] > 0 else "📉"
            html = f"""
            <div style="background:#fff; border-left:5px solid {color}; 
                        border-radius:10px; padding:0.75rem; margin-bottom:0.5rem;">
                <strong>{icon} {row['Feature']}</strong>
                <div style="float:right; color:{color}; font-weight:bold;">{row['Impact']:.3f}</div>
                <div style="clear:both; color:#666; font-size:0.9rem;">Value: {row['Value']:.2f}</div>
            </div>
            """
            if idx % 2 == 0:
                colA.markdown(html, unsafe_allow_html=True)
            else:
                colB.markdown(html, unsafe_allow_html=True)

        try:
            st.markdown("### 📈 Impact Visualization")
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_data = feature_importance.head(6).sort_values('Impact', ascending=True)
            colors = ['#51cf66' if x < 0 else '#ff6b6b' for x in plot_data['Impact']]
            ax.barh(plot_data['Feature'], plot_data['Impact'], color=colors, alpha=0.85)
            ax.set_xlabel("Impact on PCOS Risk", fontsize=12)
            ax.set_title("Feature Impact Analysis", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        except:
            st.warning("📊 Feature impact visualization not supported in this environment.")
            st.dataframe(
                feature_importance[['Feature', 'Impact', 'Value']].head(10)
                .style.format({'Impact': '{:.3f}', 'Value': '{:.2f}'})
            )

    except Exception as e:
        st.error("⚠️ Advanced analysis unavailable.")
        st.info("Basic risk assessment is still accurate. Please consult your doctor.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------- RECOMMENDATIONS -------------------
    st.markdown('<div class="card"><h3 class="section-title">💡 Recommended Action Plan</h3>', unsafe_allow_html=True)
    if prediction == 1:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            **🏥 Immediate Medical Steps**
            - Schedule appointment with gynecologist  
            - Complete hormonal panel & pelvic ultrasound  
            - Screen for glucose, insulin & thyroid  
            """)
        with colB:
            st.markdown("""
            **🌱 Lifestyle Management**
            - PCOS-friendly diet & exercise plan  
            - Weight management program  
            - Stress reduction & sleep improvement  
            """)
    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            **🛡️ Maintenance & Prevention**
            - Annual gynecological checkups  
            - Balanced diet & regular physical activity  
            """)
        with colB:
            st.markdown("""
            **🔍 Ongoing Monitoring**
            - Track menstrual cycle  
            - Watch for new symptoms  
            """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p><strong>Medical Disclaimer:</strong> This tool is for screening only and does not replace professional medical diagnosis.</p>
    <p style="font-size: 0.9rem;">Diagnosis must be confirmed by a healthcare provider using clinical criteria.</p>
    <p style="font-size: 0.8rem;">© 2024 PCOS Pro Assessment Tool</p>
</div>
""", unsafe_allow_html=True)
