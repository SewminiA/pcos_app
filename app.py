reason = next((reasons[k] for k in reasons if k in feature), 
                      "This feature influences PCOS risk through hormonal or physical factors.")

        text = f"- **{feature}** (`{value:.2f}`) **{direction}** PCOS risk — {reason}"
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



    

