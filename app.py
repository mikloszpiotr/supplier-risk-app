import streamlit as st
import pandas as pd
from src.preprocessing import engineer_features, scale_features
from src.predict import predict_risk, get_feature_importance, model

st.set_page_config(
    page_title="Supplier Risk Assessment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Upload & Options")
uploaded = st.sidebar.file_uploader("Upload supplier performance CSV", type=["csv"])

st.title("📊 Supplier Risk Assessment")

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.sidebar.markdown(f"**Records:** {raw_df.shape[0]} suppliers")
    features_df = engineer_features(raw_df)
    X_scaled, scaler = scale_features(features_df)

    # Predictions
    risk_levels = predict_risk(X_scaled)
    result_df = raw_df.assign(Risk_Level=risk_levels)

    # Layout
    col1, col2 = st.columns((2, 1))
    with col1:
        st.subheader("Risk Level Distribution")
        # Plot the actual label counts
        dist = result_df['Risk_Level'].value_counts()
        st.bar_chart(dist)

        st.subheader("Feature Importances")
        importances = get_feature_importance(features_df.columns)
        st.bar_chart(importances.sort_values(ascending=False))

        st.subheader("Risk Over Time")
        if 'order_date' in raw_df.columns:
            temp = raw_df.copy()
            temp['Risk_Level'] = risk_levels
            temp['order_date'] = pd.to_datetime(temp['order_date'])
            trend = temp.groupby([pd.Grouper(key='order_date', freq='M'), 'Risk_Level']) \
                        .size().unstack().fillna(0)
            st.line_chart(trend)
        else:
            st.info("Include 'order_date' for time trend visualization.")

    with col2:
        st.subheader("Key Metrics Summary")
        total = len(result_df)
        high = (result_df['Risk_Level'] == 'High').sum()
        med = (result_df['Risk_Level'] == 'Medium').sum()
        low = (result_df['Risk_Level'] == 'Low').sum()
        st.metric("Total Suppliers", total)
        st.metric("High Risk", high, f"{high/total:.1%}")
        st.metric("Medium Risk", med, f"{med/total:.1%}")
        st.metric("Low Risk", low, f"{low/total:.1%}")

    with st.expander("View Detailed Results"):
        st.dataframe(result_df)

else:
    st.info("Upload a CSV file to see supplier risk assessments and visualizations.")
