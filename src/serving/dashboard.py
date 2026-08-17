"""Streamlit dashboard for the geopolitical shocks / student mobility project."""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.serving.inference import load_artifacts, predict_growth, get_causal_results

st.set_page_config(page_title="Student Mobility & Geopolitical Shocks", layout="wide")

load_artifacts()

st.title("Geopolitical Shocks & International Student Mobility")
st.caption(
    "Causal findings from difference-in-differences and interrupted time-series analysis, "
    "plus a forecasting model for near-term enrollment change. "
    "**Causal results and forecasts are separate analyses and should not be conflated.**"
)

tab1, tab2 = st.tabs(["Causal Findings", "Forecast Tool"])

with tab1:
    st.subheader("Tested Shocks and Their Estimated Effects")
    results = pd.DataFrame(get_causal_results())

    col1, col2 = st.columns([1, 3])
    with col1:
        shock_type_filter = st.multiselect(
            "Filter by shock type",
            options=results["shock_type"].unique().tolist(),
            default=results["shock_type"].unique().tolist(),
        )
        reliability_filter = st.multiselect(
            "Filter by reliability",
            options=results["reliability"].unique().tolist(),
            default=results["reliability"].unique().tolist(),
        )

    with col2:
        filtered = results[
            results["shock_type"].isin(shock_type_filter)
            & results["reliability"].isin(reliability_filter)
        ]
        st.dataframe(
            filtered[["shock_id", "shock_name", "treated_country", "effect_estimate",
                      "p_value", "significant", "reliability", "notes"]],
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "**Reading this table:** effect_estimate is a coefficient on log-enrollment, "
        "roughly interpretable as proportional change. 'Significant' = No means the effect "
        "could not be statistically distinguished from zero at this sample size - this does "
        "not necessarily mean there was no real effect, just that this data/method combination "
        "couldn't detect one reliably. See the full findings report for methodology and caveats."
    )

with tab2:
    st.subheader("Enrollment Growth Forecast")
    st.caption(
        "Predicts next-year enrollment growth given current conditions. "
        "This is a statistical forecast, not a causal claim about any specific shock."
    )

    col1, col2 = st.columns(2)
    with col1:
        prior_growth = st.slider("Prior 3-year average growth rate", -0.30, 0.30, 0.03, 0.01)
        cpi = st.number_input("CPI (annual average)", value=310.0, step=1.0)
    with col2:
        any_shock = st.selectbox("Any shock currently active?", options=[0, 1], index=0)
        num_shocks = st.number_input("Number of shocks currently active", min_value=0, max_value=10, value=0)

    if st.button("Predict", type="primary"):
        result = predict_growth({
            "prior_3yr_avg_growth": prior_growth,
            "any_shock_active": any_shock,
            "num_shocks_active": num_shocks,
            "cpi_annual_avg": cpi,
        })
        st.metric("Predicted next-year growth", f"{result['predicted_next_yr_growth']*100:.1f}%")
        st.write(result["interpretation"])