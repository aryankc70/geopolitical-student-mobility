"""Event-study / difference-in-differences estimation for a single cataloged shock,
using two-way fixed effects (country + year) and clustered standard errors.

IMPORTANT: outcome variables should be log-transformed before use. Countries
in this panel vary enormously in scale (China/India in the hundreds of
thousands vs. Nepal/Ukraine in the low thousands) - comparing raw levels
violates the parallel-trends assumption almost automatically due to scale
alone, not real divergent behavior. Log-transforming converts growth RATES
(comparable across scales) into the linear trends DiD actually requires.
"""
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS


def run_did(panel: pd.DataFrame, treated_countries: list, shock_year: int,
            outcome_col: str = "log_enrollment",
            window_start: int = None, window_end: int = None) -> dict:
    """
    Estimate the causal effect of a shock using two-way fixed effects DiD.

    Args:
        panel: long-format df with columns [country, panel_year, outcome_col]
        treated_countries: list of countries affected by the shock
        shock_year: the panel_year the shock begins (post = 1 for years >= this)
        outcome_col: which column to use as the outcome variable - should be
                     a log-transformed column given the scale-mismatch issue above
        window_start / window_end: restrict analysis to this year range
                                     (important for avoiding overlapping shocks)

    Returns:
        dict with the DiD coefficient (interpretable as an approximate
        percentage effect when outcome_col is log-transformed), standard
        error, p-value, confidence interval, and n_obs.
    """
    df = panel.copy()
    if window_start is not None:
        df = df[df["panel_year"] >= window_start]
    if window_end is not None:
        df = df[df["panel_year"] <= window_end]

    df["treated"] = df["country"].isin(treated_countries).astype(int)
    df["post"] = (df["panel_year"] >= shock_year).astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]

    df = df.dropna(subset=[outcome_col])
    df = df.set_index(["country", "panel_year"])

    model = PanelOLS.from_formula(
        f"{outcome_col} ~ treated_x_post + EntityEffects + TimeEffects",
        data=df,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)

    return {
        "coefficient": result.params.get("treated_x_post"),
        "std_error": result.std_errors.get("treated_x_post"),
        "p_value": result.pvalues.get("treated_x_post"),
        "conf_int": result.conf_int().loc["treated_x_post"].tolist() if "treated_x_post" in result.params.index else None,
        "n_obs": result.nobs,
        "result_summary": result,
    }


def pre_trend_check(panel: pd.DataFrame, treated_countries: list, shock_year: int,
                     outcome_col: str = "log_enrollment",
                     pre_window_start: int = None) -> dict:
    """
    Test whether treated and control groups had parallel trends BEFORE the shock.
    Runs the same DiD specification but with a fake 'post' cutoff at the
    midpoint of the pre-period - if treated_x_post is significant here,
    the parallel-trends assumption is violated and the main result is less credible.
    """
    df = panel[panel["panel_year"] < shock_year].copy()
    if pre_window_start is not None:
        df = df[df["panel_year"] >= pre_window_start]

    years = sorted(df["panel_year"].unique())
    if len(years) < 4:
        return {"warning": "Not enough pre-period years for a meaningful pre-trend check"}
    fake_cutoff = years[len(years) // 2]

    return run_did(df, treated_countries, fake_cutoff, outcome_col)