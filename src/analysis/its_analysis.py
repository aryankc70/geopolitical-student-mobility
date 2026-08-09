"""Interrupted time-series (segmented regression) analysis for shocks that
affect all countries simultaneously, where no valid control group exists
for standard difference-in-differences.

Model (standard segmented-regression ITS specification):
    y_t = b0 + b1*time + b2*post + b3*(time_since_shock * post) + error

    b1 = pre-shock trend (slope)
    b2 = immediate LEVEL shift at the moment of the shock
    b3 = change in TREND (slope) after the shock, relative to pre-shock trend

Standard errors use Newey-West (HAC) to account for autocorrelation, which
is standard practice for time-series regression.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_its(df: pd.DataFrame, shock_year: int, outcome_col: str = "log_enrollment",
            year_col: str = "year") -> dict:
    d = df.copy().sort_values(year_col).reset_index(drop=True)
    d["time"] = d[year_col] - d[year_col].min()
    d["post"] = (d[year_col] >= shock_year).astype(int)
    d["time_since_shock"] = d["post"] * (d[year_col] - shock_year)

    X = sm.add_constant(d[["time", "post", "time_since_shock"]])
    y = d[outcome_col]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    return {
        "pre_trend_slope": model.params["time"],
        "level_shift": model.params["post"],
        "level_shift_pvalue": model.pvalues["post"],
        "trend_change": model.params["time_since_shock"],
        "trend_change_pvalue": model.pvalues["time_since_shock"],
        "model_summary": model,
    }