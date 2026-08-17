"""Build features and target for the enrollment forecasting model.

Target: next-year change in log(enrollment) for a country - i.e. given
everything known at time T, predict how much enrollment will grow or
shrink by T+1. This is a genuinely different question from the Phase 2
causal analysis and should never be conflated with it: this model
predicts outcomes, it does not estimate the causal effect of any shock.
"""
import pandas as pd
import numpy as np


def build_forecast_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(['country', 'panel_year']).reset_index(drop=True)
    df['log_enrollment'] = np.log(df['total_enrollment'])

    df['yoy_growth'] = df.groupby('country')['log_enrollment'].diff()

    df['prior_3yr_avg_growth'] = (
        df.groupby('country')['yoy_growth']
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    shock_cols = [c for c in df.columns if c.startswith('shock_') and c.endswith('_active')]
    df['any_shock_active'] = df[shock_cols].max(axis=1)
    df['num_shocks_active'] = df[shock_cols].sum(axis=1)

    df['target_next_yr_growth'] = df.groupby('country')['yoy_growth'].shift(-1)

    feature_cols = [
        'country', 'panel_year', 'prior_3yr_avg_growth', 'any_shock_active',
        'num_shocks_active', 'cpi_annual_avg', 'target_next_yr_growth'
    ]
    result = df[feature_cols].dropna(subset=['prior_3yr_avg_growth', 'target_next_yr_growth'])
    return result


if __name__ == "__main__":
    panel = pd.read_csv("data/processed/panel_dataset.csv")
    features = build_forecast_dataset(panel)
    features.to_csv("data/processed/forecast_features.csv", index=False)
    print(features.shape)