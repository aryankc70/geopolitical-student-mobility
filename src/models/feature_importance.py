"""Inspect which features actually drive the forecasting model's predictions.

Trained on the full dataset (not CV-split) since this is for interpretation,
not evaluation - the honest out-of-sample error estimate comes from the
leave-one-year-out CV in train_forecast.py, not from this script.
"""
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

FEATURE_COLS = ['prior_3yr_avg_growth', 'any_shock_active', 'num_shocks_active', 'cpi_annual_avg']


def get_feature_importance(features: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    if params is None:
        params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}

    model = XGBRegressor(**params, random_state=42)
    model.fit(features[FEATURE_COLS], features['target_next_yr_growth'])

    importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    return importance_df


if __name__ == "__main__":
    features = pd.read_csv("data/processed/forecast_features.csv")
    importance = get_feature_importance(features)
    print(importance.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance['feature'], importance['importance'], color='#3d6a94')
    ax.set_xlabel('XGBoost Feature Importance')
    ax.set_title('What Drives the Forecasting Model\'s Predictions')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png', dpi=100)
    plt.show()