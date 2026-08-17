"""Optuna hyperparameter tuning for the forecasting model, optimizing
average MAE across leave-one-year-out folds (same CV scheme as the
baseline comparison, so tuning results are directly comparable)."""
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

FEATURE_COLS = ['prior_3yr_avg_growth', 'any_shock_active', 'num_shocks_active', 'cpi_annual_avg']


def cv_mae_for_params(features: pd.DataFrame, params: dict) -> float:
    years = sorted(features['panel_year'].unique())
    errors = []
    for test_year in years:
        train = features[features['panel_year'] != test_year]
        test = features[features['panel_year'] == test_year]
        if len(test) == 0 or len(train) < 10:
            continue
        model = XGBRegressor(**params, random_state=42)
        model.fit(train[FEATURE_COLS], train['target_next_yr_growth'])
        pred = model.predict(test[FEATURE_COLS])
        errors.append(mean_absolute_error(test['target_next_yr_growth'], pred))
    return float(np.mean(errors))


def tune_model(features: pd.DataFrame, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        return cv_mae_for_params(features, params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_mae": study.best_value}


if __name__ == "__main__":
    features = pd.read_csv("data/processed/forecast_features.csv")

    default_mae = cv_mae_for_params(features, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1})
    print(f"Default params MAE: {default_mae:.4f}")

    result = tune_model(features, n_trials=30)
    print(f"Tuned MAE: {result['best_mae']:.4f}")
    print(f"Best params: {result['best_params']}")

    baseline_mae = 0.0870  # from train_forecast.py's earlier run - update if it changes
    improvement = (baseline_mae - result['best_mae']) / baseline_mae * 100
    print(f"Improvement over naive baseline: {improvement:.1f}%")