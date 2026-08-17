"""Train and evaluate an enrollment-growth forecasting model, tracked in MLflow.

Given the panel's small size, uses leave-one-year-out cross-validation
rather than a single train/test split.
"""
import os
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def naive_baseline_predict(test_df: pd.DataFrame) -> np.ndarray:
    """Naive persistence: predict next year's growth = the recent trend continuing."""
    return test_df['prior_3yr_avg_growth'].values


def evaluate_leave_one_year_out(features: pd.DataFrame) -> dict:
    years = sorted(features['panel_year'].unique())
    baseline_errors, model_errors = [], []
    feature_cols = ['prior_3yr_avg_growth', 'any_shock_active', 'num_shocks_active', 'cpi_annual_avg']

    for test_year in years:
        train = features[features['panel_year'] != test_year]
        test = features[features['panel_year'] == test_year]
        if len(test) == 0 or len(train) < 10:
            continue

        baseline_pred = naive_baseline_predict(test)
        baseline_errors.append(mean_absolute_error(test['target_next_yr_growth'], baseline_pred))

        X_train, y_train = train[feature_cols], train['target_next_yr_growth']
        X_test, y_test = test[feature_cols], test['target_next_yr_growth']

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        model_errors.append(mean_absolute_error(y_test, model.predict(X_test)))

    return {
        'baseline_mae': float(np.mean(baseline_errors)),
        'model_mae': float(np.mean(model_errors)),
        'n_folds': len(baseline_errors),
    }


def train_and_log(features: pd.DataFrame, experiment_name: str = "student-mobility-forecast"):
    mlflow.set_tracking_uri(f"sqlite:///{_PROJECT_ROOT}/mlflow.db")
    mlflow.set_experiment(experiment_name)

    result = evaluate_leave_one_year_out(features)
    improvement = (result['baseline_mae'] - result['model_mae']) / result['baseline_mae'] * 100

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("cv_method", "leave-one-year-out")
        mlflow.log_metric("n_folds", result['n_folds'])
        mlflow.log_metric("baseline_mae", result['baseline_mae'])
        mlflow.log_metric("model_mae", result['model_mae'])
        mlflow.log_metric("improvement_pct", improvement)

        print(f"Folds evaluated: {result['n_folds']}")
        print(f"Naive baseline MAE: {result['baseline_mae']:.4f}")
        print(f"XGBoost model MAE:  {result['model_mae']:.4f}")
        print(f"Improvement over baseline: {improvement:.1f}%")

    return result



def save_model(model, feature_cols, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "forecast_model.joblib"))
    with open(os.path.join(model_dir, "forecast_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"Model saved to {model_dir}/")

if __name__ == "__main__":
    features = pd.read_csv("data/processed/forecast_features.csv")
    train_and_log(features)

    # train and persist the final model using tuned hyperparameters
    feature_cols = ['prior_3yr_avg_growth', 'any_shock_active', 'num_shocks_active', 'cpi_annual_avg']
    final_params = {
        "n_estimators": 242, "max_depth": 2, "learning_rate": 0.014460553348841401,
        "subsample": 0.8620164270101208, "colsample_bytree": 0.6722495225595034,
    }
    final_model = XGBRegressor(**final_params, random_state=42)
    final_model.fit(features[feature_cols], features['target_next_yr_growth'])
    save_model(final_model, feature_cols)