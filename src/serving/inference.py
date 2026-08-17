"""Inference logic for the forecasting model and causal-results lookup."""
import os
import json
import joblib
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "forecast_model.joblib")
_COLUMNS_PATH = os.path.join(_PROJECT_ROOT, "models", "forecast_feature_columns.json")
_CAUSAL_RESULTS_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "causal_results.csv")

_model = None
_feature_columns = None
_causal_results = None


def load_artifacts():
    global _model, _feature_columns, _causal_results
    _model = joblib.load(_MODEL_PATH)
    with open(_COLUMNS_PATH) as f:
        _feature_columns = json.load(f)
    _causal_results = pd.read_csv(_CAUSAL_RESULTS_PATH)
    return _model, _feature_columns, _causal_results


def predict_growth(request_dict: dict) -> dict:
    if _model is None:
        load_artifacts()

    df = pd.DataFrame([request_dict])[_feature_columns]
    predicted = float(_model.predict(df)[0])

    pct = predicted * 100
    direction = "growth" if predicted > 0 else "decline"
    interpretation = f"Model predicts approximately {abs(pct):.1f}% {direction} in enrollment next year."

    return {
        "predicted_next_yr_growth": round(predicted, 4),
        "interpretation": interpretation,
    }


def get_causal_results(shock_id: str = None) -> list:
    if _causal_results is None:
        load_artifacts()

    df = _causal_results
    if shock_id:
        df = df[df["shock_id"] == shock_id.upper()]
    return df.to_dict(orient="records")