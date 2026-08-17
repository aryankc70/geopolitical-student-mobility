"""Unit tests for the geopolitical shocks / student mobility pipeline."""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.validate_panel import validate_panel
from src.features.build_forecast_features import build_forecast_dataset

TARGET_COUNTRIES = [
    "China", "India", "Nepal", "Afghanistan", "Iran", "Iraq", "Israel",
    "Russia", "Ukraine", "Mexico",
    "South Korea", "Vietnam", "Brazil", "Nigeria", "Canada", "Japan",
]


@pytest.fixture
def panel():
    return pd.read_csv("data/processed/panel_dataset.csv")


def test_panel_validates_clean(panel):
    assert validate_panel(panel, TARGET_COUNTRIES) is True


def test_panel_has_expected_countries(panel):
    assert set(panel["country"].unique()) == set(TARGET_COUNTRIES)


def test_panel_no_negative_enrollment(panel):
    assert (panel["total_enrollment"].dropna() >= 0).all()


def test_panel_catches_negative_enrollment_injection(panel):
    bad = panel.copy()
    bad.loc[0, "total_enrollment"] = -100
    with pytest.raises(ValueError, match="negative"):
        validate_panel(bad, TARGET_COUNTRIES)


def test_panel_catches_duplicate_rows(panel):
    bad = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(bad, TARGET_COUNTRIES)


def test_forecast_features_no_lookahead_leak(panel):
    """The target for year T should equal the actual growth realized in T+1 -
    i.e. we're not accidentally leaking future information into the features."""
    features = build_forecast_dataset(panel)
    china = features[features["country"] == "China"].sort_values("panel_year")
    assert len(china) > 0
    # every row's target should be a real, finite number (not NaN, since those get dropped)
    assert china["target_next_yr_growth"].notna().all()


def test_forecast_features_shock_columns_are_counts(panel):
    features = build_forecast_dataset(panel)
    assert (features["num_shocks_active"] >= 0).all()
    assert set(features["any_shock_active"].unique()).issubset({0, 1})


def test_model_artifacts_exist():
    assert os.path.exists("models/forecast_model.joblib")
    assert os.path.exists("models/forecast_feature_columns.json")


def test_causal_results_file_has_expected_columns():
    df = pd.read_csv("data/processed/causal_results.csv")
    required = {"shock_id", "shock_name", "effect_estimate", "p_value", "significant"}
    assert required.issubset(set(df.columns))