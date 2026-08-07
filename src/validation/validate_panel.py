"""Data quality validation for the geopolitical shocks / student mobility panel dataset."""
import pandas as pd

EXPECTED_YEAR_RANGE = (2000, 2024)

VALUE_COLUMNS = [
    "total_enrollment", "enrollment_undergrad", "enrollment_grad", "enrollment_other",
    "f1_visas_issued", "total_economic_contribution_usd", "total_jobs_supported",
    "cpi_annual_avg",
]


def validate_panel(df: pd.DataFrame, target_countries: list) -> bool:
    """
    Validate the panel dataset: schema, value ranges, and internal consistency.
    Raises ValueError if any check fails. Returns True if all checks pass.
    """
    errors = []

    required = {"country", "panel_year"} | set(VALUE_COLUMNS)
    missing_cols = required - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors))

    min_y, max_y = EXPECTED_YEAR_RANGE
    if df["panel_year"].min() < min_y or df["panel_year"].max() > max_y:
        errors.append(
            f"panel_year out of expected range {EXPECTED_YEAR_RANGE}: "
            f"found {df['panel_year'].min()}-{df['panel_year'].max()}"
        )

    unexpected_countries = set(df["country"].unique()) - set(target_countries)
    if unexpected_countries:
        errors.append(f"Unrecognized country values: {unexpected_countries}")

    for col in VALUE_COLUMNS:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                errors.append(f"Column '{col}' has {negative_count} negative value(s)")

    shock_cols = [c for c in df.columns if c.startswith("shock_") and c.endswith("_active")]
    if not shock_cols:
        errors.append("No shock_*_active columns found - expected at least one")
    for col in shock_cols:
        bad_values = set(df[col].dropna().unique()) - {0, 1}
        if bad_values:
            errors.append(f"Shock column '{col}' has non-binary values: {bad_values}")

    dupes = df.duplicated(subset=["country", "panel_year"]).sum()
    if dupes > 0:
        errors.append(f"Found {dupes} duplicate (country, panel_year) row(s)")

    level_sum = df[["enrollment_undergrad", "enrollment_grad", "enrollment_other"]].sum(axis=1, skipna=True)
    has_both = df["total_enrollment"].notna() & (level_sum > 0)
    if has_both.any():
        ratio = (level_sum[has_both] / df.loc[has_both, "total_enrollment"])
        bad_ratio = ((ratio < 0.5) | (ratio > 1.5)).sum()
        if bad_ratio > 0:
            errors.append(
                f"{bad_ratio} row(s) where academic-level sub-totals deviate >50% from "
                f"total_enrollment - worth spot-checking"
            )

    if errors:
        raise ValueError("Panel validation failed:\n" + "\n".join(errors))

    return True


if __name__ == "__main__":
    TARGET_COUNTRIES = [
        "China", "India", "Nepal", "Afghanistan", "Iran", "Iraq", "Israel",
        "Russia", "Ukraine", "Mexico",
        "South Korea", "Vietnam", "Brazil", "Nigeria", "Canada", "Japan",
    ]
    panel = pd.read_csv("data/processed/panel_dataset.csv")
    result = validate_panel(panel, TARGET_COUNTRIES)
    print("Validation passed:", result)