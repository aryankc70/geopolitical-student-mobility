"""Load NAFSA national economic contribution data (hand-compiled and
reconciled against annual press releases / PDFs)."""
import pandas as pd


def load_nafsa_data(source_path: str) -> pd.DataFrame:
    """Load and validate the NAFSA economic contribution CSV."""
    df = pd.read_csv(source_path)

    required_cols = {"academic_year", "total_economic_contribution_usd", "total_jobs_supported"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if (df["total_economic_contribution_usd"] <= 0).any():
        raise ValueError("Found non-positive economic contribution value(s)")
    if (df["total_jobs_supported"] <= 0).any():
        raise ValueError("Found non-positive jobs supported value(s)")

    return df


if __name__ == "__main__":
    df = load_nafsa_data("data/raw/nafsa_economic_contribution.csv")
    df.to_csv("data/processed/nafsa_economic_contribution.csv", index=False)
    print(df.shape)
    print(f"Year range: {df['academic_year'].min()} to {df['academic_year'].max()}")