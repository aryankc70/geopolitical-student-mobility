"""Join all Phase 0 data sources into a single country x panel_year dataset,
with one binary shock-exposure column per cataloged shock.

Time alignment note (documented simplification): different sources use
different year conventions (academic year, US fiscal year, calendar year).
This script aligns everything on `panel_year` = the starting calendar year
of each period (e.g. academic year "2020/21" -> 2020, fiscal year 2020 -> 2020,
calendar year 2020 -> 2020). This is an approximation adequate for
annual-granularity analysis, not a precise calendar mapping.
"""
import pandas as pd


TARGET_COUNTRIES = [
    "China", "India", "Nepal", "Afghanistan", "Iran", "Iraq", "Israel",
    "Russia", "Ukraine", "Mexico",
    "South Korea", "Vietnam", "Brazil", "Nigeria", "Canada", "Japan",
]


def academic_year_to_panel_year(academic_year: str) -> int:
    """'2020/21' -> 2020"""
    return int(str(academic_year).split("/")[0])


def load_enrollment(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_year"] = df["academic_year"].apply(academic_year_to_panel_year)
    return df[["country", "panel_year", "total_enrollment"]]


def load_enrollment_by_level(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_year"] = df["academic_year"].apply(academic_year_to_panel_year)
    pivoted = df.pivot_table(
        index=["country", "panel_year"],
        columns="academic_level",
        values="enrollment",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    rename_map = {
        "Under-graduate": "enrollment_undergrad",
        "Graduate": "enrollment_grad",
        "Other": "enrollment_other",
    }
    pivoted = pivoted.rename(columns=rename_map)
    for col in ["enrollment_undergrad", "enrollment_grad", "enrollment_other"]:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA
    return pivoted[["country", "panel_year", "enrollment_undergrad", "enrollment_grad", "enrollment_other"]]


def load_visa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_year"] = df["fiscal_year"].astype(int)
    return df[["country", "panel_year", "f1_visas_issued"]]


def load_nafsa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_year"] = df["academic_year"].apply(academic_year_to_panel_year)
    return df[["panel_year", "total_economic_contribution_usd", "total_jobs_supported"]]


def load_cpi(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_year"] = df["year"].astype(int)
    return df[["panel_year", "cpi_annual_avg"]]


def load_shock_flags(path: str, countries: list, year_range: range) -> pd.DataFrame:
    """One binary column per shock_id: 1 if the country was affected AND
    the panel_year falls within [start_date, end_date] (or start_date to
    present if end_date is blank)."""
    shocks = pd.read_csv(path)

    base = pd.DataFrame(
        [(c, y) for c in countries for y in year_range],
        columns=["country", "panel_year"],
    )

    for _, shock in shocks.iterrows():
        shock_id = shock["shock_id"]
        affected = [c.strip() for c in str(shock["affected_countries"]).split("|")]
        start_year = pd.to_datetime(shock["start_date"]).year
        if pd.notna(shock["end_date"]) and str(shock["end_date"]).strip() != "":
            end_year = pd.to_datetime(shock["end_date"]).year
        else:
            end_year = 9999  # ongoing

        col_name = f"shock_{shock_id}_active"

        def is_active(row, affected=affected, start_year=start_year, end_year=end_year):
            country_match = row["country"] in affected or affected == ["All countries"]
            year_match = start_year <= row["panel_year"] <= end_year
            return int(country_match and year_match)

        base[col_name] = base.apply(is_active, axis=1)

    return base


def build_panel(data_dir: str = "data") -> pd.DataFrame:
    enrollment = load_enrollment(f"{data_dir}/processed/enrollment_by_country.csv")
    enrollment_level = load_enrollment_by_level(f"{data_dir}/processed/enrollment_by_country_level.csv")
    visa = load_visa(f"{data_dir}/processed/visa_by_country.csv")
    nafsa = load_nafsa(f"{data_dir}/processed/nafsa_economic_contribution.csv")
    cpi = load_cpi(f"{data_dir}/processed/cpi_annual.csv")

    all_years = range(2000, 2025)
    shock_flags = load_shock_flags(f"{data_dir}/raw/shock_catalog.csv", TARGET_COUNTRIES, all_years)

    panel = shock_flags.copy()
    panel = panel.merge(enrollment, on=["country", "panel_year"], how="left")
    panel = panel.merge(enrollment_level, on=["country", "panel_year"], how="left")
    panel = panel.merge(visa, on=["country", "panel_year"], how="left")
    panel = panel.merge(nafsa, on="panel_year", how="left")
    panel = panel.merge(cpi, on="panel_year", how="left")

    return panel


if __name__ == "__main__":
    panel = build_panel()
    print("Panel shape:", panel.shape)
    print("\nColumns:", list(panel.columns))
    panel.to_csv("data/processed/panel_dataset.csv", index=False)
    print("\nSaved to data/processed/panel_dataset.csv")