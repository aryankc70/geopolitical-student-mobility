"""Extract country-level enrollment by academic level from the IIE Open Doors
'Places of Origin and Academic Level' workbook."""
import openpyxl
import pandas as pd

SHEET_NAME = "Places of Origin&Academic Level"

TARGET_COUNTRIES = [
    "China", "India", "Nepal", "Afghanistan", "Iran", "Iraq", "Israel",
    "Russia", "Ukraine", "Mexico",
    "South Korea", "Vietnam", "Brazil", "Nigeria", "Canada", "Japan",
]


def load_enrollment_by_level(source_path: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(source_path, data_only=True)
    ws = wb[SHEET_NAME]

    year_row = [c.value for c in ws[2]]
    level_row = [c.value for c in ws[3]]

    col_map = []
    current_year = None
    for idx, val in enumerate(year_row, start=1):
        if val is not None:
            current_year = val
        lvl = level_row[idx - 1]
        if lvl in ("Under-graduate", "Graduate", "Other"):
            col_map.append((idx, current_year, lvl))

    records = []
    matched = set()
    for row in ws.iter_rows(min_row=5, max_row=303):
        name = row[0].value
        if name is None:
            continue
        name_clean = str(name).strip()
        if name_clean in TARGET_COUNTRIES:
            matched.add(name_clean)
            for col_idx, year_label, level in col_map:
                val = row[col_idx - 1].value
                if val in (None, "-", ""):
                    continue
                records.append({
                    "country": name_clean,
                    "academic_year": year_label,
                    "academic_level": level,
                    "enrollment": val,
                })

    missing = set(TARGET_COUNTRIES) - matched
    if missing:
        print(f"WARNING: not found: {missing}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_enrollment_by_level("/Users/aryan/Documents/geopolitical-student-mobility/data/raw/Census_Places-of-Origin-Academic-Level_OD25_Website.xlsx")
    df.to_csv("data/processed/enrollment_by_country_level.csv", index=False)
    print(df.shape)