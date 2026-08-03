"""Fetch annual CPI (inflation) data from the BLS Public Data API.

Series: CUUR0000SA0 - CPI for All Urban Consumers, All Items,
U.S. city average, not seasonally adjusted.

No API key required for basic use (25 queries/day, up to 20 years per call).
If you register for a free key (https://data.bls.gov/registrationEngine/),
set BLS_API_KEY below to raise the query limit, though note the per-call
year range actually drops to 10 years with a registered key - the chunking
logic below handles either case.
"""
import requests
import pandas as pd
import time

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
SERIES_ID = "CUUR0000SA0"
BLS_API_KEY = ""  # optional - leave blank to use unregistered access


def fetch_cpi_chunk(start_year: int, end_year: int) -> list:
    payload = {
        "seriesid": [SERIES_ID],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if BLS_API_KEY:
        payload["registrationkey"] = BLS_API_KEY

    response = requests.post(BLS_API_URL, json=payload, headers={"Content-type": "application/json"})
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError(f"BLS API request failed: {data.get('message')}")

    return data["Results"]["series"][0]["data"]


def load_cpi_data(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    """Fetch annual average CPI for the given year range, chunked to respect BLS's per-call limits."""
    all_records = []
    chunk_size = 10  # conservative, works whether or not you're using an API key
    current_start = start_year

    while current_start <= end_year:
        current_end = min(current_start + chunk_size - 1, end_year)
        print(f"Fetching CPI for {current_start}-{current_end}...")
        chunk_data = fetch_cpi_chunk(current_start, current_end)
        all_records.extend(chunk_data)
        current_start = current_end + 1
        time.sleep(1)  # be polite to the API between calls

    df = pd.DataFrame(all_records)
    # keep monthly values (M01-M12), excluding M13 (annual average, not always present)
    df = df[df["period"].str.startswith("M") & (df["period"] != "M13")]
    df["value"] = df["value"].astype(float)
    df["year"] = df["year"].astype(int)

    # compute annual average CPI from monthly values
    annual = df.groupby("year")["value"].mean().reset_index()
    annual.columns = ["year", "cpi_annual_avg"]
    return annual.sort_values("year")


if __name__ == "__main__":
    df = load_cpi_data(2000, 2024)
    df.to_csv("data/processed/cpi_annual.csv", index=False)
    print(df.shape)
    print(df.head())
    print(df.tail())