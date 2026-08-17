# Geopolitical Shocks & International Student Mobility

A causal-inference and machine-learning system studying how war, diplomatic tension, and domestic visa policy affect international student enrollment in the United States — built on a hand-compiled panel of 17 geopolitical/policy shocks spanning 1999-2024.

## Motivation

Millions of international students plan their futures around US visa and enrollment policy, often navigating scattered news coverage and anecdote to assess risk. This project asks a more rigorous version of a question usually answered informally: **when a geopolitical shock happens, does it actually move the needle on student mobility — and can near-term impact be forecast?**

The project deliberately separates two different questions that are often conflated in public discussion:
- **Causal inference** (Phase 2): did shock X actually cause a measurable effect?
- **Forecasting** (Phase 3): given current conditions, what should we expect next year?

These use different methods, answer different questions, and are never presented as the same thing.

## Headline Finding

Across 11 rigorously tested shocks (war, diplomatic tension, and domestic policy), **no shock produced a statistically significant effect** using difference-in-differences — including widely-assumed disruptors like the Russia-Ukraine war. Where a signal did appear, it pointed toward **specifically-targeted student visa policy** (the 2020 restriction on Chinese STEM graduate students) rather than general diplomatic tension as the more plausible driver — a real, if statistically inconclusive, pattern worth further investigation.

The full methodology, results table, and honest limitations are in [`docs/Phase2_Causal_Findings_Report.pdf`](docs/Phase2_Causal_Findings_Report.pdf).

## Architecture

5 raw data sources → Panel dataset (country × year) → Shock exposure flags
↓
┌───────────────┴───────────────┐
↓ ↓
Phase 2: Causal Analysis Phase 3: ML Forecasting
(DiD + interrupted time-series) (XGBoost + Optuna, MLflow-tracked)
↓ ↓
└───────────────┬───────────────┘
↓
FastAPI (/forecast, /causal-results)
+ Streamlit dashboard
↓
Docker → GitHub Actions CI/CD → Docker Hub


## Key methodological decisions

- **Log-transformed enrollment as the outcome variable.** Countries in this panel vary enormously in scale (China/India in the hundreds of thousands vs. Nepal/Ukraine in the low thousands) — comparing raw levels violates DiD's parallel-trends assumption almost automatically due to scale alone. This was verified empirically: an early raw-level test failed its own pre-trend check; the same test with log-transformed enrollment passed.
- **Every causal estimate is pre-trend validated** before being trusted, using a placebo test with a fake treatment date in the pre-period.
- **A visa-policy measurement artifact was found and documented**, not papered over: a 2014 US-China agreement extending visa validity from 1 to 5 years mechanically crashed China's F-1 visa *issuance* counts without reflecting any real change in mobility. This is cataloged as its own shock (SH015) and is why total *enrollment* — not visa issuance — is used as the primary outcome variable throughout.
- **The forecasting model is evaluated against a naive persistence baseline** using leave-one-year-out cross-validation, and is never used to make or imply a causal claim.
- **5 of 17 cataloged shocks are structurally untestable** with country-vs-country DiD because they affect all countries simultaneously (no valid control group) — documented explicitly rather than forced into an inappropriate method.

## Results summary

| Shock | Type | Effect (log pts) | Significant? | Reliability |
|---|---|---|---|---|
| US-China trade war | diplomatic_tension | +0.19 | No | Good |
| Trump travel ban (Iran) | domestic_policy | +0.04 | No | Good |
| Russia-Ukraine war (Ukraine) | war | +0.05 | No | Good |
| Russia-Ukraine war (Russia) | war | -0.23 | No | Good |
| Prop 10043 / COVID overlap (China) | domestic_policy | -0.10 to -0.23 | No | Moderate |
| COVID-19 (ITS validation case) | domestic_policy | -0.30 | **Yes** | High |
| Public Charge Rule (ITS) | domestic_policy | -0.20 | Yes* | Low (*confounded w/ COVID) |

*Full 11-shock table with methodology notes in the [findings report](docs/Phase2_Causal_Findings_Report.pdf).*

## Forecasting model

XGBoost regressor predicting next-year enrollment growth, tuned via Optuna, evaluated on leave-one-year-out cross-validation:

| | MAE |
|---|---|
| Naive persistence baseline | 0.0870 |
| XGBoost (default params) | 0.0821 (5.6% improvement) |
| XGBoost (Optuna-tuned) | 0.0771 (**11.4% improvement**) |

Feature importance shows shock activity carries real, non-trivial predictive signal (`num_shocks_active` is the third most important feature), consistent with — but not proof of — the causal findings above.

## Tech stack

Python · pandas · statsmodels · linearmodels · XGBoost · Optuna · MLflow · FastAPI · Streamlit · Docker · GitHub Actions · pytest

## Project structure

geopolitical-student-mobility/
├── data/
│ ├── raw/ # Source files + hand-compiled shock catalog
│ └── processed/ # Panel dataset, forecast features, causal results
├── notebooks/
│ └── causal_analysis.ipynb # EDA, DiD tests, ITS validation
├── src/
│ ├── data/ # Extraction scripts (enrollment, visa, NAFSA, CPI)
│ ├── validation/ # Panel data quality checks
│ ├── analysis/ # DiD event-study + interrupted time-series estimators
│ ├── features/ # Forecast feature engineering
│ ├── models/ # Training, tuning, MLflow logging
│ └── serving/ # FastAPI + Streamlit
├── tests/ # pytest suite
├── models/ # Trained forecast model + feature schema
├── docs/ # Causal findings report + data reconciliation notes
├── Dockerfile
└── .github/workflows/ci.yml


## Running it locally

```bash
git clone https://github.com/aryankc70/geopolitical-student-mobility.git
cd geopolitical-student-mobility
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Serve the API
uvicorn src.serving.main:app --reload --port 8001
# -> http://127.0.0.1:8001/docs

# Run the dashboard (separate terminal)
streamlit run src/serving/dashboard.py
# -> http://localhost:8501
```

## Running with Docker

```bash
docker pull aryankc70/student-mobility-api:latest
docker run -d -p 8001:8001 aryankc70/student-mobility-api:latest
```

## CI/CD

Every push to `main` triggers: (1) the full pytest suite, gating (2) a Docker build and push to Docker Hub if tests pass. See `.github/workflows/ci.yml`.

## Limitations

This project's honest limitations — small sample sizes, a structural data floor around 1999-2000, five untestable "all countries" shocks, and the subjective judgment calls involved in building the shock catalog — are documented in full in the [findings report](docs/Phase2_Causal_Findings_Report.pdf). In short: **this is a transparent, limitations-aware research and forecasting tool, not a certified predictive authority on immigration policy.**