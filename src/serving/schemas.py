"""Pydantic schemas for the geopolitical shocks / student mobility API."""
from pydantic import BaseModel
from typing import Optional


class ForecastRequest(BaseModel):
    prior_3yr_avg_growth: float
    any_shock_active: int
    num_shocks_active: int
    cpi_annual_avg: float

    class Config:
        json_schema_extra = {
            "example": {
                "prior_3yr_avg_growth": 0.03,
                "any_shock_active": 1,
                "num_shocks_active": 2,
                "cpi_annual_avg": 310.5
            }
        }


class ForecastResponse(BaseModel):
    predicted_next_yr_growth: float
    interpretation: str


class CausalResult(BaseModel):
    shock_id: str
    shock_name: str
    shock_type: str
    treated_country: str
    pre_trend_status: str
    effect_estimate: float
    p_value: float
    significant: str
    reliability: str
    notes: str