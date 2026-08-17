"""FastAPI application serving the forecasting model and causal findings."""
from fastapi import FastAPI, HTTPException
from typing import Optional, List
from src.serving.schemas import ForecastRequest, ForecastResponse, CausalResult
from src.serving.inference import predict_growth, get_causal_results, load_artifacts

app = FastAPI(
    title="Geopolitical Shocks & Student Mobility API",
    description="Causal findings and enrollment forecasting for geopolitical/policy shocks affecting international student mobility.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "student-mobility-api"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        result = predict_growth(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/causal-results", response_model=List[CausalResult])
def causal_results(shock_id: Optional[str] = None):
    try:
        results = get_causal_results(shock_id)
        if shock_id and not results:
            raise HTTPException(status_code=404, detail=f"No results found for shock_id '{shock_id}'")
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))