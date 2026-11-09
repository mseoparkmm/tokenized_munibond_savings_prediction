from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Required so joblib can resolve the classes saved in the research models.
from model_core import Estimator  # noqa: F401
from data_and_diagnostics import RATING, MAIN_PURPOSE

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "trained_models_436"
COMPONENTS = [
    "Underwriter",
    "Bond Counsel",
    "Financial Advisor",
    "Paying Agent",
    "Rating Agency Fee",
    "Other Costs",
]
MODELS = {
    comp: joblib.load(MODEL_DIR / f"{comp.replace(' ', '_')}.joblib")
    for comp in COMPONENTS
}

OBSERVED_RATINGS = {"AAA", "Aaa", "AA+", "Aa1", "AA", "Aa2", "AA-", "Aa3", "A+", "A", "A-", "Baa3"}
TRAIN_MIN = 955_000.0
TRAIN_MAX = 263_855_000.0

app = FastAPI(title="Municipal Bond Cost Comparator", version="436-positive-fee")

class PredictionRequest(BaseModel):
    principal: float = Field(gt=0)
    rating: str
    sale: str
    government: str
    purpose: str
    rating_agency: str
    year: int = Field(ge=2015)


def make_frame(x: PredictionRequest) -> pd.DataFrame:
    if x.rating not in RATING:
        raise HTTPException(400, "Unknown credit rating.")
    if x.sale not in ("Competitive", "Negotiated"):
        raise HTTPException(400, "Sale type must be Competitive or Negotiated.")
    if x.government not in ("CITY", "COUNTY"):
        raise HTTPException(400, "Government type must be CITY or COUNTY.")
    if x.rating_agency not in ("S&P", "Moody's", "Fitch"):
        raise HTTPException(400, "Rating agency must be S&P, Moody's, or Fitch.")
    purpose = x.purpose if x.purpose in MAIN_PURPOSE else "OtherPurpose"
    return pd.DataFrame([{
        "ActualPar": float(x.principal),
        "logP": float(np.log(x.principal / 1e6)),
        "RatingScore": int(RATING[x.rating]),
        "SaleType": x.sale,
        "GovernmentType": x.government,
        "PurposeGroup": purpose,
        "FiscalYearIssuance": int(x.year),
        "RatingAgency": x.rating_agency,
    }])

@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")

@app.get("/api/health")
def health():
    return {"ok": True, "model": "436 positive-fee production model", "components": COMPONENTS}

@app.post("/api/predict")
def predict(x: PredictionRequest):
    frame = make_frame(x)
    result = {}
    for comp, model in MODELS.items():
        pred = float(model.predict(frame)[0])
        if not math.isfinite(pred) or pred < 0:
            raise HTTPException(500, f"Invalid prediction for {comp}.")
        result[comp] = pred
    total = float(sum(result.values()))
    retained = float(result["Bond Counsel"] + result["Financial Advisor"] + result["Rating Agency Fee"] + result["Other Costs"])
    warnings = []
    if x.principal < TRAIN_MIN or x.principal > TRAIN_MAX:
        warnings.append("Principal amount is outside the $0.955M-$263.855M production training range.")
    if x.rating not in OBSERVED_RATINGS:
        warnings.append("The selected rating label was recognized by the rating map but was not observed in the 436-case production sample.")
    if x.year > 2025:
        warnings.append("Fiscal year is prospective relative to the FY2015-FY2025 training sample.")
    return {
        "components_USD": result,
        "total_issuance_cost_USD": total,
        "retained_tokenized_components_USD": retained,
        "warnings": warnings,
        "model": {
            "sample_n": 436,
            "validation_n": 187,
            "validation_period": "FY2022-FY2025",
            "MAE_USD": 71119.97241680918,
            "RMSE_USD": 121964.6313350553,
            "R2": 0.8395379007736699,
            "MAPE_percent": 18.3773818366224,
            "total_definition": "sum of six component predictions",
        },
    }

# Version-pinned, local-only full-factorial cost model data, generated from
# Tokenized_Bond_3Chain_Integrated_Model_v1_20261109.
@app.get("/research-model.js")
def integrated_tokenized_model_js():
    return FileResponse(ROOT / "historical_data" / "integrated_tokenized_engine.js", media_type="application/javascript")


# Research files remain in the offline package, but the data-download endpoint
# is intentionally unavailable while section 06 is Under Development.

# Logo assets are limited to the two user-provided files.
LOGOS = {"difin-logo.png", "ubri-logo.png"}

@app.get("/assets/{filename}")
def logo_asset(filename: str):
    if filename not in LOGOS:
        raise HTTPException(status_code=404, detail="Asset not available")
    return FileResponse(ROOT / "assets" / filename)
