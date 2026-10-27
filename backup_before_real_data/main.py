from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pune_properties.csv"
LOCALITIES = ["Kothrud", "Karve Nagar", "Bavdhan", "Warje", "Baner", "Wakad", "Hinjewadi"]
FEATURES = ["locality", "bhk", "area_sqft", "furnishing", "age_years"]
CAT = ["locality", "furnishing"]
NUM = ["bhk", "area_sqft", "age_years"]

app = FastAPI(title="EstateLens Pune", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

models = {}
metrics = {}

def train(target: str):
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[target], test_size=0.2, random_state=42, stratify=df["locality"]
    )
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", NUM)])
    model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2))])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return model, len(df), {"mae": round(mae, 2), "r2": round(r2, 4)}

@app.on_event("startup")
def startup():
    models["rent"], n, metrics["rent"] = train("rent_inr")
    models["buy"], _, metrics["buy"] = train("buy_price_inr")
    models["rows"] = n
    models["metrics"] = metrics
    print(f"EstateLens: trained on {n} synthetic rows across {len(LOCALITIES)} localities.")
    print(f"  Rent Model -> MAE: ₹{metrics['rent']['mae']:,.2f}, R²: {metrics['rent']['r2']:.4f}")
    print(f"  Buy Model  -> MAE: ₹{metrics['buy']['mae']:,.2f}, R²: {metrics['buy']['r2']:.4f}")

class Request(BaseModel):
    locality: str
    bhk: int = Field(ge=1, le=5)
    area_sqft: int = Field(ge=250, le=5000)
    furnishing: Literal["Unfurnished", "Semi-furnished", "Furnished"]
    age_years: int = Field(ge=0, le=60)

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_ready": "rent" in models and "buy" in models,
        "dataset_rows": models.get("rows", 0),
        "metrics": models.get("metrics", {})
    }

@app.post("/api/predict/{mode}")
def predict(mode: Literal["rent", "buy"], req: Request):
    if mode not in models:
        raise HTTPException(503, "Model is not ready")
    if req.locality not in LOCALITIES:
        raise HTTPException(400, "Choose a supported Pune locality")
    amount = max(0, round(float(models[mode].predict(pd.DataFrame([req.model_dump()])[FEATURES])[0]) / 500) * 500)
    return {
        "mode": mode,
        "locality": req.locality,
        "estimated_price_inr": int(amount),
        "formatted_price": f"₹{amount:,.0f}" + ("/month" if mode == "rent" else ""),
        "note": "Synthetic demonstration data only; not a verified market valuation."
    }
