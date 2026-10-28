
from pathlib import Path
import re

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parent
RENT_DATA = ROOT / "data.csv"
BUY_DATA = ROOT / "house_prices.csv"

LOCALITIES = [
    "Kothrud",
    "Karve Nagar",
    "Bavdhan",
    "Warje",
    "Baner",
    "Wakad",
    "Hinjewadi",
]

FEATURES = ["locality", "bhk", "area_sqft", "furnishing"]
CAT = ["locality", "furnishing"]
NUM = ["bhk", "area_sqft"]

app = FastAPI(title="EstateLens Pune", version="1.2")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

models = {}
metrics = {}
usable_rows = {}
model_errors = {}


def clean_text(value):
    return "" if pd.isna(value) else str(value).strip()


def normalize_furnishing(value):
    s = clean_text(value).lower()
    if "semi" in s:
        return "Semi-furnished"
    if "furnish" in s and "unfurnish" not in s:
        return "Furnished"
    return "Unfurnished"


def parse_amount(value):
    """Convert listing prices such as 62 Lac or 1.65 Cr to rupees."""
    if pd.isna(value):
        return np.nan

    s = (
        str(value).strip().lower()
        .replace(",", "").replace("₹", "").replace("inr", "")
    )

    if not s or "call for price" in s or s in {"nan", "none", "-"}:
        return np.nan

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(cr|crore|crores|lac|lakh|lakhs|k)?",
        s
    )
    if not match:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2)

    if unit in {"cr", "crore", "crores"}:
        number *= 10_000_000
    elif unit in {"lac", "lakh", "lakhs"}:
        number *= 100_000
    elif unit == "k":
        number *= 1_000

    return number if number >= 100_000 else np.nan


def parse_area(value):
    if pd.isna(value):
        return np.nan

    s = str(value).lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", s)

    if not match:
        return np.nan

    area = float(match.group(1))
    return area if 250 <= area <= 10000 else np.nan


def parse_bhk(value):
    s = clean_text(value).lower()

    match = re.search(r"\b([1-5])\s*bhk\b", s)
    if match:
        return int(match.group(1))

    match = re.search(r"\b([1-5])\s*bed", s)
    return int(match.group(1)) if match else np.nan


def infer_locality(text):
    s = clean_text(text).lower()

    for locality in sorted(LOCALITIES, key=len, reverse=True):
        if locality.lower() in s:
            return locality

    aliases = {
        "karvenagar": "Karve Nagar",
        "karve nagar": "Karve Nagar",
        "hinjewadi phase": "Hinjewadi",
    }

    for needle, locality in aliases.items():
        if needle in s:
            return locality

    return None


def prepare_rent():
    if not RENT_DATA.exists():
        raise RuntimeError(f"Missing rental dataset: {RENT_DATA}")

    raw = pd.read_csv(RENT_DATA)

    required = {"city", "locality", "beds", "area", "rent", "furnishing"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Rental CSV missing columns: {sorted(missing)}")

    df = raw[
        raw["city"].astype(str).str.strip().str.lower().eq("pune")
    ].copy()

    df["locality"] = df["locality"].apply(infer_locality)
    df["bhk"] = pd.to_numeric(df["beds"], errors="coerce")
    df["area_sqft"] = pd.to_numeric(df["area"], errors="coerce")
    df["rent_target"] = pd.to_numeric(df["rent"], errors="coerce")
    df["furnishing"] = df["furnishing"].apply(normalize_furnishing)

    df = df[df["locality"].notna()]
    df = df[
        df["bhk"].between(1, 5)
        & df["area_sqft"].between(250, 5000)
        & df["rent_target"].between(1000, 500000)
    ]
    df = df.dropna(subset=["bhk", "area_sqft", "rent_target"])

    return df[FEATURES + ["rent_target"]].copy()


def prepare_buy():
    if not BUY_DATA.exists():
        raise RuntimeError(
            "Buying dataset is not available. Add house_prices.csv "
            "with the required property listing columns."
        )

    raw = pd.read_csv(BUY_DATA, low_memory=False)

    required = {
        "location", "Transaction", "Title", "Society", "Description",
        "Carpet Area", "Super Area", "Amount(in rupees)", "Furnishing"
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            f"Buying CSV missing columns: {sorted(missing)}"
        )

    raw = raw[
        raw["location"].astype(str).str.contains(
            "pune", case=False, na=False
        )
    ].copy()

    raw = raw[
        raw["Transaction"].astype(str).str.lower().isin(
            ["resale", "new property"]
        )
    ].copy()

    raw["locality"] = raw.apply(
        lambda row: infer_locality(" ".join([
            clean_text(row.get("Title")),
            clean_text(row.get("Society")),
            clean_text(row.get("Description")),
            clean_text(row.get("location")),
        ])),
        axis=1,
    )

    raw["bhk"] = raw["Title"].apply(parse_bhk)
    raw["area_sqft"] = raw["Carpet Area"].apply(parse_area)

    missing_area = raw["area_sqft"].isna()
    raw.loc[missing_area, "area_sqft"] = (
        raw.loc[missing_area, "Super Area"].apply(parse_area)
    )

    raw["buy_target"] = raw["Amount(in rupees)"].apply(parse_amount)
    raw["furnishing"] = raw["Furnishing"].apply(normalize_furnishing)

    raw = raw[raw["locality"].notna()]
    raw = raw[
        raw["bhk"].between(1, 5)
        & raw["area_sqft"].between(250, 5000)
        & raw["buy_target"].between(500_000, 100_000_000)
    ]
    raw = raw.dropna(subset=["bhk", "area_sqft", "buy_target"])

    return raw[FEATURES + ["buy_target"]].copy()


def train_model(df, target):
    if len(df) < 10:
        raise RuntimeError(
            f"Only {len(df)} usable records for {target}; at least 10 needed."
        )

    X = df[FEATURES].copy()
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])

    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            min_samples_leaf=2,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0

    return model, {
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "test_rows": len(y_test),
    }


@app.on_event("startup")
def startup():
    models.clear()
    metrics.clear()
    usable_rows.clear()
    model_errors.clear()

    # Rent model is required for the application to start.
    rent_df = prepare_rent()
    models["rent"], metrics["rent"] = train_model(
        rent_df, "rent_target"
    )
    usable_rows["rent"] = len(rent_df)

    # Buying model is optional until its dataset is available.
    try:
        buy_df = prepare_buy()
        models["buy"], metrics["buy"] = train_model(
            buy_df, "buy_target"
        )
        usable_rows["buy"] = len(buy_df)
    except Exception as exc:
        model_errors["buy"] = str(exc)
        usable_rows["buy"] = 0
        print(f"Buying model unavailable: {exc}")

    print(
        f"EstateLens started. Rent rows={usable_rows.get('rent', 0)}, "
        f"buy rows={usable_rows.get('buy', 0)}"
    )
    print("Listing prices are asking prices, not verified sale transactions.")


class Request(BaseModel):
    locality: str
    bhk: int = Field(ge=1, le=5)
    area_sqft: int = Field(ge=250, le=5000)
    furnishing: Literal["Unfurnished", "Semi-furnished", "Furnished"]
    age_years: int = Field(default=0, ge=0, le=60)


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok" if "rent" in models else "starting",
        "models_ready": "rent" in models,
        "available_models": list(models.keys()),
        "dataset_rows": {
            "rent": usable_rows.get("rent", 0),
            "buy": usable_rows.get("buy", 0),
        },
        "metrics": metrics,
        "model_errors": model_errors,
        "data_note": (
            "Estimates use cleaned property listings and are not "
            "verified market valuations."
        ),
    }


@app.post("/api/predict/{mode}")
def predict(mode: Literal["rent", "buy"], req: Request):
    if mode not in models:
        detail = model_errors.get(
            mode, f"The {mode} prediction model is not available."
        )
        raise HTTPException(status_code=503, detail=detail)

    if req.locality not in LOCALITIES:
        raise HTTPException(
            status_code=400,
            detail="Choose a supported Pune locality.",
        )

    row = pd.DataFrame([{
        "locality": req.locality,
        "bhk": req.bhk,
        "area_sqft": req.area_sqft,
        "furnishing": req.furnishing,
    }], columns=FEATURES)

    amount = max(
        0,
        int(round(float(models[mode].predict(row)[0]) / 500) * 500),
    )

    return {
        "mode": mode,
        "locality": req.locality,
        "estimated_price_inr": amount,
        "formatted_price": f"₹{amount:,.0f}" + (
            "/month" if mode == "rent" else ""
        ),
        "note": (
            "Listing-based estimate; not a verified market valuation."
        ),
    }
