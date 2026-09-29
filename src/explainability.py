"""
explainability.py  –  Feature Importance & Driver Explanations
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FEATURE_LABELS = {
    "demand_lag_1": "Yesterday's demand",
    "demand_lag_7": "Last week demand",
    "demand_lag_14": "2-week lag demand",
    "demand_lag_28": "4-week lag demand",
    "demand_rolling_mean_7": "7-day avg demand",
    "demand_rolling_mean_14": "14-day avg demand",
    "demand_rolling_mean_28": "28-day avg demand",
    "demand_rolling_std_7": "Demand volatility (7d)",
    "inventory_level": "Current inventory",
    "units_ordered": "Units on order",
    "price": "Selling price",
    "discount": "Discount %",
    "promotion": "Promotion active",
    "competitor_pricing": "Competitor price",
    "price_vs_competitor_ratio": "Price vs competitor",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend flag",
    "month": "Month",
    "quarter": "Quarter",
    "month_sin": "Month (cyclic)",
    "month_cos": "Month (cyclic)",
    "day_of_week_sin": "Day (cyclic)",
    "day_of_week_cos": "Day (cyclic)",
    "epidemic": "Epidemic period",
    "inventory_days_of_cover": "Days of inventory cover",
    "inventory_demand_ratio": "Inventory/demand ratio",
    "low_inventory_flag": "Low inventory flag",
    "inventory_utilization": "Inventory utilisation",
}


def get_demand_feature_importance():
    path = MODELS_DIR / "demand_model.pkl"
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feats, "importance": imp})
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return df.sort_values("importance", ascending=False).head(15)


def get_stockout_feature_importance():
    path = MODELS_DIR / "stockout_model.pkl"
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feats, "importance": imp})
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return df.sort_values("importance", ascending=False).head(15)


def explain_high_risk_row(row: pd.Series, top_n: int = 5) -> list[dict]:
    """
    Return a list of {driver, impact_pct} for a single high-risk row.
    Uses the stockout model's feature importances weighted by feature values.
    """
    path = MODELS_DIR / "stockout_model.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]

    imp = model.feature_importances_
    vals = np.array([row.get(f, 0) for f in feats], dtype=float)
    # normalise values to [0,1] range for weighting
    vals_norm = np.abs(vals) / (np.abs(vals).max() + 1e-9)
    scores = imp * vals_norm
    total = scores.sum() + 1e-9

    idx = np.argsort(scores)[::-1][:top_n]
    drivers = []
    for i in idx:
        label = FEATURE_LABELS.get(feats[i], feats[i])
        pct = round(scores[i] / total * 100, 1)
        drivers.append({"driver": label, "impact_pct": pct})
    return drivers
