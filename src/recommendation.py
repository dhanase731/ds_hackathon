"""
recommendation.py  –  Business Action Layer
Converts model predictions into manager-ready recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROCESSED_DIR = BASE_DIR / "data" / "processed"


def classify_risk(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def compute_reorder_qty(forecast_demand: float, current_stock: float,
                        units_ordered: float = 0, safety_factor: float = 1.2) -> int:
    recommended = forecast_demand * safety_factor
    reorder = max(0, recommended - current_stock - units_ordered)
    return int(round(reorder))


def build_recommendation_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the manager-ready recommendation table from the feature-engineered dataset.
    Uses the latest row per store-product.
    """
    from demand_model import predict_next_7_days
    from stockout_model import predict_stockout
    from explainability import explain_high_risk_row

    demand_pred = predict_next_7_days(feature_df)
    stockout_pred = predict_stockout(feature_df)

    merged = demand_pred.merge(stockout_pred, on=["store_id", "product_id"], how="inner")

    # get latest units_ordered
    latest = (
        feature_df.sort_values("date")
        .groupby(["store_id", "product_id"])
        .last()
        .reset_index()[["store_id", "product_id", "units_ordered"]]
    )
    merged = merged.merge(latest, on=["store_id", "product_id"], how="left")
    merged["units_ordered"] = merged["units_ordered"].fillna(0)

    merged["risk_level"] = merged["stockout_prob"].apply(classify_risk)
    merged["reorder_qty"] = merged.apply(
        lambda r: compute_reorder_qty(
            r["next_7_day_demand"], r["inventory_level"], r["units_ordered"]
        ), axis=1
    )

    # Explain top HIGH-risk rows
    latest_full = (
        feature_df.sort_values("date")
        .groupby(["store_id", "product_id"])
        .last()
        .reset_index()
    )
    explanations = {}
    high_risk = merged[merged["risk_level"] == "HIGH"]
    for _, row in high_risk.iterrows():
        key = (row["store_id"], row["product_id"])
        full_row = latest_full[
            (latest_full["store_id"] == row["store_id"]) &
            (latest_full["product_id"] == row["product_id"])
        ]
        if not full_row.empty:
            drivers = explain_high_risk_row(full_row.iloc[0])
            explanations[key] = "; ".join(
                [f"{d['driver']} ({d['impact_pct']}%)" for d in drivers[:3]]
            )

    merged["key"] = list(zip(merged["store_id"], merged["product_id"]))
    merged["top_drivers"] = merged["key"].map(explanations).fillna("—")
    merged = merged.drop(columns=["key"])

    merged["action"] = merged["risk_level"].map({
        "HIGH": "[HIGH] RAISE REPLENISHMENT ORDER TODAY",
        "MEDIUM": "[MEDIUM] MONITOR & CONSIDER REORDER",
        "LOW": "[LOW] NO ACTION REQUIRED",
    })

    cols = [
        "store_id", "product_id",
        "inventory_level", "next_7_day_demand",
        "stockout_prob", "risk_level",
        "reorder_qty", "top_drivers", "action"
    ]
    result = merged[cols].copy()
    result["stockout_prob"] = result["stockout_prob"].round(3)
    result["next_7_day_demand"] = result["next_7_day_demand"].round(1)
    result = result.sort_values(["risk_level", "stockout_prob"],
                                key=lambda x: x.map({"HIGH": 0, "MEDIUM": 1, "LOW": 2})
                                if x.name == "risk_level" else -x,
                                ascending=True)
    result.to_csv(PROCESSED_DIR / "recommendations.csv", index=False)
    return result


if __name__ == "__main__":
    feature_df = pd.read_csv(PROCESSED_DIR / "feature_engineered_dataset.csv",
                             parse_dates=["date"])
    rec = build_recommendation_table(feature_df)
    print(rec.head(10).to_string())
