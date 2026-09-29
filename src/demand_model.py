"""
demand_model.py  –  7-Day Demand Forecasting (XGBoost Regression)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "demand_rolling_mean_7", "demand_rolling_mean_14", "demand_rolling_mean_28",
    "demand_rolling_std_7",
    "inventory_level", "units_ordered",
    "price", "discount", "promotion",
    "competitor_pricing", "price_vs_competitor_ratio",
    "day_of_week", "is_weekend", "month", "quarter",
    "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos",
    "epidemic",
    "inventory_days_of_cover", "inventory_demand_ratio",
]


def load_feature_data():
    path = PROCESSED_DIR / "feature_engineered_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first.\n{path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_target(df):
    """next_7_day_demand = sum of demand over next 7 rows per store-product."""
    df = df.sort_values(["store_id", "product_id", "date"]).copy()
    df["next_7_day_demand"] = (
        df.groupby(["store_id", "product_id"])["demand"]
        .transform(lambda x: x.shift(-1).rolling(7, min_periods=1).sum())
    )
    df = df.dropna(subset=["next_7_day_demand"])
    return df


def time_split(df, test_ratio=0.15, val_ratio=0.10):
    dates = df["date"].sort_values().unique()
    n = len(dates)
    val_cut = dates[int(n * (1 - test_ratio - val_ratio))]
    test_cut = dates[int(n * (1 - test_ratio))]
    train = df[df["date"] < val_cut]
    val = df[(df["date"] >= val_cut) & (df["date"] < test_cut)]
    test = df[df["date"] >= test_cut]
    return train, val, test


def available_features(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def train_demand_model(df):
    df = build_target(df)
    feats = available_features(df)
    train, val, test = time_split(df)

    X_train, y_train = train[feats].fillna(0), train["next_7_day_demand"]
    X_val, y_val = val[feats].fillna(0), val["next_7_day_demand"]
    X_test, y_test = test[feats].fillna(0), test["next_7_day_demand"]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    print(f"Demand Model  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    with open(MODELS_DIR / "demand_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feats}, f)

    # attach predictions to test set for dashboard use
    test = test.copy()
    test["predicted_demand"] = preds
    test[["date", "store_id", "product_id", "demand", "predicted_demand"]].to_csv(
        PROCESSED_DIR / "demand_predictions.csv", index=False
    )

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
    return model, feats, metrics


def predict_next_7_days(df):
    """Return latest predicted 7-day demand per store-product."""
    path = MODELS_DIR / "demand_model.pkl"
    if not path.exists():
        raise FileNotFoundError("Train demand model first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]

    latest = df.sort_values("date").groupby(["store_id", "product_id"]).last().reset_index()
    X = latest[feats].fillna(0)
    latest["next_7_day_demand"] = model.predict(X)
    return latest[["store_id", "product_id", "next_7_day_demand"]]


if __name__ == "__main__":
    df = load_feature_data()
    train_demand_model(df)
