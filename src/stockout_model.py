"""
stockout_model.py  –  Stock-out Risk Classification (Random Forest)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "demand_lag_1", "demand_lag_7", "demand_lag_14",
    "demand_rolling_mean_7", "demand_rolling_mean_14",
    "demand_rolling_std_7",
    "inventory_level", "units_ordered",
    "price", "discount", "promotion",
    "inventory_days_of_cover", "inventory_demand_ratio",
    "low_inventory_flag", "inventory_utilization",
    "day_of_week", "is_weekend", "month",
    "epidemic",
]


def load_feature_data():
    path = PROCESSED_DIR / "feature_engineered_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first.\n{path}")
    return pd.read_csv(path, parse_dates=["date"])


def build_stockout_label(df):
    """stockout_flag = 1 if inventory_level < demand on that day."""
    df = df.copy()
    df["stockout_flag"] = (df["inventory_level"] < df["demand"]).astype(int)
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


def train_stockout_model(df):
    df = build_stockout_label(df)
    feats = available_features(df)
    train, val, test = time_split(df)

    X_train = pd.concat([train, val])[feats].fillna(0)
    y_train = pd.concat([train, val])["stockout_flag"]
    X_test = test[feats].fillna(0)
    y_test = test["stockout_flag"]

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight=cw, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"Stockout Model  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    with open(MODELS_DIR / "stockout_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feats}, f)

    test = test.copy()
    test["stockout_prob"] = probs
    test["stockout_pred"] = preds
    test[["date", "store_id", "product_id", "stockout_flag", "stockout_prob", "stockout_pred"]].to_csv(
        PROCESSED_DIR / "stockout_predictions.csv", index=False
    )

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc, "CM": cm}
    return model, feats, metrics


def predict_stockout(df):
    """Return latest stockout probability per store-product."""
    path = MODELS_DIR / "stockout_model.pkl"
    if not path.exists():
        raise FileNotFoundError("Train stockout model first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["features"]

    latest = df.sort_values("date").groupby(["store_id", "product_id"]).last().reset_index()
    X = latest[feats].fillna(0)
    latest["stockout_prob"] = model.predict_proba(X)[:, 1]
    return latest[["store_id", "product_id", "stockout_prob", "inventory_level"]]


if __name__ == "__main__":
    df = load_feature_data()
    train_stockout_model(df)
