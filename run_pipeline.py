"""
run_pipeline.py  –  Full StockSense pipeline
Run:  python run_pipeline.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from feature_engineering import run_feature_engineering
from demand_model import train_demand_model, load_feature_data as load_demand
from stockout_model import train_stockout_model, load_feature_data as load_stockout
from recommendation import build_recommendation_table
import pandas as pd

PROCESSED_DIR = BASE_DIR / "data" / "processed"


def run():
    print("=" * 60)
    print("STOCKSENSE PIPELINE")
    print("=" * 60)

    # 1. Feature engineering
    print("\n[1/4] Feature Engineering...")
    run_feature_engineering()

    # 2. Demand model
    print("\n[2/4] Training Demand Model...")
    df = load_demand()
    model_d, feats_d, metrics_d = train_demand_model(df)

    # 3. Stockout model
    print("\n[3/4] Training Stockout Model...")
    df2 = load_stockout()
    model_s, feats_s, metrics_s = train_stockout_model(df2)

    # 4. Recommendations
    print("\n[4/4] Generating Recommendations...")
    feature_df = pd.read_csv(PROCESSED_DIR / "feature_engineered_dataset.csv",
                             parse_dates=["date"])
    build_recommendation_table(feature_df)

    # Save combined metrics
    rows = [
        {"model": "Demand", "metric": k, "value": v}
        for k, v in metrics_d.items() if k != "CM"
    ] + [
        {"model": "Stockout", "metric": k, "value": v}
        for k, v in metrics_s.items() if k != "CM"
    ]
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)

    print("\nPipeline complete. Launch dashboard:")
    print("   streamlit run dashboard/app.py")


if __name__ == "__main__":
    run()
