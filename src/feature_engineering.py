"""
feature_engineering.py

Retail Inventory / Demand Forecasting Hackathon
------------------------------------------------

Input:
    data/processed/master_dataset.csv

Output:
    data/processed/feature_engineered_dataset.csv

Features created:
    - Calendar features
    - Lag demand features
    - Rolling demand features
    - Sales statistics
    - Inventory features
    - Order features
    - Price features
    - Promotion features
    - Weather / seasonality encoding
    - Stockout-related features

IMPORTANT:
    All demand lag and rolling features use only historical data.
    Current/future demand is never used to create forecasting features.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "master_dataset.csv"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "feature_engineered_dataset.csv"
)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

def load_master_dataset():

    print("=" * 70)
    print("LOADING MASTER DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Master dataset not found:\n{INPUT_FILE}\n\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# PREPARE DATA TYPES
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING DATA TYPES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Sort by time and entity
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Date range:",
        df["date"].min(),
        "to",
        df["date"].max()
    )

    return df


# ============================================================
# CALENDAR FEATURES
# ============================================================

def create_calendar_features(df):

    print("\n" + "=" * 70)
    print("CREATING CALENDAR FEATURES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Basic date components
    # --------------------------------------------------------

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["day"] = df["date"].dt.day

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    # --------------------------------------------------------
    # Weekend
    # --------------------------------------------------------

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    # --------------------------------------------------------
    # Month start / end
    # --------------------------------------------------------

    df["is_month_start"] = (
        df["date"]
        .dt.is_month_start
        .astype(int)
    )

    df["is_month_end"] = (
        df["date"]
        .dt.is_month_end
        .astype(int)
    )

    # --------------------------------------------------------
    # Cyclic encoding
    # --------------------------------------------------------

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


# ============================================================
# DEMAND LAG FEATURES
# ============================================================

def create_demand_lag_features(df):

    print("\n" + "=" * 70)
    print("CREATING DEMAND LAG FEATURES")
    print("=" * 70)

    df = df.copy()

    group_columns = [
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # Historical demand lags
    # --------------------------------------------------------

    lag_periods = [
        1,
        2,
        3,
        7,
        14,
        28
    ]

    for lag in lag_periods:

        column_name = (
            f"demand_lag_{lag}"
        )

        df[column_name] = (
            df.groupby(group_columns)["demand"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Units sold lags
    # --------------------------------------------------------

    for lag in [1, 7, 14]:

        column_name = (
            f"units_sold_lag_{lag}"
        )

        df[column_name] = (
            df.groupby(group_columns)["units_sold"]
            .shift(lag)
        )

    return df


# ============================================================
# ROLLING DEMAND FEATURES
# ============================================================

def create_rolling_demand_features(df):

    print("\n" + "=" * 70)
    print("CREATING ROLLING DEMAND FEATURES")
    print("=" * 70)

    df = df.copy()

    group_columns = [
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # shift(1) is applied BEFORE rolling.
    #
    # Therefore today's demand is NOT included in
    # today's historical rolling statistics.
    # --------------------------------------------------------

    historical_demand = (
        df.groupby(group_columns)["demand"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Rolling mean
    # --------------------------------------------------------

    for window in [7, 14, 28]:

        column_name = (
            f"demand_rolling_mean_{window}"
        )

        df[column_name] = (
            historical_demand
            .groupby(
                [
                    df["store_id"],
                    df["product_id"]
                ]
            )
            .transform(
                lambda x:
                x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

    # --------------------------------------------------------
    # Rolling standard deviation
    # --------------------------------------------------------

    for window in [7, 14, 28]:

        column_name = (
            f"demand_rolling_std_{window}"
        )

        df[column_name] = (
            historical_demand
            .groupby(
                [
                    df["store_id"],
                    df["product_id"]
                ]
            )
            .transform(
                lambda x:
                x.rolling(
                    window=window,
                    min_periods=2
                ).std()
            )
        )

    # --------------------------------------------------------
    # Rolling minimum
    # --------------------------------------------------------

    df["demand_rolling_min_7"] = (
        historical_demand
        .groupby(
            [
                df["store_id"],
                df["product_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1
            ).min()
        )
    )

    # --------------------------------------------------------
    # Rolling maximum
    # --------------------------------------------------------

    df["demand_rolling_max_7"] = (
        historical_demand
        .groupby(
            [
                df["store_id"],
                df["product_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1
            ).max()
        )
    )

    return df


# ============================================================
# SALES FEATURES
# ============================================================

def create_sales_features(df):

    print("\n" + "=" * 70)
    print("CREATING SALES FEATURES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Sales / demand difference
    # --------------------------------------------------------

    df["demand_sales_difference"] = (
        df["demand"] -
        df["units_sold"]
    )

    # --------------------------------------------------------
    # Demand to sales ratio
    # --------------------------------------------------------

    df["demand_sales_ratio"] = np.where(
        df["units_sold"] > 0,
        df["demand"] /
        df["units_sold"],
        0
    )

    # --------------------------------------------------------
    # Inventory to demand ratio
    #
    # This is descriptive, not future information.
    # --------------------------------------------------------

    df["inventory_demand_ratio"] = np.where(
        df["demand"] > 0,
        df["inventory_level"] /
        df["demand"],
        0
    )

    return df


# ============================================================
# INVENTORY FEATURES
# ============================================================

def create_inventory_features(df):

    print("\n" + "=" * 70)
    print("CREATING INVENTORY FEATURES")
    print("=" * 70)

    df = df.copy()

    group_columns = [
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # Previous day's inventory
    # --------------------------------------------------------

    df["inventory_lag_1"] = (
        df.groupby(group_columns)
        ["inventory_level"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous order
    # --------------------------------------------------------

    df["units_ordered_lag_1"] = (
        df.groupby(group_columns)
        ["units_ordered"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Inventory change
    # --------------------------------------------------------

    df["inventory_change"] = (
        df["inventory_level"] -
        df["inventory_lag_1"]
    )

    # --------------------------------------------------------
    # Order change
    # --------------------------------------------------------

    df["order_change"] = (
        df["units_ordered"] -
        df["units_ordered_lag_1"]
    )

    # --------------------------------------------------------
    # Inventory turnover proxy
    # --------------------------------------------------------

    df["inventory_turnover_proxy"] = np.where(
        df["inventory_level"] > 0,
        df["units_sold"] /
        df["inventory_level"],
        0
    )

    # --------------------------------------------------------
    # Low inventory indicators
    # --------------------------------------------------------

    df["low_inventory_flag"] = (
        df["inventory_level"] <=
        df["demand_rolling_mean_7"]
    ).astype(int)

    # --------------------------------------------------------
    # Inventory coverage using historical demand
    # --------------------------------------------------------

    df["inventory_days_of_cover"] = np.where(
        df["demand_rolling_mean_7"] > 0,
        df["inventory_level"] /
        df["demand_rolling_mean_7"],
        0
    )

    return df


# ============================================================
# PRICE FEATURES
# ============================================================

def create_price_features(df):

    print("\n" + "=" * 70)
    print("CREATING PRICE FEATURES")
    print("=" * 70)

    df = df.copy()

    group_columns = [
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # Previous price
    # --------------------------------------------------------

    df["price_lag_1"] = (
        df.groupby(group_columns)
        ["price"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Price change
    # --------------------------------------------------------

    df["price_change"] = (
        df["price"] -
        df["price_lag_1"]
    )

    # --------------------------------------------------------
    # Price change percentage
    # --------------------------------------------------------

    df["price_change_pct"] = np.where(
        df["price_lag_1"] > 0,
        (
            df["price"] -
            df["price_lag_1"]
        ) /
        df["price_lag_1"] * 100,
        0
    )

    # --------------------------------------------------------
    # Price difference from competitor
    # --------------------------------------------------------

    df["competitor_price_difference"] = (
        df["price"] -
        df["competitor_pricing"]
    )

    # --------------------------------------------------------
    # Relative price to competitor
    # --------------------------------------------------------

    df["price_vs_competitor_ratio"] = np.where(
        df["competitor_pricing"] > 0,
        df["price"] /
        df["competitor_pricing"],
        1
    )

    return df


# ============================================================
# PROMOTION FEATURES
# ============================================================

def create_promotion_features(df):

    print("\n" + "=" * 70)
    print("CREATING PROMOTION FEATURES")
    print("=" * 70)

    df = df.copy()

    group_columns = [
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # Previous promotion
    # --------------------------------------------------------

    df["promotion_lag_1"] = (
        df.groupby(group_columns)
        ["promotion"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Promotion change
    # --------------------------------------------------------

    df["promotion_change"] = (
        df["promotion"] -
        df["promotion_lag_1"]
    )

    # --------------------------------------------------------
    # Discount active
    # --------------------------------------------------------

    df["discount_active"] = (
        df["discount"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Combined promotion / discount
    # --------------------------------------------------------

    df["promotion_discount_active"] = (
        (
            df["promotion"] == 1
        ) &
        (
            df["discount"] > 0
        )
    ).astype(int)

    return df


# ============================================================
# WEATHER / SEASONAL FEATURES
# ============================================================

def create_external_features(df):

    print("\n" + "=" * 70)
    print("CREATING EXTERNAL FACTOR FEATURES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Weather one-hot encoding
    # --------------------------------------------------------

    weather_dummies = pd.get_dummies(
        df["weather_condition"],
        prefix="weather",
        dtype=int
    )

    df = pd.concat(
        [
            df,
            weather_dummies
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Seasonality one-hot encoding
    # --------------------------------------------------------

    season_dummies = pd.get_dummies(
        df["seasonality"],
        prefix="season",
        dtype=int
    )

    df = pd.concat(
        [
            df,
            season_dummies
        ],
        axis=1
    )

    return df


# ============================================================
# STOCKOUT FEATURES
# ============================================================

def create_stockout_features(df):

    print("\n" + "=" * 70)
    print("CREATING STOCKOUT-RELATED FEATURES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is NOT claiming to be an observed stockout label.
    #
    # It is a proxy based on available inventory and demand.
    # --------------------------------------------------------

    df["inventory_below_demand_flag"] = (
        df["inventory_level"] <
        df["demand"]
    ).astype(int)

    # --------------------------------------------------------
    # Inventory shortage amount
    # --------------------------------------------------------

    df["inventory_shortage_amount"] = np.maximum(
        df["demand"] -
        df["inventory_level"],
        0
    )

    # --------------------------------------------------------
    # Inventory utilization
    # --------------------------------------------------------

    df["inventory_utilization"] = np.where(
        df["inventory_level"] > 0,
        df["units_sold"] /
        df["inventory_level"],
        0
    )

    # --------------------------------------------------------
    # Historical low inventory
    # --------------------------------------------------------

    df["historical_low_inventory"] = (
        df["inventory_level"] <
        df["demand_rolling_mean_7"]
    ).astype(int)

    return df


# ============================================================
# CLEAN FEATURE VALUES
# ============================================================

def clean_feature_values(df):

    print("\n" + "=" * 70)
    print("CLEANING FEATURE VALUES")
    print("=" * 70)

    df = df.copy()

    # Replace infinite values

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
    )

    # --------------------------------------------------------
    # Fill rolling statistics
    #
    # Early rows naturally don't have enough history.
    # We use zero here so the dataset remains usable.
    # --------------------------------------------------------

    for column in numeric_columns:

        if (
            column.startswith(
                "demand_rolling_"
            )
            or column.startswith(
                "demand_lag_"
            )
            or column.startswith(
                "units_sold_lag_"
            )
            or column.startswith(
                "inventory_lag_"
            )
            or column.startswith(
                "units_ordered_lag_"
            )
            or column.startswith(
                "price_lag_"
            )
            or column.startswith(
                "promotion_lag_"
            )
        ):

            df[column] = (
                df[column]
                .fillna(0)
            )

    return df


# ============================================================
# REMOVE FIRST-HISTORY ROWS
# ============================================================

def create_model_ready_dataset(
    df
):

    print("\n" + "=" * 70)
    print("CREATING MODEL-READY DATASET")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Since 28-day historical features are created,
    # the first 28 days do not have complete history.
    #
    # Remove them for models that require complete lag history.
    # --------------------------------------------------------

    max_lag = 28

    group_columns = [
        "store_id",
        "product_id"
    ]

    df["history_index"] = (
        df.groupby(group_columns)
        .cumcount()
    )

    model_ready = df[
        df["history_index"] >= max_lag
    ].copy()

    model_ready = model_ready.drop(
        columns=["history_index"]
    )

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Model-ready rows: "
        f"{len(model_ready):,}"
    )

    print(
        f"Rows removed: "
        f"{len(df) - len(model_ready):,}"
    )

    return model_ready


# ============================================================
# VALIDATE FEATURE DATASET
# ============================================================

def validate_feature_dataset(
    df
):

    print("\n" + "=" * 70)
    print("VALIDATING FEATURE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print(
        "Shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_keys = (
        df.duplicated(
            subset=[
                "date",
                "store_id",
                "product_id"
            ]
        )
        .sum()
    )

    print(
        "Duplicate Date-Store-Product:",
        duplicate_keys
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_count = (
        df.isnull()
        .sum()
        .sum()
    )

    print(
        "Total missing values:",
        missing_count
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        "Infinite values:",
        infinite_count
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    print(
        "Total columns:",
        len(df.columns)
    )

    # --------------------------------------------------------
    # Demand statistics
    # --------------------------------------------------------

    if "demand" in df.columns:

        print(
            "\nDemand statistics:"
        )

        print(
            df["demand"].describe()
        )


# ============================================================
# SAVE DATASET
# ============================================================

def save_feature_dataset(
    df
):

    print("\n" + "=" * 70)
    print("SAVING FEATURE ENGINEERED DATASET")
    print("=" * 70)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering():

    print("\n")
    print("=" * 70)
    print("RETAIL DEMAND FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_master_dataset()

    # --------------------------------------------------------
    # 2. Prepare
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # 3. Calendar
    # --------------------------------------------------------

    df = create_calendar_features(
        df
    )

    # --------------------------------------------------------
    # 4. Demand history
    # --------------------------------------------------------

    df = create_demand_lag_features(
        df
    )

    # --------------------------------------------------------
    # 5. Rolling demand
    # --------------------------------------------------------

    df = create_rolling_demand_features(
        df
    )

    # --------------------------------------------------------
    # 6. Sales
    # --------------------------------------------------------

    df = create_sales_features(
        df
    )

    # --------------------------------------------------------
    # 7. Inventory
    # --------------------------------------------------------

    df = create_inventory_features(
        df
    )

    # --------------------------------------------------------
    # 8. Price
    # --------------------------------------------------------

    df = create_price_features(
        df
    )

    # --------------------------------------------------------
    # 9. Promotion
    # --------------------------------------------------------

    df = create_promotion_features(
        df
    )

    # --------------------------------------------------------
    # 10. External factors
    # --------------------------------------------------------

    df = create_external_features(
        df
    )

    # --------------------------------------------------------
    # 11. Stockout features
    # --------------------------------------------------------

    df = create_stockout_features(
        df
    )

    # --------------------------------------------------------
    # 12. Clean feature values
    # --------------------------------------------------------

    df = clean_feature_values(
        df
    )

    # --------------------------------------------------------
    # 13. Model-ready dataset
    # --------------------------------------------------------

    df = create_model_ready_dataset(
        df
    )

    # --------------------------------------------------------
    # 14. Final validation
    # --------------------------------------------------------

    validate_feature_dataset(
        df
    )

    # --------------------------------------------------------
    # 15. Save
    # --------------------------------------------------------

    save_feature_dataset(
        df
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(
        "\nFinal dataset:"
    )

    print(
        df.head()
    )

    print(
        "\nFeature columns:"
    )

    for column in df.columns:
        print(
            " -",
            column
        )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    feature_engineered_data = (
        run_feature_engineering()
    )