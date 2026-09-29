"""
preprocessing.py

Retail Inventory / Demand Forecasting Hackathon
------------------------------------------------

Responsibilities:
1. Load raw datasets
2. Standardize column names
3. Clean and validate datasets
4. Validate Date-Store-Product grain
5. Merge datasets into master dataset
6. Perform data-quality checks
7. Save processed datasets

Expected project structure:

Ds_Hackathon/
│
├── data/
│   ├── raw/
│   │   ├── transactions.csv
│   │   ├── products.csv
│   │   ├── stores.csv
│   │   ├── inventory.csv
│   │   └── external_factors.csv
│   │
│   └── processed/
│
├── src/
│   └── preprocessing.py
│
├── notebooks/
│
└── README.md
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

# preprocessing.py is located in:
# Ds_Hackathon/src/preprocessing.py
#
# Therefore:
# parent      -> src
# parent.parent -> Ds_Hackathon

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED FILES
# ============================================================

TRANSACTIONS_FILE = RAW_DIR / "transactions.csv"
PRODUCTS_FILE = RAW_DIR / "products.csv"
STORES_FILE = RAW_DIR / "stores.csv"
INVENTORY_FILE = RAW_DIR / "inventory.csv"
EXTERNAL_FACTORS_FILE = RAW_DIR / "external_factors.csv"


# ============================================================
# COMMON UTILITIES
# ============================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.

    Example:
        Store ID -> store_id
        Product ID -> product_id
        Units Sold -> units_sold
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list,
    dataset_name: str
):
    """
    Check whether all required columns exist.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def remove_duplicate_rows(
    df: pd.DataFrame,
    dataset_name: str
) -> pd.DataFrame:
    """
    Remove completely duplicated rows.
    """

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"{dataset_name}: removed "
        f"{before - after:,} duplicate rows"
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load all five raw datasets.
    """

    required_files = [
        TRANSACTIONS_FILE,
        PRODUCTS_FILE,
        STORES_FILE,
        INVENTORY_FILE,
        EXTERNAL_FACTORS_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    print("=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    products = pd.read_csv(
        PRODUCTS_FILE
    )

    stores = pd.read_csv(
        STORES_FILE
    )

    inventory = pd.read_csv(
        INVENTORY_FILE
    )

    external_factors = pd.read_csv(
        EXTERNAL_FACTORS_FILE
    )

    print(
        f"Transactions      : {transactions.shape}"
    )

    print(
        f"Products           : {products.shape}"
    )

    print(
        f"Stores             : {stores.shape}"
    )

    print(
        f"Inventory          : {inventory.shape}"
    )

    print(
        f"External factors   : {external_factors.shape}"
    )

    return (
        transactions,
        products,
        stores,
        inventory,
        external_factors
    )


# ============================================================
# CLEAN TRANSACTIONS
# ============================================================

def clean_transactions(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CLEANING TRANSACTIONS")
    print("=" * 70)

    df = standardize_columns(df)

    required_columns = [
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "demand",
        "price",
        "discount",
        "promotion",
        "competitor_pricing"
    ]

    check_required_columns(
        df,
        required_columns,
        "transactions"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "units_sold",
        "demand",
        "price",
        "discount",
        "promotion",
        "competitor_pricing"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicate_rows(
        df,
        "Transactions"
    )

    # --------------------------------------------------------
    # Remove missing key values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "store_id",
            "product_id"
        ]
    )

    # --------------------------------------------------------
    # Remove invalid business values
    # --------------------------------------------------------

    df = df[
        df["units_sold"] >= 0
    ]

    df = df[
        df["demand"] >= 0
    ]

    df = df[
        df["price"] > 0
    ]

    df = df[
        df["discount"].between(
            0,
            100
        )
    ]

    df = df[
        df["promotion"].isin(
            [0, 1]
        )
    ]

    df = df[
        df["competitor_pricing"] > 0
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "date",
            "store_id",
            "product_id"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# CLEAN INVENTORY
# ============================================================

def clean_inventory(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CLEANING INVENTORY")
    print("=" * 70)

    df = standardize_columns(df)

    required_columns = [
        "date",
        "store_id",
        "product_id",
        "inventory_level",
        "units_ordered"
    ]

    check_required_columns(
        df,
        required_columns,
        "inventory"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    df["inventory_level"] = pd.to_numeric(
        df["inventory_level"],
        errors="coerce"
    )

    df["units_ordered"] = pd.to_numeric(
        df["units_ordered"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicate_rows(
        df,
        "Inventory"
    )

    # --------------------------------------------------------
    # Remove missing keys
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "store_id",
            "product_id"
        ]
    )

    # --------------------------------------------------------
    # Invalid values
    # --------------------------------------------------------

    df = df[
        df["inventory_level"] >= 0
    ]

    df = df[
        df["units_ordered"] >= 0
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "date",
            "store_id",
            "product_id"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# CLEAN PRODUCTS
# ============================================================

def clean_products(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CLEANING PRODUCTS")
    print("=" * 70)

    df = standardize_columns(df)

    required_columns = [
        "product_id",
        "observed_categories",
        "category_counts",
        "category_consistent"
    ]

    check_required_columns(
        df,
        required_columns,
        "products"
    )

    df = remove_duplicate_rows(
        df,
        "Products"
    )

    df = df.dropna(
        subset=["product_id"]
    )

    df["product_id"] = (
        df["product_id"]
        .astype(str)
        .str.strip()
    )

    df = df.sort_values(
        "product_id"
    ).reset_index(drop=True)

    return df


# ============================================================
# CLEAN STORES
# ============================================================

def clean_stores(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CLEANING STORES")
    print("=" * 70)

    df = standardize_columns(df)

    required_columns = [
        "store_id",
        "region"
    ]

    check_required_columns(
        df,
        required_columns,
        "stores"
    )

    df = remove_duplicate_rows(
        df,
        "Stores"
    )

    df = df.dropna(
        subset=["store_id"]
    )

    df["store_id"] = (
        df["store_id"]
        .astype(str)
        .str.strip()
    )

    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Check that one store has one region
    # --------------------------------------------------------

    region_counts = (
        df.groupby("store_id")["region"]
        .nunique()
    )

    inconsistent_stores = (
        region_counts[
            region_counts > 1
        ]
    )

    if len(inconsistent_stores) > 0:

        print(
            "WARNING: Some stores have multiple regions:"
        )

        print(
            inconsistent_stores
        )

    df = df.sort_values(
        "store_id"
    ).reset_index(drop=True)

    return df


# ============================================================
# CLEAN EXTERNAL FACTORS
# ============================================================

def clean_external_factors(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CLEANING EXTERNAL FACTORS")
    print("=" * 70)

    df = standardize_columns(df)

    required_columns = [
        "date",
        "store_id",
        "weather_condition",
        "seasonality",
        "epidemic"
    ]

    check_required_columns(
        df,
        required_columns,
        "external_factors"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["epidemic"] = pd.to_numeric(
        df["epidemic"],
        errors="coerce"
    )

    df = remove_duplicate_rows(
        df,
        "External factors"
    )

    df = df.dropna(
        subset=[
            "date",
            "store_id"
        ]
    )

    df["store_id"] = (
        df["store_id"]
        .astype(str)
        .str.strip()
    )

    df["weather_condition"] = (
        df["weather_condition"]
        .astype(str)
        .str.strip()
    )

    df["seasonality"] = (
        df["seasonality"]
        .astype(str)
        .str.strip()
    )

    # Epidemic is expected to be binary
    df = df[
        df["epidemic"].isin(
            [0, 1]
        )
    ]

    df = df.sort_values(
        [
            "date",
            "store_id"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDATE TRANSACTION GRAIN
# ============================================================

def validate_transaction_grain(
    transactions: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("VALIDATING TRANSACTION GRAIN")
    print("=" * 70)

    keys = [
        "date",
        "store_id",
        "product_id"
    ]

    duplicate_keys = (
        transactions
        .duplicated(
            subset=keys
        )
        .sum()
    )

    print(
        f"Duplicate Date-Store-Product keys: "
        f"{duplicate_keys}"
    )

    if duplicate_keys > 0:

        raise ValueError(
            "Transactions contain duplicate "
            "Date-Store-Product combinations."
        )

    print(
        "Transaction grain is valid:"
    )

    print(
        "One row = one Date × Store × Product"
    )


# ============================================================
# VALIDATE INVENTORY GRAIN
# ============================================================

def validate_inventory_grain(
    inventory: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("VALIDATING INVENTORY GRAIN")
    print("=" * 70)

    keys = [
        "date",
        "store_id",
        "product_id"
    ]

    duplicate_keys = (
        inventory
        .duplicated(
            subset=keys
        )
        .sum()
    )

    print(
        f"Duplicate Date-Store-Product keys: "
        f"{duplicate_keys}"
    )

    if duplicate_keys > 0:

        raise ValueError(
            "Inventory contains duplicate "
            "Date-Store-Product combinations."
        )


# ============================================================
# VALIDATE EXTERNAL FACTORS GRAIN
# ============================================================

def validate_external_factors(
    external_factors: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("VALIDATING EXTERNAL FACTORS")
    print("=" * 70)

    keys = [
        "date",
        "store_id"
    ]

    duplicate_keys = (
        external_factors
        .duplicated(
            subset=keys
        )
        .sum()
    )

    print(
        f"Duplicate Date-Store keys: "
        f"{duplicate_keys}"
    )

    if duplicate_keys > 0:

        raise ValueError(
            "External factors contain duplicate "
            "Date-Store combinations."
        )


# ============================================================
# PRODUCT CATEGORY QUALITY CHECK
# ============================================================

def check_product_categories(
    products: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("PRODUCT CATEGORY QUALITY CHECK")
    print("=" * 70)

    inconsistent = products[
        products["category_consistent"]
        .astype(str)
        .str.lower()
        .eq("no")
    ]

    print(
        f"Total products: {len(products)}"
    )

    print(
        f"Products with inconsistent categories: "
        f"{len(inconsistent)}"
    )

    if len(inconsistent) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some Product IDs have multiple observed "
            "categories in the source data."
        )

        print(
            "No artificial category has been assigned "
            "during preprocessing."
        )


# ============================================================
# MERGE DATASETS
# ============================================================

def create_master_dataset(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    inventory: pd.DataFrame,
    external_factors: pd.DataFrame
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("CREATING MASTER DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Start with transactions
    # --------------------------------------------------------

    master = transactions.copy()

    # --------------------------------------------------------
    # Merge inventory
    # --------------------------------------------------------

    master = master.merge(
        inventory,
        on=[
            "date",
            "store_id",
            "product_id"
        ],
        how="left",
        validate="one_to_one"
    )

    print(
        "After inventory merge:",
        master.shape
    )

    # --------------------------------------------------------
    # Merge stores
    # --------------------------------------------------------

    master = master.merge(
        stores,
        on="store_id",
        how="left",
        validate="many_to_one"
    )

    print(
        "After stores merge:",
        master.shape
    )

    # --------------------------------------------------------
    # Merge external factors
    # --------------------------------------------------------

    master = master.merge(
        external_factors,
        on=[
            "date",
            "store_id"
        ],
        how="left",
        validate="many_to_one"
    )

    print(
        "After external factors merge:",
        master.shape
    )

    # --------------------------------------------------------
    # Products
    #
    # IMPORTANT:
    # products.csv contains observed category information,
    # but Product ID -> Category is not consistent.
    #
    # Therefore we DO NOT merge an artificial single
    # category into the master dataset here.
    # --------------------------------------------------------

    print(
        "\nProduct category information was not merged "
        "because category consistency is not guaranteed."
    )

    return master


# ============================================================
# MASTER DATASET VALIDATION
# ============================================================

def validate_master_dataset(
    master: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("MASTER DATASET VALIDATION")
    print("=" * 70)

    keys = [
        "date",
        "store_id",
        "product_id"
    ]

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"Master rows: {len(master):,}"
    )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_keys = (
        master
        .duplicated(
            subset=keys
        )
        .sum()
    )

    print(
        f"Duplicate master keys: "
        f"{duplicate_keys}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    total_missing = (
        master
        .isnull()
        .sum()
        .sum()
    )

    print(
        f"Total missing values: "
        f"{total_missing:,}"
    )

    # --------------------------------------------------------
    # Detailed missing report
    # --------------------------------------------------------

    missing_report = pd.DataFrame({
        "missing_count": master.isnull().sum(),
        "missing_percentage": (
            master.isnull().mean() * 100
        ).round(2)
    })

    missing_report = (
        missing_report[
            missing_report["missing_count"] > 0
        ]
        .sort_values(
            "missing_count",
            ascending=False
        )
    )

    if len(missing_report) > 0:

        print("\nMissing value report:")

        print(
            missing_report
        )

    else:

        print(
            "No missing values found."
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    print(
        "\nDate range:"
    )

    print(
        master["date"].min(),
        "to",
        master["date"].max()
    )

    # --------------------------------------------------------
    # Cardinality
    # --------------------------------------------------------

    print(
        "\nUnique stores:",
        master["store_id"].nunique()
    )

    print(
        "Unique products:",
        master["product_id"].nunique()
    )

    print(
        "Unique dates:",
        master["date"].nunique()
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if duplicate_keys > 0:

        raise ValueError(
            "Master dataset contains duplicate "
            "Date-Store-Product combinations."
        )


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def create_data_quality_report(
    master: pd.DataFrame
) -> pd.DataFrame:

    report = pd.DataFrame({
        "column": master.columns,
        "dtype": [
            str(dtype)
            for dtype in master.dtypes
        ],
        "rows": len(master),
        "missing_count": [
            master[column].isnull().sum()
            for column in master.columns
        ],
        "missing_percentage": [
            round(
                master[column].isnull().mean() * 100,
                2
            )
            for column in master.columns
        ],
        "unique_values": [
            master[column].nunique()
            for column in master.columns
        ]
    })

    return report


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    inventory: pd.DataFrame,
    external_factors: pd.DataFrame,
    master: pd.DataFrame
):

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Individual cleaned datasets
    # --------------------------------------------------------

    transactions.to_csv(
        PROCESSED_DIR /
        "transactions_clean.csv",
        index=False
    )

    products.to_csv(
        PROCESSED_DIR /
        "products_clean.csv",
        index=False
    )

    stores.to_csv(
        PROCESSED_DIR /
        "stores_clean.csv",
        index=False
    )

    inventory.to_csv(
        PROCESSED_DIR /
        "inventory_clean.csv",
        index=False
    )

    external_factors.to_csv(
        PROCESSED_DIR /
        "external_factors_clean.csv",
        index=False
    )

    # --------------------------------------------------------
    # Master dataset
    # --------------------------------------------------------

    master.to_csv(
        PROCESSED_DIR /
        "master_dataset.csv",
        index=False
    )

    # --------------------------------------------------------
    # Data quality report
    # --------------------------------------------------------

    quality_report = (
        create_data_quality_report(
            master
        )
    )

    quality_report.to_csv(
        PROCESSED_DIR /
        "data_quality_report.csv",
        index=False
    )

    print(
        "\nFiles saved to:"
    )

    print(
        PROCESSED_DIR
    )

    for file in sorted(
        PROCESSED_DIR.iterdir()
    ):

        print(
            " -",
            file.name
        )


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing():

    print("\n")
    print("=" * 70)
    print("RETAIL INVENTORY DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    (
        transactions,
        products,
        stores,
        inventory,
        external_factors
    ) = load_data()

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    transactions = clean_transactions(
        transactions
    )

    products = clean_products(
        products
    )

    stores = clean_stores(
        stores
    )

    inventory = clean_inventory(
        inventory
    )

    external_factors = clean_external_factors(
        external_factors
    )

    # --------------------------------------------------------
    # 3. Validate individual datasets
    # --------------------------------------------------------

    validate_transaction_grain(
        transactions
    )

    validate_inventory_grain(
        inventory
    )

    validate_external_factors(
        external_factors
    )

    check_product_categories(
        products
    )

    # --------------------------------------------------------
    # 4. Create master dataset
    # --------------------------------------------------------

    master = create_master_dataset(
        transactions,
        products,
        stores,
        inventory,
        external_factors
    )

    # --------------------------------------------------------
    # 5. Validate master dataset
    # --------------------------------------------------------

    validate_master_dataset(
        master
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    save_processed_data(
        transactions,
        products,
        stores,
        inventory,
        external_factors,
        master
    )

    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nFinal master dataset: "
        f"{master.shape[0]:,} rows × "
        f"{master.shape[1]} columns"
    )

    print(
        "\nMaster columns:"
    )

    for column in master.columns:
        print(
            f" - {column}"
        )

    return master


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    master_dataset = run_preprocessing()