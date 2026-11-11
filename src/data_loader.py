
"""Load and clean the Dunnhumby data."""

import pandas as pd

from src.config import DATA_DIR, KEYS, NON_MERCH_DEPARTMENTS


def load_raw(data_dir=DATA_DIR):
    trans = pd.read_csv(data_dir / "transaction_data.csv")
    product = pd.read_csv(data_dir / "product.csv")

    causal = pd.read_csv(
        data_dir / "causal_data.csv",
        dtype={
            "PRODUCT_ID": "int32",
            "STORE_ID": "int32",
            "WEEK_NO": "int8",
            "display": "category",
            "mailer": "category"
        }
    )

    return trans, product, causal


def clean_transactions(trans, product):
    # Keep merchandise products with valid sales
    dept = product["DEPARTMENT"].str.strip()
    non_merch = product.loc[
        dept.isin(NON_MERCH_DEPARTMENTS), "PRODUCT_ID"
    ]

    clean = trans[~trans["PRODUCT_ID"].isin(non_merch)]
    clean = clean[(clean["QUANTITY"] > 0) & (clean["SALES_VALUE"] > 0)]
    clean = clean[clean["RETAIL_DISC"] <= 0]

    return clean


def promotion_flags(causal):
    # Create binary promotion flags and combine duplicate records
    flags = causal[KEYS].copy()
    flags["on_display"] = (causal["display"] != "0").astype("int8")
    flags["in_mailer"] = (causal["mailer"] != "0").astype("int8")

    return flags.groupby(KEYS, as_index=False)[
        ["on_display", "in_mailer"]
    ].max()
