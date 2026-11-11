
"""Prepare the weekly dataset and create history features."""

import pandas as pd

from src.config import (
    CAT_FEATURES, FIRST_PROMO_WEEK, KEYS, LAST_WEEK,
    MIN_SELLING_WEEKS, SPLIT_WEEK, TRAIN_START, VALID_WEEK
)


def weekly_sales(clean, promo_flags):
    # Aggregate transactions by product, store and week
    weekly = (
        clean.groupby(KEYS)
        .agg(
            sales=("SALES_VALUE", "sum"),
            quantity=("QUANTITY", "sum"),
            retail_disc=("RETAIL_DISC", "sum"),
            n_baskets=("BASKET_ID", "nunique")
        )
        .reset_index()
    )

    return weekly[weekly["STORE_ID"].isin(promo_flags["STORE_ID"].unique())]


def regular_pairs(weekly):
    # Keep pairs with enough sales history before validation
    past = weekly[weekly["WEEK_NO"] < VALID_WEEK]
    n_weeks = past.groupby(["PRODUCT_ID", "STORE_ID"])["WEEK_NO"].nunique()

    return n_weeks[n_weeks >= MIN_SELLING_WEEKS].index


def modeling_table(weekly, pairs, promo_flags, product):
    # Create all product-store-week combinations
    weeks = pd.DataFrame({"WEEK_NO": range(1, LAST_WEEK + 1)})
    grid = pairs.to_frame(index=False).merge(weeks, how="cross")

    data = grid.merge(weekly, on=KEYS, how="left")

    sales_cols = ["sales", "quantity", "retail_disc", "n_baskets"]
    data[sales_cols] = data[sales_cols].fillna(0)

    # Add promotion information
    n_before = len(data)
    data = data.merge(promo_flags, on=KEYS, how="left")
    assert len(data) == n_before, "Promotion merge duplicated rows"

    data[["on_display", "in_mailer"]] = (
        data[["on_display", "in_mailer"]].fillna(0).astype(int)
    )
    data["promoted"] = (
        (data["on_display"] == 1) | (data["in_mailer"] == 1)
    ).astype(int)

    # Add product information
    data = data.merge(
        product[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "BRAND"]],
        on="PRODUCT_ID",
        how="left"
    )
    assert len(data) == n_before, "Product merge duplicated rows"

    return data


def add_history_features(data):
    # Calculate features using previous weeks only
    data = data[data["WEEK_NO"] >= FIRST_PROMO_WEEK]
    data = data.sort_values(KEYS).reset_index(drop=True)

    g = data.groupby(["PRODUCT_ID", "STORE_ID"])

    data["sales_lag1"] = g["sales"].shift(1)
    data["sales_lag2"] = g["sales"].shift(2)
    data["sales_avg4"] = g["sales"].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )
    data["sales_avg_past"] = g["sales"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    data["promo_lag1"] = g["promoted"].shift(1)
    data["promo_count4"] = g["promoted"].transform(
        lambda s: s.shift(1).rolling(4).sum()
    )

    return data[data["WEEK_NO"] >= TRAIN_START].reset_index(drop=True)


def time_split(data):
    # Split by week to preserve the time order
    data = data.copy()

    for col in CAT_FEATURES:
        data[col] = pd.Categorical(
            data[col],
            categories=sorted(data[col].dropna().unique())
        )

    train = data[data["WEEK_NO"] < VALID_WEEK]
    valid = data[
        (data["WEEK_NO"] >= VALID_WEEK) &
        (data["WEEK_NO"] < SPLIT_WEEK)
    ]
    test = data[data["WEEK_NO"] >= SPLIT_WEEK]

    return train, valid, test
