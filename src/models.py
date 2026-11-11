
"""Uplift models: baseline, S-learner and T-learner."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    CAT_FEATURES, FEATURES, GB_PARAMS, NUM_FEATURES, RF_PARAMS, TARGET
)


def make_gb():
    return HistGradientBoostingRegressor(**GB_PARAMS)


def make_rf():
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), CAT_FEATURES)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(**RF_PARAMS))
    ])


def _with_promo(df, value):
    X = df[FEATURES].copy()
    X["promoted"] = value
    return X


class SLearner:
    """One model trained with promotion as an input feature."""

    def __init__(self, make_model):
        self.make_model = make_model

    def fit(self, df):
        self.model = self.make_model().fit(df[FEATURES], df[TARGET])
        return self

    def predict(self, df):
        return self.model.predict(df[FEATURES])

    def uplift(self, df):
        pred_promo = self.model.predict(_with_promo(df, 1))
        pred_no_promo = self.model.predict(_with_promo(df, 0))
        return pred_promo - pred_no_promo


class TLearner:
    """Separate models for promoted and non-promoted weeks."""

    def __init__(self, make_model):
        self.make_model = make_model

    def fit(self, df):
        promo = df[df["promoted"] == 1]
        no_promo = df[df["promoted"] == 0]

        self.model_promo = self.make_model().fit(promo[FEATURES], promo[TARGET])
        self.model_no = self.make_model().fit(no_promo[FEATURES], no_promo[TARGET])

        return self

    def predict(self, df):
        pred_promo = self.model_promo.predict(df[FEATURES])
        pred_no_promo = self.model_no.predict(df[FEATURES])

        return np.where(df["promoted"] == 1, pred_promo, pred_no_promo)

    def uplift(self, df):
        pred_promo = self.model_promo.predict(df[FEATURES])
        pred_no_promo = self.model_no.predict(df[FEATURES])

        return pred_promo - pred_no_promo


class Baseline:
    """Estimate uplift from the average difference within each product-store pair."""

    def fit(self, df):
        pair_means = (
            df.groupby(["PRODUCT_ID", "STORE_ID", "promoted"])[TARGET]
            .mean()
            .unstack()
            .dropna()
        )

        self.uplift_rate = pair_means[1].mean() / pair_means[0].mean() - 1
        return self

    def predict(self, df):
        normal_sales = df["sales_avg_past"]

        return np.where(
            df["promoted"] == 1,
            normal_sales * (1 + self.uplift_rate),
            normal_sales
        )

    def uplift(self, df):
        return (df["sales_avg_past"] * self.uplift_rate).to_numpy()
