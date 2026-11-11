
"""Calculate prediction metrics and evaluate uplift on real sales."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import TARGET


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def _real_uplift(df):
    """Calculate the sales difference between promoted and non-promoted rows."""
    promoted_sales = df.loc[df["promoted"] == 1, TARGET].mean()
    non_promoted_sales = df.loc[df["promoted"] == 0, TARGET].mean()
    return promoted_sales - non_promoted_sales


def real_uplift_by_decile(pred_uplift, df):
    tmp = df[[TARGET, "promoted"]].copy()
    tmp["pred_uplift"] = np.asarray(pred_uplift)

    tmp["decile"] = pd.qcut(
        tmp["pred_uplift"].rank(method="first"),
        10,
        labels=range(1, 11),
    )

    means = tmp.groupby(
        ["decile", "promoted"], observed=True
    )[TARGET].mean().unstack()

    return means[1] - means[0]


def real_uplift_top_share(pred_uplift, df, shares=range(10, 101, 10)):
    tmp = df[[TARGET, "promoted"]].copy()
    tmp["pred_uplift"] = np.asarray(pred_uplift)
    tmp = tmp.sort_values("pred_uplift", ascending=False)

    return pd.Series({
        share: _real_uplift(tmp.head(int(len(tmp) * share / 100)))
        for share in shares
    })


def compare_methods(methods, train, valid):
    """Fit each method on train and evaluate it on validation data."""
    rows = {}
    uplifts = {}

    for name, method in methods.items():
        method.fit(train)
        uplift = method.uplift(valid)
        top = real_uplift_top_share(uplift, valid)

        train_metrics = regression_metrics(
            train[TARGET], method.predict(train)
        )
        valid_metrics = regression_metrics(
            valid[TARGET], method.predict(valid)
        )

        rows[name] = {
            "MAE_train": train_metrics["MAE"],
            "MAE_valid": valid_metrics["MAE"],
            "RMSE_valid": valid_metrics["RMSE"],
            "uplift_top10": top[10],
            "uplift_top20": top[20],
        }

        uplifts[name] = uplift

    return pd.DataFrame(rows).T, uplifts
