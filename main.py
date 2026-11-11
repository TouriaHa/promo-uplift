"""Run the full pipeline: load -> clean -> modeling table -> compare methods -> final test evaluation."""
import pandas as pd

from src.config import OUTPUT_DIR
from src.data_loader import clean_transactions, load_raw, promotion_flags
from src.evaluation import compare_methods, real_uplift_top_share, regression_metrics
from src.features import add_history_features, modeling_table, regular_pairs, time_split, weekly_sales
from src.models import Baseline, SLearner, TLearner, make_gb, make_rf


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    trans, product, causal = load_raw()
    clean = clean_transactions(trans, product)
    flags = promotion_flags(causal)

    print("Building modeling table...")
    weekly = weekly_sales(clean, flags)
    pairs = regular_pairs(weekly)
    data = add_history_features(modeling_table(weekly, pairs, flags, product))
    train, valid, test = time_split(data)
    print(f"train {len(train)} | valid {len(valid)} | test {len(test)} rows")

    print("Comparing methods on validation...")
    methods = {
        "Baseline": Baseline(),
        "S-learner GB": SLearner(make_gb),
        "T-learner GB": TLearner(make_gb),
        "S-learner RF": SLearner(make_rf),
        "T-learner RF": TLearner(make_rf),
    }
    comparison, _ = compare_methods(methods, train, valid)
    print(comparison.round(3))
    comparison.to_csv(OUTPUT_DIR / "comparison_validation.csv")

    print("Final model (S-learner GB) refit on train + validation, evaluated once on test...")
    train_full = pd.concat([train, valid])
    final, baseline = SLearner(make_gb).fit(train_full), Baseline().fit(train_full)

    test_results = pd.DataFrame({
        "Baseline": {**regression_metrics(test["sales"], baseline.predict(test)),
                     **real_uplift_top_share(baseline.uplift(test), test).add_prefix("uplift_top").to_dict()},
        "S-learner GB": {**regression_metrics(test["sales"], final.predict(test)),
                         **real_uplift_top_share(final.uplift(test), test).add_prefix("uplift_top").to_dict()},
    }).T
    print(test_results.round(3))
    test_results.to_csv(OUTPUT_DIR / "test_results.csv")

    predictions = test[["PRODUCT_ID", "STORE_ID", "WEEK_NO", "DEPARTMENT", "COMMODITY_DESC", "BRAND",
                        "promoted", "sales", "retail_disc"]].copy()
    predictions["pred_uplift"] = final.uplift(test)
    predictions.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()