"""Project configuration: paths, features and model parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

# Departments excluded from the analysis
NON_MERCH_DEPARTMENTS = [
    "KIOSK-GAS", "MISC SALES TRAN", "MISC. TRANS.",
    "COUP/STR & MFG", "TRAVEL & LEISUR", ""
]

KEYS = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]

# Study period
FIRST_PROMO_WEEK = 9
LAST_WEEK = 101
TRAIN_START = 13
VALID_WEEK = 72
SPLIT_WEEK = 87

# Minimum number of selling weeks
MIN_SELLING_WEEKS = 15

TARGET = "sales"

CAT_FEATURES = ["DEPARTMENT", "BRAND"]

NUM_FEATURES = [
    "promoted", "sales_lag1", "sales_lag2", "sales_avg4",
    "sales_avg_past", "promo_lag1", "promo_count4"
]

FEATURES = NUM_FEATURES + CAT_FEATURES

RANDOM_STATE = 42

GB_PARAMS = dict(
    loss="poisson",
    max_iter=300,
    learning_rate=0.05,
    categorical_features="from_dtype",
    early_stopping=False,
    random_state=RANDOM_STATE
)

RF_PARAMS = dict(
    n_estimators=100,
    min_samples_leaf=50,
    max_samples=0.3,
    n_jobs=-1,
    random_state=RANDOM_STATE
)