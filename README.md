# Promotion Uplift: which products to promote, and where?

Predict the incremental sales of a promotion (display / mailer) for each product × store × week, and rank promotions by expected uplift.

**Data:** Dunnhumby – The Complete Journey (Kaggle).
(https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) (Kaggle). Download it and put the CSVs in `data/raw/` (not included in the repo: ~1 GB).

**Run:**
    pip install -r requirements.txt
    python main.py

Results are saved in `outputs/`. The full analysis is in `notebooks/promo_uplift.ipynb`.

**Method:** time-based split (train / validation / test), baseline vs S- and T-learners (gradient boosting, Random Forest), uplift evaluated on real sales.

