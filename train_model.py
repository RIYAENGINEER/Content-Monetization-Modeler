# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, MODEL_DIR, RANDOM_STATE, TEST_SIZE
from src.data_load import load_data
from src.preprocess import (
    remove_duplicates,
    handle_missing,
    basic_feature_engineering,
    build_preprocessor,
)

# ensure models dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed_data.csv")
RANDOM_STATE = RANDOM_STATE if RANDOM_STATE is not None else 42
TEST_SIZE = TEST_SIZE if TEST_SIZE is not None else 0.2

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
    }

def load_preprocessed_or_build():
    """
    Load preprocessed CSV if present, otherwise load raw CSV and preprocess it.
    Returns dataframe (preprocessed).
    """
    if os.path.exists(PREPROCESSED_PATH):
        print(f"Loading preprocessed data from: {PREPROCESSED_PATH}")
        df = pd.read_csv(PREPROCESSED_PATH)
        return df
    else:
        print("Preprocessed file not found — loading raw CSV and preprocessing now.")
        df = load_data()
        df = remove_duplicates(df)
        df = handle_missing(df)
        df = basic_feature_engineering(df)
        # Save copy for next runs
        out_path = os.path.join(PROJECT_ROOT, "data", "preprocessed_data.csv")
        df.to_csv(out_path, index=False)
        print("Saved preprocessed data to:", out_path)
        return df

def train_and_save():
    # 1. Load preprocessed data (or build it)
    df = load_preprocessed_or_build()

    # 2. Validate target
    target = "ad_revenue_usd"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {df.columns.tolist()}")

    # 3. Features / target
    features = [c for c in df.columns if c != target]
    if len(features) == 0:
        raise ValueError("No features found to train on.")

    X = df[features]
    y = df[target]

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 5. Build preprocessor (detect numeric/categorical using df)
    preprocessor, num_cols, cat_cols = build_preprocessor(df, features)
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # 6. Candidate models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    # 7. Train & evaluate
    for name, estimator in models.items():
        print(f"\nTraining {name} ...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        metrics = eval_metrics(y_test, preds)
        results[name] = metrics
        print(f"{name} metrics: {metrics}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_pipeline = pipe

    # 8. Save best model and metrics
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best model: {best_name} -> {model_path}")

    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"))
    print(f"Saved metrics to: {os.path.join(MODEL_DIR, 'metrics.csv')}")

    return results, best_name

if __name__ == "__main__":
    res, best = train_and_save()
    print("\nAll done. Best model:", best)
    print(res)
