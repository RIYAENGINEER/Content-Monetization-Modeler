import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import PROJECT_ROOT

# ---------------------------
# STEP 1: CLEANING
# ---------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from dataset."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"✅ Removed {before - after} duplicate rows.")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values for numeric and categorical features."""
    df = df.copy()

    # Drop rows where target is missing (we cannot train without revenue)
    if "ad_revenue_usd" in df.columns:
        df = df.dropna(subset=["ad_revenue_usd"])

    # Fill numeric NaN with 0 (or median if you prefer)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill categorical NaN with "Unknown"
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    print("✅ Missing values handled.")
    return df

# ---------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra useful features like engagement and upload_month."""
    df = df.copy()

    # Engagement rate = (likes + comments) / views
    if {"likes", "comments", "views"}.issubset(df.columns):
        df["engagement"] = (df["likes"] + df["comments"]) / df["views"].replace(0, np.nan)
        df["engagement"] = df["engagement"].fillna(0)

    # Extract upload_month from date if available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["upload_month"] = df["date"].dt.month.fillna(-1).astype(int)

    print("✅ Feature engineering applied.")
    return df

# ---------------------------
# STEP 3: PREPROCESSOR PIPELINE
# ---------------------------

def build_preprocessor(df: pd.DataFrame, features: list):
    """
    Create a ColumnTransformer that preprocesses numeric + categorical features.
    Returns: (preprocessor, numeric_features, categorical_features)
    """
    # Detect column types
    numeric_features = [c for c in features if df[c].dtype in [np.int64, np.float64]]
    categorical_features = [c for c in features if c not in numeric_features]

    # Numeric pipeline
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine into column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features)
        ]
    )

    print("✅ Preprocessor pipeline built.")
    return preprocessor, numeric_features, categorical_features
def save_preprocessed_data(df, filename="preprocessed_data.csv"):
    """Save preprocessed DataFrame to the data folder."""
    output_path = os.path.join(PROJECT_ROOT, "data", filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved at: {output_path}")
    return output_path