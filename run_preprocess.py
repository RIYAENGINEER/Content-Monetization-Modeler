from src.data_load import load_data
from src.preprocess import remove_duplicates, handle_missing, basic_feature_engineering, build_preprocessor,save_preprocessed_data

def main():
    # 1. Load dataset
    df = load_data()   # uses config.DATA_PATH (data/youtube_ad_revenue_dataset.csv)

    # 2. Clean
    df = remove_duplicates(df)
    df = handle_missing(df)

    # 3. Feature engineering
    df = basic_feature_engineering(df)

    # 4. Define features (all except target)
    target = "ad_revenue_usd"
    features = [c for c in df.columns if c != target]

    # 5. Build preprocessing pipeline
    preprocessor, num_cols, cat_cols = build_preprocessor(df, features)

    print("\n✅ Preprocessing finished.")
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    save_preprocessed_data(df)


if __name__ == "__main__":
    main()

