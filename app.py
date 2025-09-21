# app.py
import os
import joblib
import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT, MODEL_DIR

st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="wide")

PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")


@st.cache_data(ttl=3600)
def load_preprocessed_df(path=PREPROCESSED_PATH):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def build_input_widgets(df, target_col="ad_revenue_usd"):
    """
    Create a dict of Streamlit input widgets for all features based on df.
    Returns (user_df, input_order) where user_df is a single-row DataFrame ready for model.predict
    and input_order is list of feature names used.
    """
    st.sidebar.header("Input features")
    features = [c for c in df.columns if c != target_col]

    input_data = {}
    for feature in features:
        col_series = df[feature].dropna()
        # detect numeric vs categorical
        if pd.api.types.is_numeric_dtype(col_series):
            # choose a sensible default (median) and range
            default = float(col_series.median() if len(col_series) > 0 else 0.0)
            min_val = float(col_series.min()) if len(col_series) > 0 else 0.0
            max_val = float(col_series.max()) if len(col_series) > 0 else default * 10 if default != 0 else 100.0
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            # enforce sensible min/max for number_input
            val = st.sidebar.number_input(
                label=f"{feature} (numeric)",
                value=default,
                min_value=min_val if min_val != float("-inf") else None,
                max_value=max_val if max_val != float("inf") else None,
                step=step,
                format="%.3f"
            )
            input_data[feature] = val
        else:
            # categorical: use most common options (top 50 unique)
            uniques = col_series.astype(str).value_counts().index.tolist()
            options = uniques[:50] if len(uniques) > 0 else ["Unknown"]
            default = options[0]
            val = st.sidebar.selectbox(label=f"{feature} (categorical)", options=options, index=0)
            input_data[feature] = val

    # create a single-row DataFrame in same column order
    user_df = pd.DataFrame([input_data], columns=features)
    return user_df, features


def main():
    st.title("YouTube Ad Revenue Predictor")
    st.markdown(
        "Enter video/channel metrics on the left. The app will use the trained pipeline "
        "(`models/best_model.joblib`) to predict `ad_revenue_usd`."
    )

    df = load_preprocessed_df()
    model = load_model()

    if df is None:
        st.error(f"Preprocessed data not found at `{PREPROCESSED_PATH}`. Run preprocessing first (e.g. `python run_preprocess.py`).")
        return

    if model is None:
        st.error(f"Saved model not found at `{MODEL_PATH}`. Run training first (e.g. `python train_model.py`).")
        return

    # Build inputs dynamically based on preprocessed_data columns
    user_df, features = build_input_widgets(df)

    st.subheader("Preview of input (single row)")
    st.dataframe(user_df.T, use_container_width=True)

    if st.button("Predict ad revenue (USD)"):
        try:
            preds = model.predict(user_df)
            pred = float(preds[0])
            st.metric(label="Estimated ad revenue (USD)", value=f"${pred:,.2f}")
            # show a small explanation: model name and features used
            model_type = type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__
            st.caption(f"Model used: {model_type}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Model & data info")
    st.sidebar.markdown(f"- Preprocessed data: `{PREPROCESSED_PATH}`")
    st.sidebar.markdown(f"- Model file: `{MODEL_PATH}`")
    st.sidebar.markdown("- Tip: if a categorical option you want isn't available, add it to the preprocessed CSV or retrain with that category present.")


if __name__ == "__main__":
    main()
