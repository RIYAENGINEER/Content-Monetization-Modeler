"""
src/data_load.py

Simple, defensive CSV loader for the project.
Uses src/config.py -> DATA_PATH by default, but allows overriding.
"""

import os
import pandas as pd

# Try to import project config; fall back to a sensible default path
try:
    from src.config import DATA_PATH
except Exception:
    # If imports fail (running module as script directly), derive a default path
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "youtube_ad_revenue_dataset.csv")

def load_data(path: str = None, nrows: int | None = None) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame.

    Args:
        path: optional path to CSV. If None, uses DATA_PATH from config.
        nrows: optional number of rows to read (useful for quick tests).

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: if the file cannot be found.
        pd.errors.EmptyDataError / pd.errors.ParserError: if CSV is malformed.
    """
    csv_path = path if path else DATA_PATH

    if not os.path.isabs(csv_path):
        # make it relative to project root (when src is a package)
        possible = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)
        if os.path.exists(possible):
            csv_path = possible

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}\n"
                                "Place your file at data/youtube_ad_revenue_dataset.csv or pass the path to load_data().")

    # Read CSV with a few safe options
    try:
        df = pd.read_csv(csv_path, nrows=nrows)
    except pd.errors.EmptyDataError:
        raise
    except pd.errors.ParserError:
        # Try a more tolerant read
        df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    return df

# Optional quick test when running file directly:
if __name__ == "__main__":
    print("Running quick loader test...")
    try:
        df = load_data(nrows=5)
        print("Loaded shape:", df.shape)
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print("Error loading data:", e)
