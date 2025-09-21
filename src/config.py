"""
src/config.py

Centralized configuration for the Content Monetization project.
Keeps paths and constants in one place so they can be reused.
"""

import os

# Project root (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Default dataset path
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "youtube_ad_revenue_dataset.csv")

# Models directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Random seed for reproducibility
RANDOM_STATE = 42

# Test split size
TEST_SIZE = 0.2

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_PATH:", DATA_PATH)
    print("MODEL_DIR:", MODEL_DIR)
    print("RANDOM_STATE:", RANDOM_STATE)
