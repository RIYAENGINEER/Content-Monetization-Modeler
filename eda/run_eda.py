import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Paths
DATA_PATH = os.path.join("data", "youtube_ad_revenue_dataset.csv")
OUTPUT_DIR = os.path.join("eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

# 3. Basic info
print("📊 Shape:", df.shape)
print("\n🔎 Columns:", df.columns.tolist())
print("\nℹ️ Info:")
print(df.info())
print("\n🧾 Summary statistics:")
print(df.describe(include="all").T)

# 4. Missing values
print("\n❓ Missing values per column:")
print(df.isnull().sum())

# 5. Duplicates
duplicates = df.duplicated().sum()
print(f"\n🌀 Duplicates: {duplicates}")

# 6. Correlation heatmap (numeric only)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# 7. Distribution of numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{col}.png"))
    plt.close()

# 8. Target vs feature scatter plots
target = "ad_revenue_usd"
if target in df.columns:
    for col in numeric_cols:
        if col != target:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target], alpha=0.5)
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{col}_vs_{target}.png"))
            plt.close()

print(f"\n✅ EDA completed. Plots saved in: {OUTPUT_DIR}/")
