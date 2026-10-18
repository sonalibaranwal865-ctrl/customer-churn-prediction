"""Run basic EDA on the Telco Customer Churn dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Read README.md for download instructions."
        )

    df = pd.read_csv(DATA_PATH)
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- First five rows ---")
    print(df.head())
    print("\n--- Data types ---")
    print(df.dtypes)
    print("\n--- Missing values ---")
    print(df.isna().sum().sort_values(ascending=False))
    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())
    print("\n--- Target distribution ---")
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' was not found.")
    print(df["Churn"].value_counts(dropna=False))
    print("\n--- Numeric summary ---")
    print(df.describe(include="all").T)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig(REPORTS / "churn_distribution.png", dpi=150)
    plt.close()

    if "tenure" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x="tenure", hue="Churn", bins=25, kde=True)
        plt.title("Tenure Distribution by Churn")
        plt.tight_layout()
        plt.savefig(REPORTS / "tenure_by_churn.png", dpi=150)
        plt.close()

    print(f"\nEDA plots saved to: {REPORTS}")

if __name__ == "__main__":
    main()
