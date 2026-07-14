"""
eda.py

Exploratory Data Analysis for Car Price Prediction
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Create images/eda folder if it doesn't exist
os.makedirs("images/eda", exist_ok=True)


def perform_eda(original_df: pd.DataFrame):
    """
    Generate and save EDA visualizations.
    """

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    sns.set_style("whitegrid")

    # -----------------------------
    # 1. Price Distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(original_df["price"], kde=True)
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig("images/eda/price_distribution.png")
    plt.close()

    # -----------------------------
    # 2. Correlation Heatmap
    # -----------------------------
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        original_df.select_dtypes(include=["number"]).corr(),
        cmap="coolwarm",
        annot=False
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("images/eda/correlation_heatmap.png")
    plt.close()

    # -----------------------------
    # 3. Brand-wise Average Price
    # -----------------------------
    original_df["Brand"] = original_df["CarName"].apply(lambda x: x.split()[0])

    brand_price = (
        original_df.groupby("Brand")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=brand_price.index,
        y=brand_price.values
    )

    plt.xticks(rotation=90)
    plt.title("Average Price by Brand")
    plt.tight_layout()
    plt.savefig("images/eda/brand_price.png")
    plt.close()

    # -----------------------------
    # 4. Horsepower vs Price
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="horsepower",
        y="price",
        data=original_df
    )
    plt.title("Horsepower vs Price")
    plt.tight_layout()
    plt.savefig("images/eda/horsepower_vs_price.png")
    plt.close()

    # -----------------------------
    # 5. Engine Size vs Price
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="enginesize",
        y="price",
        data=original_df
    )
    plt.title("Engine Size vs Price")
    plt.tight_layout()
    plt.savefig("images/eda/enginesize_vs_price.png")
    plt.close()

    # -----------------------------
    # 6. Fuel Type Distribution
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sns.countplot(
        x="fueltype",
        data=original_df
    )
    plt.title("Fuel Type Distribution")
    plt.tight_layout()
    plt.savefig("images/eda/fueltype_distribution.png")
    plt.close()

    # -----------------------------
    # 7. Car Body Distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x="carbody",
        data=original_df
    )

    plt.xticks(rotation=20)
    plt.title("Car Body Distribution")
    plt.tight_layout()
    plt.savefig("images/eda/carbody_distribution.png")
    plt.close()

    # -----------------------------
    # 8. Highway MPG vs Price
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="highwaympg",
        y="price",
        data=original_df
    )
    plt.title("Highway MPG vs Price")
    plt.tight_layout()
    plt.savefig("images/eda/highwaympg_vs_price.png")
    plt.close()

    print("✅ All EDA graphs saved in images/eda/")