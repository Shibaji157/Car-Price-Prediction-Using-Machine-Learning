"""
feature_engineering.py

Feature engineering for Car Price Prediction.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.DataFrame: Feature-engineered dataset.
    """

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Extract Brand Name
    df["Brand"] = df["CarName"].apply(lambda x: x.split()[0].lower())

    # Correct incorrect brand names
    brand_mapping = {
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
    }

    df["Brand"] = df["Brand"].replace(brand_mapping)

    print("\nUnique Brands:")
    print(sorted(df["Brand"].unique()))

    # Remove unnecessary columns
    df.drop(columns=["car_ID", "CarName"], inplace=True)

    # One-Hot Encoding
    categorical_columns = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

    print("\nFeature Engineering Completed.")
    print(f"Dataset Shape : {df.shape}")
    print("\nFirst 10 Columns:")
    print(df.columns[:10].tolist())

    print("\nTarget Variable:")
    print("price")
    return df