"""
preprocessing.py

Author : Shibaji Biswas
Project: Car Price Prediction using Machine Learning

This module is responsible for:
1. Loading the dataset
2. Inspecting the dataset
3. Cleaning the dataset
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV dataset.

    Parameters:
        file_path (str): Path of the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.\n")
        return df

    except FileNotFoundError:
        print("❌ Dataset not found.")
        raise

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def inspect_data(df: pd.DataFrame) -> None:
    """
    Display important information about the dataset.
    """

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records.

    Parameters:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """

    initial_rows = len(df)

    df = df.drop_duplicates()

    final_rows = len(df)

    print("\nCleaning Report")
    print("-" * 40)
    print(f"Rows before cleaning : {initial_rows}")
    print(f"Rows after cleaning  : {final_rows}")
    print(f"Duplicates removed   : {initial_rows - final_rows}")

    return df