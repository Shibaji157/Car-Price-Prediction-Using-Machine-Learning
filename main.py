"""
main.py

Author : Shibaji Biswas
Project : Car Price Prediction using Machine Learning
"""

from src.model_saver import save_model
from src.evaluation import evaluate_model
from src.preprocessing import (
    load_data,
    inspect_data,
    clean_data,
)
from src.feature_engineering import engineer_features
from src.eda import perform_eda
from src.model_training import train_models


def main():
    """
    Main function of the project.
    """

    print("=" * 70)
    print("CAR PRICE PREDICTION USING MACHINE LEARNING")
    print("=" * 70)

    # Load Dataset
    df = load_data("data/CarPrice_Assignment.csv")

    # Keep a copy for EDA
    original_df = df.copy()

    # Inspect Dataset
    inspect_data(df)

    # Clean Dataset
    df = clean_data(df)

    print("\n✅ Preprocessing Completed Successfully!")

    # Feature Engineering
    df = engineer_features(df)

    # Exploratory Data Analysis
    perform_eda(original_df)

    # Model Training
    best_model, X_test, y_test, results = train_models(df)

    # Model Evaluation
    evaluate_model(best_model, X_test, y_test)

    # Save Best Model
    save_model(best_model)

    print("\n" + "=" * 60)
    print("PROCESSED DATASET (First 5 Rows)")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    for model_name, score in results.items():
        print(f"{model_name:<25}: {score:.4f}")

    print("\n🏆 Best Model Selected Successfully!")


if __name__ == "__main__":
    main()