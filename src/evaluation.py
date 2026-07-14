"""
evaluation.py

Evaluate the trained machine learning model.
"""

import os
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # Save metrics
    os.makedirs("outputs", exist_ok=True)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
        "Value": [mae, mse, rmse, r2]
    })

    metrics.to_csv("outputs/model_metrics.csv", index=False)

    print("\n✅ Metrics saved to outputs/model_metrics.csv")

    # Prediction plot
    os.makedirs("images/model_results", exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2,
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")

    plt.tight_layout()

    plt.savefig("images/model_results/actual_vs_predicted.png")

    plt.close()

    print("✅ Prediction graph saved.")