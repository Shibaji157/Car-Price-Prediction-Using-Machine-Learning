"""
Feature Importance Visualization
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X_train):
    """
    Plot feature importance for tree-based models.
    """

    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available.")
        return

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    )

    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))

    importance.head(15).plot(kind="barh")

    plt.title("Top 15 Important Features")

    plt.xlabel("Importance Score")

    plt.gca().invert_yaxis()

    os.makedirs("images", exist_ok=True)

    plt.tight_layout()

    plt.savefig("images/feature_importance.png")

    plt.close()

    print("✅ Feature Importance graph saved.")