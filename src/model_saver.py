"""
model_saver.py

Save trained machine learning model.
"""

import os
import joblib


def save_model(model, filename="models/best_model.pkl"):
    """
    Save trained model.
    """

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, filename)

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)
    print(f"Model saved successfully at: {filename}")