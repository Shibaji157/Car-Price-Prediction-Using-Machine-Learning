"""
prediction.py

Predict car price using the saved model.
"""

import joblib
import pandas as pd


def predict_price(sample_data):
    """
    Predict car price.

    Parameters:
        sample_data (dict): Dictionary containing feature values.
    """

    model = joblib.load("models/best_model.pkl")

    sample_df = pd.DataFrame([sample_data])

    prediction = model.predict(sample_df)

    print("\n" + "=" * 60)
    print("CAR PRICE PREDICTION")
    print("=" * 60)
    print(f"Predicted Price : ₹{prediction[0]:,.2f}")

    return prediction[0]