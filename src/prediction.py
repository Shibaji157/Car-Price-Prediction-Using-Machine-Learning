"""
Prediction Module

Author : Shibaji Biswas
"""

import joblib
import pandas as pd


def predict_price(sample_data):
    """
    Predict car price using the saved model.

    Parameters
    ----------
    sample_data : pandas.DataFrame

    Returns
    -------
    float
    """

    model = joblib.load("models/best_model.pkl")

    prediction = model.predict(sample_data)

    return prediction[0]