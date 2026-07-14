"""
model_training.py

Train and compare multiple regression models.

Author : Shibaji Biswas
"""

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


def train_models(df: pd.DataFrame):
    """
    Train multiple regression models and select the best one.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.

    Returns
    -------
    best_model
        Best trained model.
    X_train
        Training features.
    X_test
        Testing features.
    y_test
        Testing target.
    results
        Dictionary containing R² scores of all models.
    """

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # -----------------------------
    # Features and Target
    # -----------------------------
    X = df.drop("price", axis=1)
    y = df["price"]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}

    best_model = None
    best_score = float("-inf")
    best_name = ""

    print("\nTraining Models...\n")

    # -----------------------------
    # Train Every Model
    # -----------------------------
    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        score = r2_score(y_test, prediction)

        results[name] = score

        print(f"{name:<22} R² Score : {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # -----------------------------
    # Best Model
    # -----------------------------
    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model    : {best_name}")
    print(f"R² Score : {best_score:.4f}")

    # -----------------------------
    # Cross Validation
    # -----------------------------
    print("\nRunning 5-Fold Cross Validation...")

    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\nCross Validation Scores")

    for i, score in enumerate(cv_scores, start=1):
        print(f"Fold {i} : {score:.4f}")

    print(f"\nAverage CV Score   : {cv_scores.mean():.4f}")
    print(f"Standard Deviation : {cv_scores.std():.4f}")

    # -----------------------------
    # Return
    # -----------------------------
    return best_model, X_train, X_test, y_test, results