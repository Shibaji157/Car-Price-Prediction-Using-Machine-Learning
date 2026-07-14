"""
model_training.py

Train and compare multiple regression models.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score


def train_models(df: pd.DataFrame):

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
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
    best_score = -999
    best_name = ""

    print("\nTraining Models...\n")

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

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model : {best_name}")
    print(f"R² Score : {best_score:.4f}")

    return best_model, X_test, y_test, results