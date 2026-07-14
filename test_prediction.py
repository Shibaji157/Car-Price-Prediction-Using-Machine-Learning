import joblib
import pandas as pd

# Load processed dataset
df = pd.read_csv("outputs/processed_dataset.csv")

# Remove target column
X = df.drop("price", axis=1)

# Load model
model = joblib.load("models/best_model.pkl")

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("=" * 50)
print("Predicted Price")
print("=" * 50)
print(prediction[0])