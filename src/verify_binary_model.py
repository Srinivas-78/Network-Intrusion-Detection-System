from preprocess import DataPreprocessor
from sklearn.metrics import accuracy_score
import joblib

processor = DataPreprocessor()

X_train, X_test, y_train, y_test = (
    processor.prepare_binary_data()
)

model = joblib.load(
    "models/best_random_forest.pkl"
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Verified Accuracy:", accuracy)