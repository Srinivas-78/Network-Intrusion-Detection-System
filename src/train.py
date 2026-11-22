from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocess import DataPreprocessor


def main():

    print("=" * 60)
    print("NETWORK IDS - RANDOM FOREST TRAINING")
    print("=" * 60)

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        processor.prepare_binary_classification_data()
    )

    print("\nTraining Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training Completed.")

    print("\nMaking Predictions...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "models/random_forest_model.pkl"
    )

    print("\nModel Saved:")
    print("models/random_forest_model.pkl")


if __name__ == "__main__":
    main()