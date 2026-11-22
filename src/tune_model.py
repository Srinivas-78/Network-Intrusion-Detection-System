from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from preprocess import DataPreprocessor


def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred


def main():

    print("=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        processor.prepare_binary_classification_data()
    )

    configurations = [

        {
            "name": "Model_1",
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": None
        },

        {
            "name": "Model_2",
            "n_estimators": 300,
            "max_depth": 25,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": None
        },

        {
            "name": "Model_3",
            "n_estimators": 300,
            "max_depth": 30,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": None
        },

        {
            "name": "Model_4",
            "n_estimators": 300,
            "max_depth": 30,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced"
        }

    ]

    best_accuracy = 0
    best_model = None
    best_name = None

    results = []

    for config in configurations:

        print("\n" + "=" * 50)
        print(f"Training {config['name']}")
        print("=" * 50)

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight=config["class_weight"],
            random_state=42,
            n_jobs=-1
        )

        accuracy, y_pred = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        print(f"Accuracy: {accuracy:.4f}")

        results.append({
            "Model": config["name"],
            "Accuracy": accuracy
        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_name = config["name"]

    print("\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    print(results_df)

    print("\nBest Model:")
    print(best_name)

    print(f"Best Accuracy: {best_accuracy:.4f}")

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_random_forest.pkl"
    )

    results_df.to_csv(
        "reports/tuning_results.csv",
        index=False
    )

    print("\nSaved:")
    print("models/best_random_forest.pkl")
    print("reports/tuning_results.csv")

    print("\nClassification Report Of Best Model:\n")

    final_predictions = best_model.predict(X_test)

    print(
        classification_report(
            y_test,
            final_predictions
        )
    )


if __name__ == "__main__":
    main()