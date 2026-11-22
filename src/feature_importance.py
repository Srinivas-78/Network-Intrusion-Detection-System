from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from preprocess import DataPreprocessor


def main():

    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    model_path = Path("models/random_forest_model.pkl")

    if not model_path.exists():
        print("Model not found.")
        print("Run train.py first.")
        return

    model = joblib.load(model_path)

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        processor.prepare_binary_classification_data()
    )

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features:\n")
    print(importance_df.head(20))

    Path("reports").mkdir(exist_ok=True)

    importance_df.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(12, 8))

    top_features = importance_df.head(15)

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png",
        dpi=300
    )

    print("\nSaved:")
    print("reports/feature_importance.csv")
    print("reports/feature_importance.png")


if __name__ == "__main__":
    main()