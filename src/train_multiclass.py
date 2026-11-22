from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocess import DataPreprocessor


def main():

    print("=" * 60)
    print("MULTI-CLASS ATTACK CLASSIFICATION")
    print("=" * 60)

    processor = DataPreprocessor()

    train_df, test_df = processor.load_data()

    train_df, test_df = processor.encode_categorical_features(
        train_df,
        test_df
    )

    attack_encoder = LabelEncoder()

    combined_labels = pd.concat(
        [
            train_df["attack_cat"],
            test_df["attack_cat"]
        ]
    )

    attack_encoder.fit(combined_labels)

    train_df["attack_cat"] = attack_encoder.transform(
        train_df["attack_cat"]
    )

    test_df["attack_cat"] = attack_encoder.transform(
        test_df["attack_cat"]
    )

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        attack_encoder,
        "models/attack_label_encoder.pkl"
    )

    drop_columns = [
        "id",
        "label"
    ]

    X_train = train_df.drop(
        columns=drop_columns + ["attack_cat"]
    )

    y_train = train_df["attack_cat"]

    X_test = test_df.drop(
        columns=drop_columns + ["attack_cat"]
    )

    y_test = test_df["attack_cat"]

    print("\nTraining Multi-Class Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training Completed.")

    print("\nPredicting...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print("MULTI-CLASS RESULTS")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClass Mapping:")

    for index, name in enumerate(
        attack_encoder.classes_
    ):
        print(f"{index} -> {name}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(
        model,
        "models/multiclass_rf.pkl"
    )

    print("\nSaved:")
    print("models/multiclass_rf.pkl")
    print("models/attack_label_encoder.pkl")


if __name__ == "__main__":
    main()