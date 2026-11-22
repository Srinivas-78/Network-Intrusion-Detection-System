from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


class DataPreprocessor:

    def __init__(self):
        self.train_path = Path("data/train.xlsx")
        self.test_path = Path("data/test.xlsx")

        self.categorical_columns = [
            "proto",
            "service",
            "state"
        ]

        self.label_encoders = {}

    def load_data(self):
        print("Loading datasets...")

        train_df = pd.read_excel(self.train_path)
        test_df = pd.read_excel(self.test_path)

        print(f"Train Shape: {train_df.shape}")
        print(f"Test Shape : {test_df.shape}")

        return train_df, test_df

    def encode_categorical_features(self, train_df, test_df):

        for column in self.categorical_columns:

            encoder = LabelEncoder()

            combined_values = pd.concat(
                [
                    train_df[column],
                    test_df[column]
                ],
                axis=0
            ).astype(str)

            encoder.fit(combined_values)

            train_df[column] = encoder.transform(
                train_df[column].astype(str)
            )

            test_df[column] = encoder.transform(
                test_df[column].astype(str)
            )

            self.label_encoders[column] = encoder

        return train_df, test_df

    def save_encoders(self):

        Path("models").mkdir(exist_ok=True)

        joblib.dump(
            self.label_encoders,
            "models/label_encoders.pkl"
        )

        print("Encoders saved successfully.")

    def prepare_binary_classification_data(self):

        train_df, test_df = self.load_data()

        train_df, test_df = self.encode_categorical_features(
            train_df,
            test_df
        )

        self.save_encoders()

        drop_columns = [
            "id",
            "attack_cat"
        ]

        X_train = train_df.drop(
            columns=drop_columns + ["label"]
        )

        y_train = train_df["label"]

        X_test = test_df.drop(
            columns=drop_columns + ["label"]
        )

        y_test = test_df["label"]

        print("\nBinary Classification Dataset Ready")

        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)

        print("X_test :", X_test.shape)
        print("y_test :", y_test.shape)

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        processor.prepare_binary_classification_data()
    )