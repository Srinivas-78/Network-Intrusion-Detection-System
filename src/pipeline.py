import joblib
import pandas as pd


class NetworkIDSPipeline:

    def __init__(self):

        print("Loading Models...")

        self.binary_model = joblib.load(
            "models/best_random_forest.pkl"
        )

        self.multiclass_model = joblib.load(
            "models/multiclass_rf.pkl"
        )

        self.attack_encoder = joblib.load(
            "models/attack_label_encoder.pkl"
        )

        self.label_encoders = joblib.load(
            "models/label_encoders.pkl"
        )

        print("Models Loaded Successfully")

    def preprocess(self, df):

        data = df.copy()

        for col in ["proto", "service", "state"]:

            data[col] = (
                self.label_encoders[col]
                .transform(data[col].astype(str))
            )

        X = data.drop(
            columns=[
                "id",
                "label",
                "attack_cat"
            ],
            errors="ignore"
        )

        return X

    def predict_binary(self, df):

        X = self.preprocess(df)

        prediction = self.binary_model.predict(X)

        return prediction

    def predict_attack_type(self, df):

        X = self.preprocess(df)

        prediction = self.multiclass_model.predict(X)

        attack_names = (
            self.attack_encoder
            .inverse_transform(prediction)
        )
        return attack_names