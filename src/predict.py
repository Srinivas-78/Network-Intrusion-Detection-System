import joblib
import pandas as pd


def main():

    print("=" * 60)
    print("NETWORK IDS PREDICTION ENGINE")
    print("=" * 60)

    model = joblib.load(
        "models/multiclass_rf.pkl"
    )

    attack_encoder = joblib.load(
        "models/attack_label_encoder.pkl"
    )

    label_encoders = joblib.load(
        "models/label_encoders.pkl"
    )

    df = pd.read_excel(
        "data/test.xlsx"
    )

    for i in range(10):

        sample = df.iloc[[i]].copy()

        actual_attack = sample["attack_cat"].values[0]

        for column in ["proto", "service", "state"]:

            sample[column] = (
                label_encoders[column]
                .transform(sample[column].astype(str))
            )

        sample = sample.drop(
            columns=[
                "id",
                "label",
                "attack_cat"
            ]
        )

        prediction = model.predict(sample)

        predicted_attack = (
            attack_encoder
            .inverse_transform(prediction)[0]
        )

        print("\n" + "=" * 40)
        print(f"Record {i}")
        print("=" * 40)
        print(f"Actual    : {actual_attack}")
        print(f"Predicted : {predicted_attack}")


if __name__ == "__main__":
    main()