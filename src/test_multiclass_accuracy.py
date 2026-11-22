import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("models/multiclass_rf.pkl")
attack_encoder = joblib.load("models/attack_label_encoder.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

df = pd.read_excel("data/test.xlsx")

for col in ["proto", "service", "state"]:
    df[col] = label_encoders[col].transform(
        df[col].astype(str)
    )

X = df.drop(
    columns=[
        "id",
        "label",
        "attack_cat"
    ]
)

y_true = df["attack_cat"]

y_pred_num = model.predict(X)

y_pred = attack_encoder.inverse_transform(
    y_pred_num
)

acc = accuracy_score(y_true, y_pred)

print("Accuracy:", acc)

print("\nFirst 20 Predictions:\n")

for i in range(20):
    print(
        f"{i}: Actual={y_true.iloc[i]} | Predicted={y_pred[i]}"
    )