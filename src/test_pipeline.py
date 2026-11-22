import pandas as pd
from pipeline import NetworkIDSPipeline


pipeline = NetworkIDSPipeline()

df = pd.read_excel(
    "data/test.xlsx"
)

sample = df.iloc[[0]]

print("\nActual Label:")
print(sample["label"].values[0])

binary_prediction = pipeline.predict_binary(
    sample
)

print("\nBinary Prediction:")
print(binary_prediction[0])

attack_prediction = (
    pipeline.predict_attack_type(
        sample
    )
)
print("\nAttack Prediction:")
print(attack_prediction[0])