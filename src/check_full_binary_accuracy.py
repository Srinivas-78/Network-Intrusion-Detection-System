from pipeline import NetworkIDSPipeline
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

pipeline = NetworkIDSPipeline()

df = pd.read_excel("data/test.xlsx")

actual = df["label"]

predicted = pipeline.predict_binary(df)

print("\nAccuracy:")
print(accuracy_score(actual, predicted))

print("\nClassification Report:\n")
print(
    classification_report(
        actual,
        predicted
    )
)