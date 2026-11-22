import pandas as pd
from pipeline import NetworkIDSPipeline

pipeline = NetworkIDSPipeline()

df = pd.read_excel("data/test.xlsx")

correct = 0

for i in range(20):

    sample = df.iloc[[i]]

    actual = sample["label"].values[0]

    predicted = pipeline.predict_binary(sample)[0]

    if actual == predicted:
        correct += 1

    print(
        f"Row {i} | Actual={actual} | Predicted={predicted}"
    )

print("\nCorrect:", correct)
print("Accuracy on first 20 rows:", correct / 20)