import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import NetworkIDSPipeline

st.set_page_config(
    page_title="Network IDS",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")

st.markdown("---")

pipeline = NetworkIDSPipeline()

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.markdown("---")

    st.subheader("Running Detection...")

    binary_predictions = pipeline.predict_binary(df)

    attack_predictions = pipeline.predict_attack_type(df)

    results = df.copy()

    results["Predicted_Label"] = binary_predictions

    results["Predicted_Attack"] = attack_predictions

    normal_count = (
        results["Predicted_Label"] == 0
    ).sum()

    attack_count = (
        results["Predicted_Label"] == 1
    ).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Traffic",
            int(normal_count)
        )

    with col2:
        st.metric(
            "Attack Traffic",
            int(attack_count)
        )

    st.markdown("---")

    st.subheader("Attack Distribution")

    attack_summary = (
        results["Predicted_Attack"]
        .value_counts()
        .reset_index()
    )

    attack_summary.columns = [
        "Attack Type",
        "Count"
    ]

    st.bar_chart(
        attack_summary.set_index(
            "Attack Type"
        )
    )

    st.markdown("---")

    st.subheader("Prediction Results")

    st.dataframe(results.head(100))

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "ids_predictions.csv",
        "text/csv"
    )