# Network Intrusion Detection System Using Machine Learning

## Overview

The Network Intrusion Detection System (NIDS) is a machine learning-based cybersecurity project designed to detect malicious network traffic and classify different types of cyberattacks.

The system utilizes the UNSW-NB15 dataset and implements Random Forest classification for binary and multi-class intrusion detection.

## Key Features

- Binary Classification of normal and malicious network traffic.
- Multi-Class Attack Classification.
- Random Forest Model Development.
- Hyperparameter Tuning.
- Feature Importance Analysis.
- Interactive Streamlit Dashboard.
- CSV Prediction Report Generation.

## Dataset

The project uses the UNSW-NB15 dataset.

It contains network traffic records categorized into normal traffic and multiple attack types.

Dataset source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Model Performance

| Classification | Accuracy |
|---|---:|
| Binary Classification | 89.54% |
| Multi-Class Classification | 69.05% |

## Project Structure

- `app/` - Streamlit application.
- `src/` - Data preprocessing, training, prediction and evaluation scripts.
- `data/` - Dataset directory.
- `models/` - Trained model storage.
- `reports/` - Model evaluation reports and visualizations.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project:

```bash
cd NetworkIDS
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the UNSW-NB15 dataset files in the data directory.

## Running the Application

Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

## Results

The system generates intrusion detection predictions and provides downloadable CSV reports through the Streamlit dashboard.

## Future Enhancements

- Real-time network traffic monitoring.
- Advanced machine learning algorithms.
- Improved minority attack detection.
- Automated security alerts.
- Cloud deployment.

## Author

Developed as an undergraduate Machine Learning project.