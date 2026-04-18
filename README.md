# Startup Outcome Predictor

A machine learning application that predicts whether a startup is likely to succeed — 
defined as being acquired or achieving an IPO — based on its funding history and 
company profile data.

---

## Project Overview

This project implements an end-to-end machine learning pipeline covering data 
preparation, feature engineering, model training, evaluation, hyperparameter tuning, 
and deployment via a Streamlit web application.

Three classification models are compared: Logistic Regression (baseline), 
Decision Tree, and Random Forest. The tuned Random Forest is selected as the 
final production model.

---

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit application (trains model on launch, serves predictions) |
| `full_model_code.py` | Standalone end-to-end pipeline script |
| `notebook_steps.ipynb` | Step-by-step Google Colab notebook |
| `requirements.txt` | Python dependencies |
| `big_startup_secsees_dataset.csv` | Source dataset (place in same directory) |

---

## Getting Started

### 1. Clone or download this repository

```bash
git clone <repository-url>
cd startup-outcome-predictor
```

### 2. Install dependencies

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

### 3. Ensure the dataset is present

Place `big_startup_secsees_dataset.csv` in the same directory as `app.py`.

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## How the Application Works

- On first launch, the application loads the dataset and trains the model. 
  This typically takes 30–60 seconds depending on hardware.
- Subsequent loads use Streamlit's caching mechanism, so the model is not retrained.
- Enter company details in the left sidebar and click **Generate Prediction**.
- The main panel displays the predicted outcome, probability score, and a confidence breakdown.

---

## Target Variable

| Label | Status Values | Meaning |
|-------|--------------|---------|
| 1 — Successful | `acquired`, `ipo` | Confirmed positive outcome |
| 0 — Unsuccessful | `closed` | Company ceased operations |

Records with `status = operating` are excluded as their outcome is not yet confirmed.

---

## Features Used

| Feature | Description |
|---------|-------------|
| Business sector | Primary category extracted from `category_list` |
| Total funding (USD) | Cleaned numeric value from `funding_total_usd` |
| Number of funding rounds | `funding_rounds` |
| Country, State, Region, City | Geographic identifiers |
| Year founded | Extracted from `founded_at` |
| First funding year | Extracted from `first_funding_at` |
| Latest funding year | Extracted from `last_funding_at` |
| Time to first funding (days) | Difference between founding date and first funding date |
| Time between first and last funding (days) | Duration of the funding period |

---

## Running the Standalone Script

To run the full pipeline outside of Streamlit:

```bash
python full_model_code.py
```

This will train all three models, print evaluation metrics, perform hyperparameter 
tuning, and output feature importances to the console.

---

## Limitations

- The dataset reflects historical funding patterns and is weighted towards 
  US-based technology companies. Predictions for companies in other geographies 
  or sectors may be less reliable.
- The model was trained on data up to approximately 2015 and may not reflect 
  current market conditions.
- Predictions should be treated as probabilistic indicators rather than 
  definitive outcomes.
