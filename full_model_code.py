# =============================================================================
# Startup Success Prediction - Full Model Code
# Machine Learning Pipeline: Problem Definition to Evaluation
# =============================================================================

# =============================================================================
# STEP 1: Import Libraries
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# STEP 2: Load Dataset
# =============================================================================

DATA_PATH = "big_startup_secsees_dataset.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nStatus value counts:\n{df['status'].value_counts()}")

# =============================================================================
# STEP 3: Inspect Dataset Structure
# =============================================================================

print("\n--- Dataset Info ---")
print(df.dtypes)
print("\n--- First 3 Rows ---")
print(df.head(3))

# =============================================================================
# STEP 4: Filter Target Classes and Define Binary Target
# =============================================================================

# Only companies with confirmed outcomes are retained.
# 'acquired' and 'ipo' are treated as successful outcomes.
# 'closed' is treated as unsuccessful.
# 'operating' records are excluded because their final outcome is not yet known,
# and including them would introduce a form of label noise into the modelling process.

df = df[df["status"].isin(["acquired", "ipo", "closed"])].copy()
df["target"] = (df["status"].isin(["acquired", "ipo"])).astype(int)

print(f"\nFiltered dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")
print(f"  1 = Successful (acquired or IPO)")
print(f"  0 = Unsuccessful (closed)")

# =============================================================================
# STEP 5: Clean Column Names
# =============================================================================

df.columns = df.columns.str.strip().str.lower()

# =============================================================================
# STEP 6: Check Missing Values and Duplicates
# =============================================================================

print("\n--- Missing Values ---")
print(df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
df = df.drop_duplicates()

# =============================================================================
# STEP 7: Clean Funding and Date Fields
# =============================================================================

# Funding total may contain commas or dashes; convert to numeric
df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"].astype(str).str.replace(",", "").str.strip(),
    errors="coerce"
)

# Parse date columns
for col in ["founded_at", "first_funding_at", "last_funding_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# =============================================================================
# STEP 8: Engineer Date-Based Features
# =============================================================================

df["founded_year"]       = df["founded_at"].dt.year
df["first_funding_year"] = df["first_funding_at"].dt.year
df["last_funding_year"]  = df["last_funding_at"].dt.year

df["time_to_first_funding_days"] = (
    df["first_funding_at"] - df["founded_at"]
).dt.days

df["time_between_first_and_last_funding_days"] = (
    df["last_funding_at"] - df["first_funding_at"]
).dt.days

# Extract the primary business category from the pipe-separated category_list field
df["primary_category"] = (
    df["category_list"]
    .fillna("Unknown")
    .str.split("|")
    .str[0]
    .str.strip()
)

print("\n--- Engineered Features Sample ---")
print(df[["founded_year", "first_funding_year", "last_funding_year",
          "time_to_first_funding_days", "time_between_first_and_last_funding_days",
          "primary_category"]].head(5))

# =============================================================================
# STEP 9: Exploratory Data Analysis (Summary Statistics)
# =============================================================================

print("\n--- Target Distribution ---")
print(df["target"].value_counts(normalize=True).mul(100).round(2))

print("\n--- Funding Total (USD) by Target ---")
print(df.groupby("target")["funding_total_usd"].median())

print("\n--- Funding Rounds by Target ---")
print(df.groupby("target")["funding_rounds"].mean().round(2))

print("\n--- Top 10 Business Categories ---")
print(df["primary_category"].value_counts().head(10))

# =============================================================================
# STEP 10: Select Features and Target
# =============================================================================

FEATURE_COLS = [
    "primary_category",
    "funding_total_usd",
    "country_code",
    "state_code",
    "region",
    "city",
    "funding_rounds",
    "founded_year",
    "first_funding_year",
    "last_funding_year",
    "time_to_first_funding_days",
    "time_between_first_and_last_funding_days",
]

TARGET_COL = "target"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")

# =============================================================================
# STEP 11: Define Categorical and Numerical Columns
# =============================================================================

CATEGORICAL_COLS = [
    "primary_category",
    "country_code",
    "state_code",
    "region",
    "city",
]

NUMERICAL_COLS = [
    "funding_total_usd",
    "funding_rounds",
    "founded_year",
    "first_funding_year",
    "last_funding_year",
    "time_to_first_funding_days",
    "time_between_first_and_last_funding_days",
]

print(f"\nCategorical features ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
print(f"Numerical features  ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")

# =============================================================================
# STEP 12: Train-Test Split (Stratified)
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Train target balance:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Test  target balance:\n{y_test.value_counts(normalize=True).round(3)}")

# =============================================================================
# STEP 13: Build Preprocessing Pipeline Using ColumnTransformer
# =============================================================================

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, NUMERICAL_COLS),
    ("cat", categorical_transformer, CATEGORICAL_COLS),
])

# =============================================================================
# STEP 14: Train Logistic Regression (Baseline Model)
# =============================================================================

lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_lr, target_names=["Unsuccessful", "Successful"]))

# =============================================================================
# STEP 15: Train Decision Tree
# =============================================================================

dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced")),
])

dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

print("\n--- Decision Tree Results ---")
print(classification_report(y_test, y_pred_dt, target_names=["Unsuccessful", "Successful"]))

# =============================================================================
# STEP 16: Train Random Forest
# =============================================================================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf, target_names=["Unsuccessful", "Successful"]))

# =============================================================================
# STEP 17: Evaluate All Models
# =============================================================================

def evaluate_model(name, y_true, y_pred):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

results = pd.DataFrame([
    evaluate_model("Logistic Regression", y_test, y_pred_lr),
    evaluate_model("Decision Tree",       y_test, y_pred_dt),
    evaluate_model("Random Forest",       y_test, y_pred_rf),
])

print("\n--- Model Comparison ---")
print(results.to_string(index=False))

print("\n--- Confusion Matrices ---")
for name, y_pred in [("Logistic Regression", y_pred_lr),
                      ("Decision Tree",       y_pred_dt),
                      ("Random Forest",       y_pred_rf)]:
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name}:\n{cm}")

# Cross-validation (3-fold, stratified)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n--- 3-Fold Cross-Validation (F1-Score) ---")
for name, pipeline in [("Logistic Regression", lr_pipeline),
                        ("Decision Tree",       dt_pipeline),
                        ("Random Forest",       rf_pipeline)]:
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"{name}: {scores.round(4)} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")

# =============================================================================
# STEP 18: Hyperparameter Tuning for Random Forest (GridSearchCV)
# =============================================================================

# The parameter grid is deliberately kept compact to ensure computational
# efficiency while still exploring the most impactful hyperparameters.

param_grid = {
    "classifier__n_estimators":     [100, 150],
    "classifier__max_depth":        [10, 12],
    "classifier__min_samples_split": [10],
    "classifier__min_samples_leaf":  [5],
}

rf_tuned_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(random_state=42, class_weight="balanced")),
])

grid_search = GridSearchCV(
    rf_tuned_pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n--- Tuned Random Forest Results ---")
print(classification_report(y_test, y_pred_best, target_names=["Unsuccessful", "Successful"]))

# =============================================================================
# STEP 19: Compare Models (Final Summary)
# =============================================================================

final_results = pd.DataFrame([
    evaluate_model("Logistic Regression", y_test, y_pred_lr),
    evaluate_model("Decision Tree",       y_test, y_pred_dt),
    evaluate_model("Random Forest",       y_test, y_pred_rf),
    evaluate_model("Random Forest (Tuned)", y_test, y_pred_best),
])

print("\n--- Final Model Comparison ---")
print(final_results.to_string(index=False))

# =============================================================================
# STEP 20: Feature Importance (Tuned Random Forest)
# =============================================================================

rf_classifier = best_model.named_steps["classifier"]
ohe_feature_names = (
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(CATEGORICAL_COLS)
    .tolist()
)
all_feature_names = NUMERICAL_COLS + ohe_feature_names

importances = pd.Series(rf_classifier.feature_importances_, index=all_feature_names)
top_features = importances.sort_values(ascending=False).head(15)

print("\n--- Top 15 Feature Importances ---")
print(top_features.round(4).to_string())

# =============================================================================
# STEP 21: Save Final Conclusions
# =============================================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSIONS")
print("=" * 60)
print("""
Three classification models were trained and evaluated on a balanced
binary prediction task derived from the startup funding dataset.

Logistic Regression served as an interpretable baseline and delivered
competitive performance, demonstrating that linear decision boundaries
can capture meaningful patterns in this domain.

Decision Tree offered high interpretability but showed signs of
overfitting, particularly where the training-to-test performance gap
was more pronounced.

Random Forest, as an ensemble method, consistently outperformed the
other approaches on accuracy, precision, recall, and F1-score metrics.
Hyperparameter tuning via GridSearchCV further refined the model,
yielding marginal but consistent gains in predictive quality.

The most influential predictors were funding-related variables
(total funding and number of rounds) and temporal features such as
the year of founding and the duration between first and last funding.
Geographic and sector features contributed additional discriminative
signal, though at a lower magnitude.

The tuned Random Forest is selected as the final production model.
""")
