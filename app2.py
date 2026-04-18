"""
Startup Outcome Predictor
A professional machine learning application built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Startup Outcome Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Roboto', 'Source Sans Pro', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f8f9fb;
    border-right: 1px solid #e8eaf0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Main area */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Result cards */
.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
}

.result-card-success {
    border-left: 5px solid #16a34a;
    background: #f0fdf4;
}

.result-card-fail {
    border-left: 5px solid #dc2626;
    background: #fff5f5;
}

.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.5rem;
}

.result-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.3rem;
}

.result-message {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.6;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    padding: 0.8rem 0 0.3rem 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 0.5rem;
}

.sidebar-section:first-of-type {
    border-top: none;
    padding-top: 0;
}

/* Metric strip */
.metric-strip {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    flex: 1;
    text-align: center;
}

.metric-box .metric-num {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111827;
}

.metric-box .metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.15rem;
}

/* Probability bar */
.prob-bar-container {
    background: #f3f4f6;
    border-radius: 6px;
    height: 8px;
    margin-top: 0.6rem;
    overflow: hidden;
}

.prob-bar-fill-success {
    background: #16a34a;
    height: 8px;
    border-radius: 6px;
    transition: width 0.4s ease;
}

.prob-bar-fill-fail {
    background: #dc2626;
    height: 8px;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* Expander styling */
div[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #fafafa;
}

/* Button */
div[data-testid="stButton"] button {
    background: #111827;
    color: #ffffff;
    border: none;
    border-radius: 7px;
    padding: 0.55rem 1.5rem;
    font-weight: 500;
    font-size: 0.9rem;
    width: 100%;
    transition: background 0.2s;
}

div[data-testid="stButton"] button:hover {
    background: #374151;
}

/* Hide Streamlit defaults */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data & Model (Cached)
# =============================================================================

CATEGORICAL_COLS = ["primary_category", "country_code", "state_code", "region", "city"]
NUMERICAL_COLS   = [
    "funding_total_usd", "funding_rounds", "founded_year",
    "first_funding_year", "last_funding_year",
    "time_to_first_funding_days", "time_between_first_and_last_funding_days",
]


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("big_startup_secsees_dataset.csv", low_memory=False)
    df = df[df["status"].isin(["acquired", "ipo", "closed"])].copy()
    df["target"] = (df["status"].isin(["acquired", "ipo"])).astype(int)

    df["funding_total_usd"] = pd.to_numeric(
        df["funding_total_usd"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )

    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["founded_year"]       = df["founded_at"].dt.year
    df["first_funding_year"] = df["first_funding_at"].dt.year
    df["last_funding_year"]  = df["last_funding_at"].dt.year
    df["time_to_first_funding_days"] = (df["first_funding_at"] - df["founded_at"]).dt.days
    df["time_between_first_and_last_funding_days"] = (
        df["last_funding_at"] - df["first_funding_at"]
    ).dt.days

    df["primary_category"] = (
        df["category_list"].fillna("Unknown").str.split("|").str[0].str.strip()
    )

    return df


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_and_prepare_data()
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERICAL_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            n_estimators=150, max_depth=12,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )),
    ])

    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline


@st.cache_data(show_spinner=False)
def get_category_options():
    df = load_and_prepare_data()
    cats = sorted(df["primary_category"].dropna().unique().tolist())
    return cats


@st.cache_data(show_spinner=False)
def get_field_options(field):
    df = load_and_prepare_data()
    opts = sorted(df[field].dropna().unique().tolist())
    return ["(not specified)"] + [str(o) for o in opts]


# =============================================================================
# Load model with spinner shown only once
# =============================================================================

with st.spinner("Preparing model — this may take a moment on first load…"):
    model       = train_model()
    categories  = get_category_options()
    df_loaded   = load_and_prepare_data()

# =============================================================================
# Sidebar — Inputs
# =============================================================================

with st.sidebar:
    st.markdown("## Input Details")
    st.markdown("Enter the company profile below to generate a prediction.")
    st.markdown("---")

    # --- Company Profile ---
    st.markdown('<div class="sidebar-section">Company Profile</div>', unsafe_allow_html=True)

    selected_category = st.selectbox(
        "Business sector",
        options=categories,
        index=categories.index("Software") if "Software" in categories else 0,
    )

    country_opts = get_field_options("country_code")
    selected_country = st.selectbox(
        "Country",
        options=country_opts,
        index=country_opts.index("USA") if "USA" in country_opts else 0,
    )

    state_opts = get_field_options("state_code")
    selected_state = st.selectbox("State or area", options=state_opts)

    region_opts = get_field_options("region")
    selected_region = st.selectbox("Region", options=region_opts)

    city_opts = get_field_options("city")
    selected_city = st.selectbox("City", options=city_opts)

    # --- Funding Information ---
    st.markdown('<div class="sidebar-section">Funding Information</div>', unsafe_allow_html=True)

    funding_total = st.number_input(
        "Total funding (USD)",
        min_value=0,
        max_value=10_000_000_000,
        value=5_000_000,
        step=500_000,
        help="Total capital raised across all funding rounds.",
    )

    funding_rounds = st.slider(
        "Number of funding rounds",
        min_value=1,
        max_value=20,
        value=2,
    )

    # --- Timeline ---
    st.markdown('<div class="sidebar-section">Timeline</div>', unsafe_allow_html=True)

    founded_year = st.number_input("Year founded", min_value=1980, max_value=2023, value=2010)
    first_year   = st.number_input("First funding year", min_value=1980, max_value=2023, value=2011)
    last_year    = st.number_input("Latest funding year", min_value=1980, max_value=2023, value=2014)

    st.markdown("---")
    predict_button = st.button("Generate Prediction")

# =============================================================================
# Main Content Area
# =============================================================================

col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Startup Outcome Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Predicts whether a startup is likely to succeed (acquisition or IPO) '
        'based on funding history and company profile.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =============================================================================
# Prediction Logic
# =============================================================================

if predict_button:
    # Resolve null sentinel
    def resolve(val):
        return None if val == "(not specified)" else val

    time_to_first    = (first_year - founded_year) * 365 if first_year >= founded_year else 0
    time_between     = (last_year - first_year) * 365    if last_year  >= first_year  else 0

    input_df = pd.DataFrame([{
        "primary_category": selected_category,
        "country_code":     resolve(selected_country),
        "state_code":       resolve(selected_state),
        "region":           resolve(selected_region),
        "city":             resolve(selected_city),
        "funding_total_usd":                          float(funding_total),
        "funding_rounds":                             float(funding_rounds),
        "founded_year":                               float(founded_year),
        "first_funding_year":                         float(first_year),
        "last_funding_year":                          float(last_year),
        "time_to_first_funding_days":                 float(time_to_first),
        "time_between_first_and_last_funding_days":   float(time_between),
    }])

    prediction   = model.predict(input_df)[0]
    probability  = model.predict_proba(input_df)[0]

    success_prob = round(probability[1] * 100, 1)
    fail_prob    = round(probability[0] * 100, 1)

    col1, col2 = st.columns([3, 2])

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-card-success">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Likely Successful</div>
                <div class="result-message">
                    Based on the inputs provided, this company profile is consistent with startups 
                    that have gone on to be acquired or achieve an IPO. The model assigns a 
                    <strong>{success_prob}%</strong> probability of a successful outcome.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{success_prob}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-card-fail">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Likely Unsuccessful</div>
                <div class="result-message">
                    Based on the inputs provided, this company profile is more consistent with 
                    startups that have closed. The model assigns a 
                    <strong>{fail_prob}%</strong> probability of an unsuccessful outcome.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-fail" style="width:{fail_prob}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card" style="height:100%;">
            <div class="result-label">Confidence Breakdown</div>
            <div style="margin-top:0.8rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="font-size:0.85rem;color:#374151;">Successful</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#16a34a;">{success_prob}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{success_prob}%"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.8rem;margin-bottom:0.4rem;">
                    <span style="font-size:0.85rem;color:#374151;">Unsuccessful</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#dc2626;">{fail_prob}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-fail" style="width:{fail_prob}%"></div>
                </div>
            </div>
            <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid #e5e7eb;">
                <div class="result-label">Sector</div>
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{selected_category}</div>
                <div class="result-label" style="margin-top:0.6rem;">Country</div>
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{selected_country}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Default state — empty result area
    st.markdown("""
    <div class="result-card" style="text-align:center;padding:3rem 2rem;">
        <div style="font-size:1.6rem;margin-bottom:0.5rem;color:#d1d5db;">◈</div>
        <div style="font-size:0.95rem;color:#9ca3af;">
            Complete the fields in the sidebar and click <strong>Generate Prediction</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# How It Works (Expander)
# =============================================================================

st.markdown("---")
with st.expander("How this works"):
    st.markdown("""
    **Model**  
    This tool uses a tuned Random Forest classifier trained on historical startup data. 
    The model was selected after comparing it against Logistic Regression and Decision Tree baselines.

    **Training data**  
    Only startups with a confirmed outcome — either acquired, IPO, or closed — were used for training. 
    Companies still in operation were excluded, as their final outcome remains unknown.

    **Features used**  
    The model draws on funding amounts, number of rounds, business sector, geography, 
    founding year, and the timing between funding events.

    **Limitations**  
    No model can predict startup outcomes with certainty. This tool is intended as a 
    decision-support aid and should not be used as the sole basis for investment decisions. 
    Results may reflect biases present in the original dataset, particularly towards 
    US-based technology companies.
    """)
