import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Rossmann Store Sales Forecasting")

st.markdown("""
Predict store sales using the trained Machine Learning model.
Upload a CSV file with the required features.
""")

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_FILE = "rossmann_rf_model.pkl"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        st.error(
            f"Model file '{MODEL_FILE}' not found."
        )
        st.stop()

    return joblib.load(MODEL_FILE)

model = load_model()

# =====================================================
# REQUIRED FEATURES
# =====================================================

required_columns = [

    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",

    "StoreType",
    "Assortment",

    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",

    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",

    "Year",
    "Month",
    "Day",

    "Quarter",
    "WeekNumber",

    "Weekend",

    "Season",
    "CompetitionAge"
]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Project Information")

st.sidebar.info("""
Rossmann Store Sales Forecasting

Upload a CSV file containing the
same features used during model training.
""")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(
            data.head()
        )

        # ------------------------------
        # Validate Columns
        # ------------------------------

        missing_columns = [

            col

            for col in required_columns

            if col not in data.columns

        ]

        if len(missing_columns) > 0:

            st.error(
                f"Missing Columns: {missing_columns}"
            )

            st.stop()

        # ------------------------------
        # Select Required Columns
        # ------------------------------

        X = data[required_columns]

        # ------------------------------
        # Prediction
        # ------------------------------

        predictions = model.predict(X)

        result = data.copy()

        result["PredictedSales"] = predictions

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            result.head()
        )

        # ------------------------------
        # Statistics
        # ------------------------------

        st.subheader(
            "Prediction Summary"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Sales",
            f"{predictions.mean():,.0f}"
        )

        col2.metric(
            "Maximum Sales",
            f"{predictions.max():,.0f}"
        )

        col3.metric(
            "Minimum Sales",
            f"{predictions.min():,.0f}"
        )

        # ------------------------------
        # Plot
        # ------------------------------

        st.subheader(
            "Forecast Visualization"
        )

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        ax.plot(
            predictions
        )

        ax.set_title(
            "Predicted Sales"
        )

        ax.set_xlabel(
            "Record Number"
        )

        ax.set_ylabel(
            "Sales"
        )

        st.pyplot(fig)

        # ------------------------------
        # Download CSV
        # ------------------------------

        csv = result.to_csv(
            index=False
        )

        st.download_button(

            label="Download Predictions",

            data=csv,

            file_name="sales_predictions.csv",

            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

# =====================================================
# SAMPLE SCHEMA
# =====================================================

st.markdown("---")

st.subheader("Required CSV Columns")

st.code(
"\n".join(required_columns)
)

st.markdown("---")

st.caption(
    "Rossmann Store Sales Forecasting Project"
)
