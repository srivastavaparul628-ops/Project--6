import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Rossmann Store Sales Forecasting")

st.markdown(
    """
    Upload a CSV file containing the same features
    used during model training.
    """
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("rossmann_rf_model.pkl")

model = load_model()

# ----------------------------------
# FILE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(data.head())

    # -----------------------------
    # Prediction
    # -----------------------------

    predictions = model.predict(data)

    result = data.copy()

    result["PredictedSales"] = predictions

    st.subheader("Prediction Results")

    st.dataframe(
        result.head()
    )

    # -----------------------------
    # Plot Forecast
    # -----------------------------

    st.subheader(
        "Forecast Plot"
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        result["PredictedSales"],
        linewidth=2
    )

    ax.set_title(
        "Predicted Sales"
    )

    ax.set_xlabel(
        "Observation"
    )

    ax.set_ylabel(
        "Sales"
    )

    st.pyplot(fig)

    # -----------------------------
    # Download Results
    # -----------------------------

    csv = result.to_csv(
        index=False
    )

    st.download_button(

        label="Download Predictions",

        data=csv,

        file_name="sales_predictions.csv",

        mime="text/csv"
    )

# ----------------------------------
# SAMPLE INPUT FORMAT
# ----------------------------------

st.markdown("---")

st.subheader("Required Input Columns")

st.code(
"""
Store
DayOfWeek
Open
Promo
StateHoliday
SchoolHoliday
StoreType
Assortment
CompetitionDistance
CompetitionOpenSinceMonth
CompetitionOpenSinceYear
Promo2
Promo2SinceWeek
Promo2SinceYear
PromoInterval
Year
Month
Day
Quarter
WeekNumber
Weekend
Season
CompetitionAge
"""
)

st.markdown("---")

st.markdown(
    "Rossmann Sales Forecasting Project"
)

