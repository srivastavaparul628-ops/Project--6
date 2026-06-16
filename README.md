# Rossmann Store Sales Forecasting using Machine Learning and Deep Learning

## Project Overview

Rossmann operates over 1,000 retail stores across Europe. Accurate sales forecasting helps improve inventory management, workforce planning, promotional strategy, and business decision-making.

The objective of this project is to predict daily store sales using historical sales data, promotions, holidays, competition information, and store attributes. The project combines Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning, Deep Learning (LSTM), MLOps practices, and deployment through a web application.

---

## Project Objectives

* Analyze historical sales patterns.
* Study the impact of promotions, holidays, competitors, and assortment strategies.
* Engineer meaningful business features.
* Build and compare multiple regression models.
* Develop an LSTM-based forecasting model.
* Track experiments using MLflow.
* Version datasets and models using DVC.
* Deploy the final forecasting solution using Streamlit.

---

## Dataset Description

### train.csv

Contains historical sales records.

Key Features:

* Store
* DayOfWeek
* Date
* Sales
* Customers
* Open
* Promo
* StateHoliday
* SchoolHoliday

### test.csv

Contains future records for prediction.

### store.csv

Contains store metadata.

Key Features:

* StoreType
* Assortment
* CompetitionDistance
* CompetitionOpenSinceMonth
* CompetitionOpenSinceYear
* Promo2
* Promo2SinceWeek
* Promo2SinceYear
* PromoInterval

---

## Exploratory Data Analysis

The following analyses were performed:

### Promotion Analysis

* Promotion distribution in train and test datasets
* Promotion impact on sales
* Promotion impact on customer footfall

### Holiday Analysis

* Before holiday sales
* During holiday sales
* After holiday sales
* Christmas analysis
* Easter analysis
* Public holiday analysis

### Competitor Analysis

* Competition distance
* Competitor age
* Impact on store sales

### Store Operations Analysis

* Weekend stores
* Weekday stores
* Revenue comparison

### Assortment Analysis

* Basic assortment
* Extra assortment
* Extended assortment

---

## Feature Engineering

Created Features:

* Day
* Month
* Year
* Weekday
* Weekend
* Quarter
* Week Number
* Season
* Festival Flag
* Days Before Holiday
* Days After Holiday
* Promo Duration
* Competitor Age

---

## Data Preprocessing

Implemented using:

* Pipeline()
* ColumnTransformer()

Processing Steps:

1. Missing Value Imputation
2. One-Hot Encoding
3. Feature Scaling
4. Train-Test Split

---

## Machine Learning Models

Implemented Models:

1. Random Forest Regressor
2. Extra Trees Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor

Evaluation Metrics:

* RMSE
* MAE
* RMSPE
* R² Score

---

## Deep Learning Model

Implemented:

### LSTM Forecasting Model

Steps:

* Time Series Conversion
* Stationarity Testing (ADF & KPSS)
* Differencing
* ACF/PACF Analysis
* Sliding Window Creation
* Data Scaling (-1,1)
* Two-Layer LSTM Network

Architecture:

Input
↓
LSTM (50)
↓
LSTM (25)
↓
Dense (1)

---

## MLOps

### MLflow

Tracked:

* Parameters
* Metrics
* Models

### DVC

Tracked:

* Dataset Versions
* Model Versions

---

## Deployment

Framework:

* Streamlit

Application Features:

* Store ID Input
* CSV Upload
* Holiday Selection
* Promotion Selection
* Sales Forecasting
* Forecast Visualization
* Downloadable Results

---

## Project Structure

Rossmann_Sales_Forecasting/

├── notebooks/

│ ├── Part1_EDA.ipynb

│ ├── Part2_ML_Modeling.ipynb

│ └── Part3_LSTM_MLOps.ipynb

├── data/

│ ├── train.csv

│ ├── test.csv

│ └── store.csv

├── models/

│ ├── best_model.pkl

│ └── rossmann_lstm.h5

├── streamlit/

│ └── app.py

├── reports/

│ ├── Final_Report.pdf

│ └── Presentation.pptx

└── README.md

---

## Installation

Create Environment

```bash
pip install -r requirements.txt
```

Run Notebook

```bash
jupyter notebook
```

Run Streamlit Application

```bash
streamlit run app.py
```

---

## Results

Best Performing Model:

* [Update after training]

Evaluation Metrics:

* RMSE: [Update]
* MAE: [Update]
* RMSPE: [Update]
* R²: [Update]

## Author

Name: [Parul Srivastava]

Project: Rossmann Store Sales Forecasting


Year: 2026

