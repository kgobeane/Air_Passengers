# ✈️ Air Passenger Forecasting Dashboard

An interactive Streamlit application for time series forecasting, built on the classic Air Passengers dataset. The dataset is a univariate monthly series (passenger counts), making it a clean, well-known benchmark for demonstrating and comparing classical forecasting methods such as SARIMA and Holt-Winters Exponential Smoothing.

## Overview

The dashboard walks through a complete forecasting workflow — exploratory analysis, model comparison against held-out data, and extended future forecasting — in a single interface. Users can work with the bundled dataset or upload their own compatible time series.

## Features

### Exploratory Data Analysis
- Raw time series visualization
- Augmented Dickey-Fuller (ADF) test for stationarity
- Multiplicative seasonal decomposition (trend, seasonality, residuals)
- ACF and PACF plots to guide model order selection

### Model Comparison
- SARIMA and Holt-Winters forecasts evaluated on a held-out 12-month test set
- RMSE and MAPE reported for both models
- Side-by-side visualization of forecasts against actual values

### Forecast Visualization
- Extended forecasts from 12 to 60 months ahead
- Toggle between SARIMA and Holt-Winters as the forecasting model
- Confidence intervals — model-derived for SARIMA, an approximate ±1.96 × residual standard deviation band for Holt-Winters
- Forecast results downloadable as CSV

### Data Input
- Upload a custom CSV dataset directly in the app
- Built-in validation: the file must contain `Month` (parseable date) and `#Passengers` (numeric) columns, with no gaps once aligned to a monthly frequency
- Falls back to the bundled dataset automatically if no file is uploaded

## Project Structure

```
Air_Passengers/
├── air_passanger/
│   ├── airpassengers.py      # Streamlit application
│   ├── air_passengers.csv    # Default dataset (univariate)
│   └── requirements.txt      # Python dependencies
└── .streamlit/
    └── config.toml           # Theme configuration
```

## Installation and Usage

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/Air_Passengers.git
cd Air_Passengers/air_passanger
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run airpassengers.py
```

## CSV Upload Requirements

Custom datasets must include at minimum:

| Column | Format |
|---|---|
| `Month` | Date, e.g. `YYYY-MM` |
| `#Passengers` | Numeric |

Files missing these columns, containing unparseable dates, or with gaps after alignment to a monthly frequency will be rejected with a descriptive error message, and the app will stop gracefully rather than proceeding with invalid data.

## Deployment

1. Push the repository to GitHub.
2. Create a new app on [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set `airpassengers.py` as the entry point.
4. Streamlit Cloud installs dependencies from `requirements.txt` and serves the app automatically.

## Theming

Visual theming (colors, fonts, background) is configured in `.streamlit/config.toml` and can be adjusted independently of the application code.

## Notes and Limitations

- The dataset is univariate — forecasts are based solely on historical passenger counts, with no exogenous variables.
- Confidence intervals widen with longer forecast horizons; SARIMA's intervals are statistically derived, while Holt-Winters' are an approximation and should be treated as directional rather than exact.
- SARIMA and Holt-Winters are included for demonstration purposes; the architecture accommodates additional models (e.g. Prophet, ARIMA variants, or machine learning-based approaches) for future extension.
