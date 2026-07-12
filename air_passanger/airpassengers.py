import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

st.title("✈️ Air Passenger Forecasting Dashboard")

# ----------------------------------------------------------------------
# File uploader (with fallback to bundled dataset)
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Validation step
    required_cols = {"Month", "#Passengers"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain 'Month' and '#Passengers' columns.")
        st.stop()  # Stop execution if file is invalid

    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df = df.asfreq('MS')
else:
    # Fallback to bundled dataset using os.path
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "air_passengers.csv")
    df = pd.read_csv(file_path)

    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df = df.asfreq('MS')


# ----------------------------------------------------------------------
# Cached model fitting helpers
# ----------------------------------------------------------------------
@st.cache_resource
def fit_sarima(series):
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

@st.cache_resource
def fit_hw(series, trend, seasonal):
    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=12)
    return model.fit()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["EDA", "Model Comparison", "Forecast Visualization"])

# ----------------------------------------------------------------------
# Tab 1: EDA
# ----------------------------------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis")

    st.subheader("Raw Time Series")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['#Passengers'])
    ax.set_title("Air Passenger Counts")
    st.pyplot(fig)

    result = adfuller(df['#Passengers'])
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])

    st.subheader("Seasonal Decomposition")
    decomposition = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    st.pyplot(fig)

    st.subheader("ACF and PACF")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(df['#Passengers'], lags=40, ax=axes[0])
    plot_pacf(df['#Passengers'], lags=40, ax=axes[1])
    st.pyplot(fig)

# ----------------------------------------------------------------------
# Tab 2: Model Comparison
# ----------------------------------------------------------------------
with tab2:
    st.header("SARIMA vs Holt-Winters")

    train = df.iloc[:-12]
    test = df.iloc[-12:]

    sarima_fit = fit_sarima(train['#Passengers'])
    sarima_pred = sarima_fit.forecast(12)

    hw_fit = fit_hw(train['#Passengers'], 'add', 'mul')
    hw_pred = hw_fit.forecast(12)

    sarima_rmse = np.sqrt(mean_squared_error(test['#Passengers'], sarima_pred))
    sarima_mape = mean_absolute_percentage_error(test['#Passengers'], sarima_pred)
    hw_rmse = np.sqrt(mean_squared_error(test['#Passengers'], hw_pred))
    hw_mape = mean_absolute_percentage_error(test['#Passengers'], hw_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("SARIMA RMSE", f"{sarima_rmse:.2f}")
        st.metric("SARIMA MAPE", f"{sarima_mape:.2%}")
    with col2:
        st.metric("Holt-Winters RMSE", f"{hw_rmse:.2f}")
        st.metric("Holt-Winters MAPE", f"{hw_mape:.2%}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['#Passengers'], label="Train")
    ax.plot(test.index, test['#Passengers'], label="Actual", color="black", marker="o")
    ax.plot(test.index, sarima_pred, label="SARIMA Forecast", linestyle="--", color="green")
    ax.plot(test.index, hw_pred, label="Holt-Winters Forecast", linestyle="--", color="red")
    ax.legend()
    ax.set_title("Forecast vs Actual (last 12 months)")
    st.pyplot(fig)

# ----------------------------------------------------------------------
# Tab 3: Forecast Visualization
# ----------------------------------------------------------------------
with tab3:
    st.header("Extended Forecast")

    horizon = st.slider("Forecast horizon (months)", 12, 60, 36)

    # Model toggle
    model_choice = st.radio("Select forecasting model", ["Holt-Winters", "SARIMA"])

    if model_choice == "Holt-Winters":
        final_fit = fit_hw(df['#Passengers'], 'add', 'mul')
        forecast = final_fit.forecast(horizon)
    else:
        final_fit = fit_sarima(df['#Passengers'])
        forecast = final_fit.forecast(horizon)

    # Confidence intervals (approximate)
    residuals = df['#Passengers'] - final_fit.fittedvalues
    std = np.std(residuals)
    upper = forecast + 1.96 * std
    lower = forecast - 1.96 * std

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['#Passengers'], label="Historical")
    ax.plot(forecast.index, forecast, label=f"{model_choice} Forecast", color="red")
    ax.fill_between(forecast.index, lower, upper, color="pink", alpha=0.3, label="95% CI")
    ax.axvline(df.index[-1], color="gray", linestyle=":", label="Forecast start")
    ax.legend()
    ax.set_title(f"{model_choice} Forecast ({horizon} months ahead)")
    st.pyplot(fig)

    st.write("Forecasted values:")
    st.dataframe(forecast.rename("Forecasted Passengers"))

    # Download button
    csv = forecast.to_csv().encode('utf-8')
    st.download_button(
        label="Download forecast as CSV",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv",
    )

    st.caption(
        "Note: the 95% CI band is an approximation (±1.96 × residual std dev). "
        "Treat it as a rough uncertainty guide, especially further out in the horizon."
    )
