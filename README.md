# Air_Passengers
✈️ Air Passenger Forecasting Dashboard
An interactive Streamlit app for time series forecasting using the classic Air Passenger dataset.
The dataset is univariate (monthly passenger counts), making it ideal for demonstrating forecasting models like SARIMA and Holt‑Winters.

📊 Features
Exploratory Data Analysis (EDA)

Raw time series visualization

Seasonal decomposition

ADF test for stationarity

ACF & PACF plots

Model Comparison

SARIMA vs Holt‑Winters forecasts

RMSE and MAPE metrics

Side‑by‑side forecast vs actuals

Forecast Visualization

Extended forecasts (12–60 months)

Confidence intervals (approximate ±1.96 × residual std)

Model toggle (choose SARIMA or Holt‑Winters)

Download forecast results as CSV

User Interaction

Upload your own CSV dataset

Built‑in validation (must contain Month and #Passengers columns)

Fallback to bundled dataset if no file is uploaded

📂 Project Structure
Code
Air_Passengers/
├── air_passanger/
│   ├── airpassengers.py        # Streamlit app
│   ├── air_passengers.csv      # Default dataset (univariate)
│   └── requirements.txt        # Dependencies
└── .streamlit/
    └── config.toml             # Theme configuration
⚙️ Installation & Usage
1. Clone the repo
bash
git clone https://github.com/<your-username>/Air_Passengers.git
cd Air_Passengers/air_passanger
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the app
bash
streamlit run airpassengers.py
📌 CSV Upload Requirements
If you upload your own dataset, it must contain at least two columns:

Month → dates in YYYY-MM format

#Passengers → numeric values

If these columns are missing, the app will show an error message and stop gracefully.

🚀 Deployment
Push your repo to GitHub.

Deploy on Streamlit Cloud.

Select airpassengers.py as the entry point.

Streamlit Cloud will install dependencies from requirements.txt and serve your app online.

🎨 Theme
The app uses a custom theme defined in .streamlit/config.toml.
You can adjust colors, fonts, and backgrounds there.

📖 Notes
The dataset is univariate (only passenger counts over time).

Confidence intervals are approximate and widen with longer horizons.

SARIMA and Holt‑Winters are compared for demonstration; you can extend with other models.
