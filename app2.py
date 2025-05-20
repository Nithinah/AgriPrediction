import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
from googletrans import Translator

# Streamlit Setup
st.set_page_config(page_title="Agri Commodity Price Forecasting App", layout="wide")

# Language Setup
translator = Translator()
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Malayalam": "ml"
}
selected_language = st.sidebar.selectbox("🌐 Choose Language", list(language_map.keys()))
target_lang = language_map[selected_language]

@st.cache_data(show_spinner=False)
def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        return translator.translate(text, dest=lang).text
    except:
        return text

def t(text):
    return translate_text(text, target_lang)

st.title(t("🌾 Agri Commodity Price Forecasting App"))
uploaded_file = st.sidebar.file_uploader(t("📤 Upload Agri Commodity CSV"), type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(t("✅ File uploaded successfully!"))

    with st.expander(t("📄 Show CSV Columns")):
        st.write(df.columns.tolist())

    # Detect columns
    def detect_column(possible_names):
        for col in df.columns:
            if col.strip() in possible_names:
                return col
        return None

    date_col = detect_column(['Date', 'date', 'DATE', 'Arrival_Date', 'arrival_date'])
    commodity_col = detect_column(['Commodity', 'commodity'])
    state_col = detect_column(['State', 'state', 'Origin_State'])

    if not all([date_col, commodity_col, state_col]):
        st.error(t("❌ Required columns (`Date`, `Commodity`, `State`) not found."))
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', commodity_col: 'Commodity', state_col: 'State'})
    df = df.dropna(subset=['Date'])

    # Inputs
    st.sidebar.markdown("## " + t("🧩 Select Inputs"))
    selected_commodity = st.sidebar.selectbox(t("Select Commodity"), sorted(df['Commodity'].dropna().unique()))
    selected_state = st.sidebar.selectbox(t("Select State"), sorted(df[df['Commodity'] == selected_commodity]['State'].dropna().unique()))

    # Filter
    filtered_df = df[(df['Commodity'] == selected_commodity) & (df['State'] == selected_state)].copy()
    if filtered_df.empty:
        st.error(t("❌ No data available for the selected commodity and state."))
        st.stop()

    grouped_df = filtered_df.groupby('Date').mean(numeric_only=True).reset_index().sort_values('Date')

    if 'Modal Price (Rs./Quintal)' in grouped_df.columns:
        target_col = 'Modal Price (Rs./Quintal)'
    else:
        target_col = st.sidebar.selectbox(t("Select Price Column"), grouped_df.select_dtypes(include='number').columns.tolist())

    grouped_df['Target'] = grouped_df[target_col]
    grouped_df.dropna(inplace=True)

    # Normalize
    scaler = MinMaxScaler()
    grouped_df['Scaled'] = scaler.fit_transform(grouped_df[['Target']])

    # Sequences
    def create_sequences(data, window_size=3):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    window_size = 3
    data_series = grouped_df['Scaled'].values
    X, y = create_sequences(data_series, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Forecast
    def forecast_next_days(model, last_sequence, steps, scaler):
        future_preds = []
        seq = last_sequence.copy()
        for _ in range(steps):
            pred = model.predict(seq.reshape(1, window_size, 1), verbose=0)[0][0]
            future_preds.append(pred)
            seq = np.append(seq[1:], pred)
        return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    forecast_prices = forecast_next_days(model, data_series[-window_size:], 3, scaler)

    # Plot
    st.subheader(t(f"📈 Price Trend for {selected_commodity} in {selected_state}"))

    full_preds = model.predict(X).flatten()
    full_preds_actual = scaler.inverse_transform(full_preds.reshape(-1, 1)).flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grouped_df['Date'][window_size:], full_preds_actual, label=t("Predicted Price"), linestyle='--')
    ax.plot(grouped_df['Date'], grouped_df['Target'], label=t("Actual Price"), alpha=0.7)
    ax.set_xlabel(t("Date"))
    ax.set_ylabel(t("Price (Rs./Quintal)"))
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Forecast Table
    st.subheader(t("🔮 Forecast for Next 3 Days"))
    forecast_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]

    forecast_df = pd.DataFrame({
        t('Forecast Date'): forecast_dates,
        t('Predicted Price (Rs./Quintal)'): [round(p, 2) for p in forecast_prices],
        t('Predicted Price (Rs./Kg)'): [round(p / 100, 2) for p in forecast_prices]
    })

    st.table(forecast_df)
    st.success(f"✅ {t('Model RMSE')}: {rmse:.2f}")

    # Insight Generator
    def generate_insight(actual_prices, predicted_prices):
        trend = "increasing" if predicted_prices[-1] > actual_prices[-3] else "decreasing"
        change = predicted_prices[-1] - actual_prices[-3]
        percent_change = (change / actual_prices[-3]) * 100
        insight = (
            f"The price trend for {selected_commodity} in {selected_state} is currently {trend}. "
            f"In the next 3 days, the price is expected to change by approximately ₹{change:.2f} "
            f"({percent_change:.2f}%)."
        )
        return insight

    # Display insight
    st.subheader(t("🧠 Model Inference"))
    insight = generate_insight(grouped_df['Target'].values, forecast_prices)
    st.info(translate_text(insight, target_lang))
