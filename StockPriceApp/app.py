import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

st.title("📈 Stock Price Prediction App")
st.write("Predict future stock prices using historical data")

# Step 1: Get Stock Data
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
forecast_days = st.slider("Days to Predict", 1, 60, 30)

if ticker:
    data = yf.download(ticker, start='2018-01-01', end='2025-12-31')
    data.reset_index(inplace=True)
    st.subheader("Recent Stock Data")
    st.dataframe(data.tail())

    # Step 2: Prepare Data
    data_close = data[['Close']].copy()
    data_close['Prediction'] = data_close['Close'].shift(-forecast_days)

    # Check if enough data is available
    if len(data_close) > forecast_days:
        X = np.array(data_close.drop(['Prediction'], axis=1))[:-forecast_days]
        y = np.array(data_close['Prediction'])[:-forecast_days]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")

        # Step 3: Predict Future Prices
        X_forecast = np.array(data_close.drop(['Prediction'], axis=1))[-forecast_days:]
        forecast_prediction = model.predict(X_forecast)

        # Step 4: Visualize
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Price'))
        forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_days+1)[1:]
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prediction, name='Predicted Price'))
        fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price ($)')
        st.plotly_chart(fig)
    else:
        st.warning("Not enough data to make predictions. Try reducing the number of forecast days.")