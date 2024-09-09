#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Page configuration
st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")

# Title
st.title('Tesla Stock Analysis & Forecasting (2013)')

# File upload option
uploaded_file = st.file_uploader("Upload Tesla CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter the data for 2013
    tesla_data = data[data['Date'].dt.year == 2013]

    # Set the 'Date' as index
    tesla_data.set_index('Date', inplace=True)

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    options = st.sidebar.radio("Select analysis", ['Summary', 'Time Series', 'Moving Averages', 'Bollinger Bands', 'MACD', 'ACF', 'ARIMA', 'SARIMA', 'SARIMAX'])

    # Show data summary
    if options == 'Summary':
        st.header('Data Summary')
        st.write(tesla_data.describe())
        st.write(tesla_data.head())
        st.write(tesla_data.tail())
        
        # Time series plot of closing prices
    if options == 'Time Series':
        st.header('Tesla Closing Price Over Time')
        plt.figure(figsize=(12, 5))
        plt.plot(tesla_data.index, tesla_data['Close'], label='Close Price')
        plt.title('Tesla Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

    # Moving averages plot
    if options == 'Moving Averages':
        st.header('Moving Averages (50-Day and 200-Day)')
        tesla_data['MA50'] = tesla_data['Close'].rolling(window=50).mean()
        tesla_data['MA200'] = tesla_data['Close'].rolling(window=200).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(tesla_data['Close'], label='Close Price')
        plt.plot(tesla_data['MA50'], label='50-Day MA', color='red')
        plt.plot(tesla_data['MA200'], label='200-Day MA', color='green')
        plt.title('Tesla Stock Prices with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    # Bollinger Bands plot
    if options == 'Bollinger Bands':
        st.header('Bollinger Bands')
        tesla_data['MA20'] = tesla_data['Close'].rolling(window=20).mean()
        tesla_data['BB_up'] = tesla_data['MA20'] + 2 * tesla_data['Close'].rolling(window=20).std()
        tesla_data['BB_down'] = tesla_data['MA20'] - 2 * tesla_data['Close'].rolling(window=20).std()
        plt.figure(figsize=(12, 6))
        plt.plot(tesla_data['Close'], label='Close Price')
        plt.plot(tesla_data['MA20'], label='20-Day MA', color='blue')
        plt.plot(tesla_data['BB_up'], label='Upper Bollinger Band', color='red')
        plt.plot(tesla_data['BB_down'], label='Lower Bollinger Band', color='green')
        plt.fill_between(tesla_data.index, tesla_data['BB_down'], tesla_data['BB_up'], color='gray', alpha=0.2)
        plt.title('Tesla Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

    # MACD plot
    if options == 'MACD':
        st.header('Moving Average Convergence Divergence (MACD)')
        tesla_data['EMA12'] = tesla_data['Close'].ewm(span=12, adjust=False).mean()
        tesla_data['EMA26'] = tesla_data['Close'].ewm(span=26, adjust=False).mean()
        tesla_data['MACD'] = tesla_data['EMA12'] - tesla_data['EMA26']
        tesla_data['Signal'] = tesla_data['MACD'].ewm(span=9, adjust=False).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(tesla_data['MACD'], label='MACD', color='blue')
        plt.plot(tesla_data['Signal'], label='Signal Line', color='red')
        plt.title('Tesla MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        st.pyplot(plt)

    # ACF plot
    if options == 'ACF':
        st.header('Autocorrelation Function (ACF)')
        plt.figure(figsize=(10, 5))
        plot_acf(tesla_data['Close'].dropna(), lags=50)
        plt.title('Autocorrelation Function for Close Price')
        st.pyplot(plt)

    # ARIMA modeling
    if options == 'ARIMA':
        st.header('ARIMA Model for Forecasting')

        # Train-test split (80%-20%)
        split_index = int(len(tesla_data) * 0.8)
        train = tesla_data.iloc[:split_index]
        test = tesla_data.iloc[split_index:]

        model = ARIMA(train['Close'], order=(1, 1, 1))
        model_fit = model.fit()

        forecast, conf_int = model_fit.get_forecast(steps=len(test)).summary_frame().iloc[:, :2].values.T
        forecast_series = pd.Series(forecast, index=test.index)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train['Close'], label='Train')
        plt.plot(test['Close'], label='Test')
        plt.plot(forecast_series, label='Forecast', color='green', linestyle='--')
        plt.fill_between(forecast_series.index, conf_int[0], conf_int[1], color='gray', alpha=0.2)
        plt.title('ARIMA Forecast')
        plt.legend()
        st.pyplot(plt)

    # SARIMA model
    if options == 'SARIMA':
        st.header('SARIMA Model for Forecasting')

        # Train-test split (80%-20%)
        train = tesla_data['Close'][:'2013-10-18']
        test = tesla_data['Close']['2013-10-18':]

        # Auto ARIMA to find the best order and seasonal order
        auto_model = auto_arima(train, seasonal=True, m=12, stepwise=True)

        # Make predictions
        forecast, conf_int = auto_model.predict(n_periods=len(test), return_conf_int=True)
        forecast_series = pd.Series(forecast, index=test.index)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(forecast_series, label='Forecast', linestyle='--', color='green')
        plt.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
        plt.title('SARIMA Forecast')
        plt.legend()
        st.pyplot(plt)

    # SARIMAX model
    if options == 'SARIMAX':
        st.header('SARIMAX Model for Forecasting')

        # Train-test split
        train = tesla_data['Close']['2013-01-01':'2013-10-18']
        test = tesla_data['Close']['2013-10-18':]

        # SARIMAX model
        sarimax_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

        # Forecast
        forecast = sarimax_model.get_forecast(steps=len(test))
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(forecast_values, label='Forecast', linestyle='--', color='green')
        plt.fill_between(forecast_values.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.2)
        plt.title('SARIMAX Forecast')
        plt.legend()
        st.pyplot(plt)


# In[ ]:




