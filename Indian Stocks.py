import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# =========================
# MODULE 1: DATA ACQUISITION & CLEANING (INDIAN MARKET)
# =========================
def get_stock_data(symbol, start="2024-01-01", end="2026-02-07"):
    print("Fetching Indian stock data...")
    data = yf.download(symbol, start=start, end=end)
    data = data[['Close']]
    data.dropna(inplace=True)
    return data


# =========================
# MODULE 2: FEATURE ENGINEERING
# =========================
def add_features(data):
    print("Generating technical indicators...")
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data


# =========================
# MODULE 3: AI PREDICTION ENGINE (RandomForest)
# =========================
def train_model(data):
    X = data[['MA10', 'MA20', 'Return']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    print("Training AI model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    print("Model RMSE:", round(rmse, 2))

    return model, X_test, y_test, predictions


# =========================
# MODULE 4: RISK & DECISION ENGINE
# =========================
def risk_and_decision_engine(data, predicted_price):
    last_price = float(data['Close'].iloc[-1])
    predicted_price = float(predicted_price)

    volatility = float(data['Return'].std() * 100)

    if predicted_price > last_price and volatility < 2:
        decision = "BUY"
    elif predicted_price < last_price and volatility >= 2:
        decision = "SELL"
    else:
        decision = "HOLD"

    return last_price, volatility, decision


# =========================
# MODULE 5: VISUALIZATION & REPORT
# =========================
def visualize_results(y_test, predictions, stock_symbol):
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"{stock_symbol} Price Prediction (Indian Market)")
    plt.xlabel("Time")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.show()


# =========================
# MAIN PROGRAM
# =========================
def main():
    print("INDIAN STOCK MARKET AI PREDICTION SYSTEM")
    print("Example symbols: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")

    stock_symbol = input("Enter NSE Stock Symbol: ").upper()

    # Module 1
    data = get_stock_data(stock_symbol)

    # Module 2
    data = add_features(data)

    # Module 3
    model, X_test, y_test, predictions = train_model(data)

    # Predict next trading day price
    last_row = data[['MA10', 'MA20', 'Return']].iloc[-1].values.reshape(1, -1)
    next_day_price = model.predict(last_row)[0]

    # Module 4
    last_price, volatility, decision = risk_and_decision_engine(data, next_day_price)

    # Module 5
    visualize_results(y_test, predictions, stock_symbol)

    print("\n========== FINAL REPORT ==========")
    print("Stock:", stock_symbol)
    print("Last Closing Price (INR):", round(last_price, 2))
    print("Predicted Next Price (INR):", round(next_day_price, 2))
    print("Market Volatility (%):", round(volatility, 2))
    print("Investment Decision:", decision)
    print("==================================")


main()
