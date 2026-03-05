
import numpy as np
import pandas as pd
import datetime
import requests

def get_bitcoin_historical_prices(days=365):
    """
    (Unused now) Previously downloaded BTC price data from CoinGecko.
    Kept for compatibility but not used.
    """
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["Timestamp", "Close Price"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.normalize()
    df = df[["Date", "Close Price"]].set_index("Date")

    return df


def extract_features(days=365):
    """
    Generate Bitcoin technical indicator features using BitstampData.csv
    instead of the CoinGecko API.
    """

    # --- CHANGED LINE ---
    df = pd.read_csv("BitstampData.csv")

    # Keep only the close price and rename to match training pipeline
    df = df[['Close']]
    df.columns = ['Close Price']

    price = df["Close Price"]

    features = pd.DataFrame(index=df.index)

    # 1. Exponential Moving Average
    features["EMA_12"] = price.ewm(span=12).mean()

    # 2. Rate of Change (momentum)
    features["ROC_10"] = price.pct_change(10)

    # 3. Momentum
    features["Momentum_5"] = price.diff(5)

    # 4. Relative Strength Index (RSI)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    features["RSI_14"] = 100 - (100 / (1 + rs))

    # 5. Moving Average
    features["MA_20"] = price.rolling(20).mean()

    features = features.dropna()

    return features
