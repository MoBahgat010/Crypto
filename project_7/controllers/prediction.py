import yfinance as yf
import pandas as pd
import numpy as np  


def compute_rsi(series, period=14):
    """Compute Relative Strength Index"""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """Compute Average True Range"""
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    return df['TR'].rolling(window=period).mean()
def fetch_crypto_data(symbol, interval, lookback):
    """Fetch historical crypto data from Yahoo Finance"""
    df = yf.download(symbol, period="60d", interval=interval)  # Adjusted period to a valid format
    df.reset_index(inplace=True)
    df = df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df.set_index("timestamp", inplace=True)
    return df


def add_technical_indicators(df):
    """Add technical indicators to dataset"""
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['ATR'] = compute_atr(df)
    df.dropna(inplace=True)
    return df

