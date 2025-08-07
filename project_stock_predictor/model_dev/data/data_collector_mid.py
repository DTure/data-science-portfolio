import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ta
import time
import requests


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

tickers = get_sp500_tickers()

# 📅 Дата діапазон: 3 роки
end_date = datetime.strptime("2025-06-20", "%Y-%m-%d")
start_date = end_date - timedelta(days=3 * 365)

# 📥 Функція завантаження історичних даних
def collect_data(tickers, start, end):
    all_data = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for ticker in tickers:
        try:
            print(f"🔍 Завантаження даних для {ticker}...")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&period1={int(datetime.strptime(start, '%Y-%m-%d').timestamp())}&period2={int(datetime.strptime(end, '%Y-%m-%d').timestamp())}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()["chart"]["result"][0]
                df = pd.DataFrame(data["indicators"]["quote"][0])
                df["Date"] = pd.to_datetime(data["timestamp"], unit="s")
                df["Ticker"] = ticker
                df = df[["Date", "Ticker", "open", "high", "low", "close", "volume"]]
                all_data.append(df)
            else:
                print(f"❌ Помилка при обробці {ticker}: {response.status_code}")
            time.sleep(0.8)

        except Exception as e:
            print(f"❌ Помилка при обробці {ticker}: {e}")
            continue

    return pd.concat(all_data) if all_data else pd.DataFrame()

# 🧠 Додавання технічних індикаторів для середньострокового прогнозу
def add_mid_term_indicators(df):
    df = df.copy()
    df = df.sort_values(by="Date")

    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["Momentum"] = ta.momentum.roc(df["close"], window=10)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    return df

# 🧪 Збір даних
raw_data = collect_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# ➕ Додавання індикаторів
processed_data = raw_data.groupby("Ticker").apply(lambda x: add_mid_term_indicators(x)).reset_index(drop=True)
processed_data.dropna(inplace=True)

# 💾 Збереження
processed_data.to_csv("data/mid_term.csv", index=False)
print("✅ Дані для середньострокового прогнозу збережені.")
