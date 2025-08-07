import pandas as pd
from pandas.tseries.offsets import BDay 
from datetime import datetime, timedelta
import ta
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, ticker: str):
        self.ticker = self._clean_ticker(ticker)    
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        self.validate_ticker()

    def _clean_ticker(self, ticker: str) -> str:
        """Formats the ticker to a standard form."""
        return ticker.replace('.', '-').upper().strip()

    def validate_ticker(self):
        """Checks if the ticker exists."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise ValueError(f"Ticker {self.ticker} not found")
        except Exception as e:
            raise ValueError(f"Error validating ticker {self.ticker}: {e}")

    def get_latest_trading_date(self, max_lookback_days: int = 7) -> datetime.date:
        """Finds the most recent date with trading data."""
        today = datetime.now().date()
        
        for delta in range(max_lookback_days):
            check_date = today - timedelta(days=delta)
            logger.info(f"Checking date {check_date} for {self.ticker}")
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}?" \
            f"interval=1d&period1={int(datetime.combine(check_date - timedelta(days=1), datetime.min.time()).timestamp())}" \
            f"&period2={int(datetime.combine(check_date + timedelta(days=1), datetime.min.time()).timestamp())}"
  
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if 'timestamp' in data['chart']['result'][0]:
                    logger.info(f"Data found for {self.ticker} on {check_date}")
                    return check_date
            
            time.sleep(0.5)
        
        raise Exception(f"Failed to find trading data in the last {max_lookback_days} days for {self.ticker}")

    def download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            logger.info(f"Downloading data for {self.ticker} from {start_date} to {end_date}")
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}?" \
                  f"interval=1d&period1={int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())}" \
                  f"&period2={int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"HTTP error: {response.status_code}")
                
            data = response.json()
            
            if not data['chart']['result']:
                raise Exception("Empty result received")
                
            result = data['chart']['result'][0]
            df = pd.DataFrame(result['indicators']['quote'][0])
            df['date'] = pd.to_datetime(result['timestamp'], unit='s')
            df['ticker'] = self.ticker
            
            df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
          
            if df.empty:
                raise Exception("Received an empty DataFrame")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error downloading data for {self.ticker}: {e}")

    def add_short_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by="date")
        
        if len(df) >= 7:
            df["SMA_5"] = ta.trend.sma_indicator(df["close"], window=5)
            df["RSI_7"] = ta.momentum.rsi(df["close"], window=7)
            bb = ta.volatility.BollingerBands(df["close"], window=7)
            df["BB_high"] = bb.bollinger_hband()
            df["BB_low"] = bb.bollinger_lband()
        else:
            logger.warning(f"Not enough data to calculate indicators ({len(df)} rows)")
            
        return df
    
    def add_mid_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by="date")
        
        if len(df) >= 7:
            df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
            df["SMA_50"] = ta.trend.sma_indicator(df["close"], window=50)
            df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)    
            bb = ta.volatility.BollingerBands(df["close"], window=20)
            df["BB_high"] = bb.bollinger_hband()
            df["BB_low"] = bb.bollinger_lband()
            df["MACD"] = ta.trend.macd_diff(df["close"])
            df["Momentum"] = ta.momentum.roc(df["close"], window=10)
            df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        else:
            logger.warning(f"Not enough data to calculate indicators ({len(df)} rows)")
            
        return df

    def collect(self, mode: str = "Short-term") -> pd.DataFrame:
        try:
            if mode == "Short-term":
                window_days = 30
                buffer_days = 20
                indicator_func = self.add_short_indicators
                forecast_horizon = 7
            elif mode == "Medium-term":
                window_days = 90
                buffer_days = 60
                indicator_func = self.add_mid_indicators
                forecast_horizon = 30
            else:
                raise ValueError("Unknown mode. Use 'Short-term' or 'Mid-term'.")

            end_date = self.get_latest_trading_date()
            total_days = window_days + buffer_days
            start_date = (pd.Timestamp(end_date) - BDay(total_days)).date()

            logger.info(f"Collecting data ({mode}) for {self.ticker} from {start_date} to {end_date}")
            df = self.download_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            df = indicator_func(df)
            df = df.dropna()

            if df.empty:
                logger.warning(f"Empty DataFrame after processing for {self.ticker}")
                return df, [] 

            df = df.tail(window_days).reset_index(drop=True)

            df.rename(columns={
                "ticker": "Ticker"
            }, inplace=True)

            next_business_days = pd.date_range(
                start=end_date,
                periods=forecast_horizon,
                freq=BDay()
            ).date.tolist()

            return df, next_business_days

        except Exception as e:
            logger.error(f"Error collecting data for {self.ticker} in {mode} mode: {e}")
            raise
