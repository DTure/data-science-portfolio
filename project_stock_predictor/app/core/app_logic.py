from datetime import datetime
from PySide6.QtCore import QObject, Signal
from typing import List, Dict, Tuple
from pathlib import Path
from core.data_collector import DataCollector
from core.model.short_term_model import LSTMForecaster
from core.model.mid_term_model import TCNForecaster
import json

class AppLogic(QObject):
    recommendations_updated = Signal(list)

    def __init__(self):
        super().__init__()
        self.company_data = self._load_sp500_data()  

    def _load_sp500_data(self) -> Dict[str, str]:
        data_path = Path(__file__).parent.parent / "data" / "sp500_companies.json"      
        with open(data_path, 'r', encoding='utf-8') as f:
            companies = json.load(f)
            return {item['symbol']: item['name'] for item in companies}
    
    def get_recommendations(self, search_text: str) -> List[str]:
        if not search_text:
            return []

        search_text = search_text.upper()
        exact_matches = []
        startswith_matches = []
        contains_matches = []

        for ticker, name in self.company_data.items():
            ticker_upper = ticker.upper()
            name_upper = name.upper()

            if ticker_upper == search_text:
                exact_matches.append(f"{ticker} - {name}")
            elif ticker_upper.startswith(search_text) or name_upper.startswith(search_text):
                startswith_matches.append(f"{ticker} - {name}")
            elif search_text in ticker_upper or search_text in name_upper:
                contains_matches.append(f"{ticker} - {name}")

        all_matches = exact_matches + startswith_matches + contains_matches
        return all_matches[:5]

    def search_company(self, search_text: str) -> Tuple[str, str]:
        search_text = search_text.upper()
        for ticker, name in self.company_data.items():
            if search_text == ticker:
                return (ticker, name)
        for ticker, name in self.company_data.items():
            if search_text == name.upper():
                return (ticker, name)
        return (None, None)
    
    def exact_match(self, search_text: str) -> Tuple[str, str]:
        search_text = search_text.upper()
        for ticker, name in self.company_data.items():
            if ticker == search_text or name.upper() == search_text:
                return ticker, name
        return None, None
    
    def run_forecast(self, ticker: str, mode: str) -> Tuple[List[Tuple[datetime.date, float]], List[Tuple[datetime.date, float]]]:
        print(f"Received forecast request for ticker: {ticker}, mode: {mode}")
        try:
            collector = DataCollector(ticker)
            df, future_dates = collector.collect(mode)
            print(f"Data for {ticker} successfully collected. Last rows:")
            print(df.tail(5))
            print("Number of rows:", len(df))
            print("Future dates:", future_dates)

            if mode == "Short-term":
                model = LSTMForecaster()
                predicted_values = model.predict(df)
                
                historical_data = [
                    (row['date'].date(), row['close'])
                    for _, row in df.tail(14).iterrows()
                ]
                
                forecast_data = [
                    (future_dates[i], float(predicted_values[i]))
                    for i in range(min(len(predicted_values), len(future_dates)))
                ]
                
                print(f"📈 Forecast for {len(forecast_data)} days:")
                for date, value in forecast_data:
                    print(f"{date}: {value:.2f}")

                return historical_data, forecast_data

            elif mode == "Medium-term":
                model = TCNForecaster()
                predicted_values = model.predict(df)
                
                historical_data = [
                    (row['date'].date(), row['close'])
                    for _, row in df.tail(60).iterrows()
                ]
                
                forecast_data = [
                    (future_dates[i], float(predicted_values[i]))
                    for i in range(min(len(predicted_values), len(future_dates)))
                ]
                
                print(f"📈 Forecast for {len(forecast_data)} days:")
                for date, value in forecast_data:
                    print(f"{date}: {value:.2f}")

                return historical_data, forecast_data

            else:
                print(f"Model for mode '{mode}' is not implemented.")
                return [], []

        except Exception as e:
            print(f"Error while processing forecast for {ticker}: {e}")
            return [], []
