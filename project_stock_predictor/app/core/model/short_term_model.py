import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os

class LSTMShortTermModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=7, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

class LSTMForecaster:
    def __init__(self, artifact_dir="core/model/data/"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.artifact_dir = artifact_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = ["open", "high", "low", "close", "volume", "SMA_5", "RSI_7", "BB_high", "BB_low"]

        self._load_artifacts()

    def _load_artifacts(self):
        self.scaler = joblib.load(os.path.join(self.artifact_dir, "scaler_lstm.pkl"))
        self.label_encoder = joblib.load(os.path.join(self.artifact_dir, "label_encoder_l.pkl"))

        input_size = len(self.features) + 1  
        self.model = LSTMShortTermModel(input_size=input_size, output_size=7, device=self.device)

        weights_path = os.path.join(self.artifact_dir, "lstm_weights.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Ваги моделі не знайдено за шляхом: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def _inverse_transform_close(self, normalized_values: np.ndarray) -> np.ndarray:
        dummy_data = np.zeros((len(normalized_values), len(self.features)))
        close_index = self.features.index("close")
        dummy_data[:, close_index] = normalized_values
        real_values = self.scaler.inverse_transform(dummy_data)[:, close_index]
        return real_values

    def predict(self, df_30_rows: pd.DataFrame) -> np.ndarray:
        if df_30_rows.shape[0] != 30:
            raise ValueError("Вхідні дані повинні містити рівно 30 днів історії")
        
        ticker = df_30_rows["Ticker"].iloc[0]
        ticker_encoded = self.label_encoder.transform([ticker])[0]
        df = df_30_rows.copy()
        df[self.features] = self.scaler.transform(df[self.features])
        df["Ticker"] = ticker_encoded
        values = df[self.features + ["Ticker"]].values
        input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy().flatten()

        real_prediction = self._inverse_transform_close(prediction)
        return real_prediction
