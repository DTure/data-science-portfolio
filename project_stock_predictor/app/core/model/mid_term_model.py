import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Union

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if out.size(-1) != x.size(-1):
            out = out[:, :, -x.size(-1):]

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels=[64, 64], kernel_size=3, dropout=0.2, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation,
                                        padding=(kernel_size - 1) * dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.to(self.device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

class TCNForecaster:
    def __init__(self, artifacts_dir: str = "core/model/data/", window_size: int = 90, pred_horizon: int = 30):
        self.artifacts_dir = artifacts_dir
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.features = [
            "open", "high", "low", "close", "volume",
            "SMA_20", "SMA_50", "RSI_14", "BB_high", "BB_low",
            "MACD", "Momentum", "ATR", "Ticker"
        ]
        self.close_idx = self.features.index("close")

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        self.scaler = joblib.load(f"{self.artifacts_dir}/scaler_tcn.pkl")
        self.label_encoder = joblib.load(f"{self.artifacts_dir}/label_encoder_t.pkl")

        input_size = len(self.features)
        self.model = TCN(input_size=input_size, output_size=self.pred_horizon, device=self.device)
        self.model.load_state_dict(torch.load(f"{self.artifacts_dir}/tcn_weights.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _inverse_transform_close(self, values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        mean = self.scaler.mean_[self.close_idx]
        std = np.sqrt(self.scaler.var_[self.close_idx])
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        return values * std + mean

    def predict(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        ticker = df["Ticker"].iloc[0]
        
        if ticker not in self.label_encoder.classes_:
            print(f"Ticker '{ticker}' is not present in the training set.")
            return None

        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < self.window_size:
            print(f"Not enough data for ticker '{ticker}' (required: {self.window_size}, available: {len(df)})")
            return None

        df_features = df[self.features[:-1]].copy()  
        df_features_scaled = self.scaler.transform(df_features)
        
        ticker_encoded = self.label_encoder.transform([ticker])[0]
        df_scaled = np.concatenate([
            df_features_scaled, 
            np.full((len(df), 1), ticker_encoded)
        ], axis=1)

        x_input = df_scaled[-self.window_size:]
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(x_tensor)
            prediction = prediction.squeeze(0).cpu().numpy()

        return self._inverse_transform_close(prediction)
