import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
from typing import Dict

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
        num_levels = len(num_channels)
        for i in range(num_levels):
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

class TCNTrainer:
    def __init__(self, csv_path, window_size=90, pred_horizon=30):
        self.csv_path = csv_path
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.ticker_code_to_name: Dict[int, str] = {}

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, parse_dates=["Date"])
        df.sort_values(["Ticker", "Date"], inplace=True)
        return df

    def preprocess_data(self, df: pd.DataFrame):

        df["Ticker"] = self.label_encoder.fit_transform(df["Ticker"])
        self.ticker_code_to_name = {i: name for i, name in enumerate(self.label_encoder.classes_)}

        features = [
            "open", "high", "low", "close", "volume",
            "SMA_20", "SMA_50", "RSI_14", "BB_high", "BB_low", "MACD", "Momentum", "ATR", "Ticker"
        ]
        df[features[:-1]] = self.scaler.fit_transform(df[features[:-1]])

        X_train, y_train, X_test, y_test = [], [], [], []

        grouped = df.groupby("Ticker")
        for _, group in grouped:
            group = group.reset_index(drop=True)
            values = group[features].values
            closes = group["close"].values

            total_length = len(group)
            min_required = self.window_size + self.pred_horizon
            if total_length < min_required + 1:
                continue

            for i in range(total_length - min_required - 1):
                x = values[i:i + self.window_size]
                y = closes[i + self.window_size:i + self.window_size + self.pred_horizon]
                X_train.append(x)
                y_train.append(y)

            '''
            i = total_length - self.window_size - self.pred_horizon
            x_test = values[i:i + self.window_size]
            y_test_sample = closes[i + self.window_size:i + self.window_size + self.pred_horizon]
            X_test.append(x_test)
            y_test.append(y_test_sample)
            ''' 
            '''torch.tensor(np.array(X_test), dtype=torch.float32),
            torch.tensor(np.array(y_test), dtype=torch.float32)'''

        return (
            torch.tensor(np.array(X_train), dtype=torch.float32),
            torch.tensor(np.array(y_train), dtype=torch.float32),
            
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32)
        )

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 30, batch_size: int = 32, lr: float = 0.001, n_splits: int = 5, patience: int = 5):
        if self.model is None:
            input_size = X_train.shape[2]
            self.model = TCN(input_size=input_size, output_size=self.pred_horizon, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        weights = torch.linspace(2, 1, steps=self.pred_horizon).to(self.device)
        def weighted_mse(pred, target):
            return torch.mean(weights * (pred - target) ** 2)

        best_val_loss = float('inf')
        patience_counter = 0

        ticker_ids = X_train[:, 0, -1].unique()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = weighted_mse(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Walk-forward validation
            val_losses = []
            for split in range(n_splits):
                val_tickers = ticker_ids[split::n_splits]
                val_mask = torch.isin(X_train[:, 0, -1], val_tickers)
                X_val, y_val = X_train[val_mask], y_train[val_mask]

                if len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val.to(self.device))
                        val_loss = weighted_mse(val_pred, y_val.to(self.device)).item()
                        val_losses.append(val_loss)

            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_weights = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"Epoch {epoch+1}/{epochs} ➤ Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        self.model.load_state_dict(best_weights)
        print(f"Best validation MSE: {best_val_loss:.4f}")


    def evaluate_by_ticker(self, X_test: torch.Tensor, y_test: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            ticker_ids = X_test[:, 0, -1].cpu().numpy()
            unique_tickers = np.unique(ticker_ids)

            results = []
            close_idx = 3

            for ticker_id in unique_tickers:
                mask = ticker_ids == ticker_id
                X_ticker = X_test[mask].to(self.device)
                y_ticker = y_test[mask].cpu().numpy()

                if len(X_ticker) == 0:
                    continue

                preds = self.model(X_ticker).cpu().numpy()

                # Denormalization
                mean = self.scaler.mean_[close_idx]
                std = np.sqrt(self.scaler.var_[close_idx])
                y_denorm = y_ticker * std + mean
                preds_denorm = preds * std + mean

                mae = np.mean(np.abs(y_denorm - preds_denorm))
                mape = np.mean(np.abs((y_denorm - preds_denorm) / (y_denorm + 1e-8))) * 100
                acc = 100 - mape
                ticker_name = self.ticker_code_to_name.get(int(ticker_id), "Unknown")

                actual_last = float(y_denorm[0, -1])
                predicted_last = float(preds_denorm[0, -1])

                results.append({
                    "Ticker": ticker_name,
                    "MAE": mae,
                    "MAPE": mape,
                    "Accuracy": acc,
                    "Actual_Last": actual_last,
                    "Predicted_Last": predicted_last
                })

        df_results = pd.DataFrame(results)
        print(df_results.sort_values("Accuracy", ascending=False).to_string(index=False))
        
        print("\n🔹 Overall Statistics:")
        print(f"Average accuracy:      {df_results['Accuracy'].mean():.2f}%")
        print(f"Highest accuracy:      {df_results['Accuracy'].max():.2f}%")
        print(f"Lowest accuracy:       {df_results['Accuracy'].min():.2f}%")
        print(f"Average MAE:           {df_results['MAE'].mean():.4f}")
        print(f"Average MAPE:          {df_results['MAPE'].mean():.2f}%")
        
        return df_results

    def save_artifacts(self, dir_path: str = "tcn_model/"):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{dir_path}/tcn_weights.pth")
        joblib.dump(self.scaler, f"{dir_path}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{dir_path}/label_encoder.pkl")
        print(f"Artifacts saved to {dir_path}")

    def load_model(self, dir_path: str = "tcn_model/"):
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist")

        if self.model is None:
            # Need to initialize model first with correct input size
            raise ValueError("Model has not been initialized. Call preprocess_data() first or initialize model manually.")
            
        self.model.load_state_dict(torch.load(f"{dir_path}/tcn_weights.pth", map_location=self.device))
        self.scaler = joblib.load(f"{dir_path}/scaler.pkl")
        self.label_encoder = joblib.load(f"{dir_path}/label_encoder.pkl")
        self.ticker_code_to_name = {i: name for i, name in enumerate(self.label_encoder.classes_)}
        print(f"Model loaded from {dir_path}")

    def training_pipeline(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001, patience: int = 5):
        print("1. Loading data...")
        df = self.load_data()

        print("2. Preprocessing...")
        X_train, y_train, X_test, y_test = self.preprocess_data(df)

        print(f"3. Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"   Test shape: X={X_test.shape}, y={y_test.shape}")

        print("4. Training with cross-validation...")
        self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)

        print("5. Saving artifacts...")
        self.save_artifacts()

    def testing_pipeline(self, model_path: str = "tcn_model/"):
        print("1. Loading data...")
        df = self.load_data()

        print("2. Preprocessing data...")
        _, _, X_test, y_test = self.preprocess_data(df)
        print(f"   ➤ X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        print("3. Initializing model...")
        input_size = X_test.shape[2]
        self.model = TCN(input_size=input_size, output_size=self.pred_horizon, device=self.device)

        print("4. Loading trained weights and scalers...")
        self.load_model(dir_path=model_path)

        print("5. Evaluating model per ticker...")
        self.evaluate_by_ticker(X_test, y_test)

if __name__ == "__main__":
    trainer = TCNTrainer(csv_path="data/mid_term.csv")
    trainer.training_pipeline(epochs=50, batch_size=32, lr=0.001, patience=7)
    #trainer.testing_pipeline(model_path="tcn_model/")