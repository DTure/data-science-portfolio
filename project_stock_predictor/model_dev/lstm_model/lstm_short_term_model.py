import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

class LSTMTrainer:
    def __init__(self, csv_path, window_size=30, pred_horizon=7):
        self.csv_path = csv_path
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.csv_path, parse_dates=["Date"])
        df.sort_values(["Ticker", "Date"], inplace=True)
        return df

    def preprocess_data(self, df, test_examples=0):
        df["Ticker"] = self.label_encoder.fit_transform(df["Ticker"])
        self.ticker_code_to_name = {i: name for i, name in enumerate(self.label_encoder.classes_)}

        features = ["open", "high", "low", "close", "volume", "SMA_5", "RSI_7", "BB_high", "BB_low"]
        df[features] = self.scaler.fit_transform(df[features])

        X_train, y_train = [], []
        X_test, y_test = [], []

        grouped = df.groupby("Ticker")
        for _, group in grouped:
            group = group.reset_index(drop=True)
            values = group[features + ["Ticker"]].values
            closes = group["close"].values

            total_length = len(group)
            min_required = self.window_size + self.pred_horizon + test_examples - 1
            if total_length < min_required:
                continue

            # --- Training Windows ---
            for i in range(total_length - self.window_size - self.pred_horizon - test_examples):
                x = values[i:i + self.window_size]
                y = closes[i + self.window_size:i + self.window_size + self.pred_horizon]
                X_train.append(x)
                y_train.append(y)

            # --- Multiple test examples (shifted by 1 each) ---
            '''for shift in range(test_examples):
                i = total_length - self.window_size - self.pred_horizon - (test_examples - 1 - shift)
                x_test = values[i:i + self.window_size]
                y_test_sample = closes[i + self.window_size:i + self.window_size + self.pred_horizon]
                X_test.append(x_test)
                y_test.append(y_test_sample)'''

        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
        X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
        y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

        return X_train, y_train, X_test, y_test


    def train(self, X_train, y_train, epochs=10, batch_size=64, lr=0.001, n_splits=5):
        if self.model is None:
            input_size = X_train.shape[2]
            self.model = LSTMShortTermModel(input_size=input_size, output_size=self.pred_horizon, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')

        # Ticker-based cross-validation
        ticker_ids = X_train[:, 0, -1].unique()
        split_size = len(ticker_ids) // n_splits

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # --- Validation (Walk-Forward) ---
            val_losses = []
            for split in range(n_splits):
                val_tickers = ticker_ids[split::n_splits]
                val_mask = torch.isin(X_train[:, 0, -1], val_tickers)
                X_val, y_val = X_train[val_mask], y_train[val_mask]

                if len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val.to(self.device))
                        val_loss = criterion(val_pred, y_val.to(self.device)).item()
                        val_losses.append(val_loss)

            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')

            print(f"Epoch {epoch+1}/{epochs} ➤ "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = self.model.state_dict()

        self.model.load_state_dict(best_weights)
        print(f"\nBest validation MSE: {best_val_loss:.4f}")

    def load_model(self, path):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def evaluate_by_ticker(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            ticker_ids = X_test[:, 0, -1].cpu().numpy()
            unique_tickers = np.unique(ticker_ids)

            results = []

            for ticker_id in unique_tickers:
                mask = ticker_ids == ticker_id
                X_ticker = X_test[mask].to(self.device)
                y_ticker = y_test[mask].cpu().numpy()

                if len(X_ticker) == 0:
                    continue

                preds = self.model(X_ticker).cpu().numpy()

                close_idx = 3
                mean = self.scaler.mean_[close_idx]
                std = np.sqrt(self.scaler.var_[close_idx])

                y_ticker_denorm = y_ticker * std + mean
                preds_denorm = preds * std + mean

                mae = np.mean(np.abs(y_ticker_denorm - preds_denorm))
                mape = np.mean(np.abs((y_ticker_denorm - preds_denorm) / (y_ticker_denorm + 1e-8))) * 100
                acc = 100 - mape
                ticker_name = self.ticker_code_to_name.get(int(ticker_id), "Unknown")

                actual_last = float(np.mean(y_ticker_denorm[:, -1]))  # середнє з останніх днів
                predicted_last = float(np.mean(preds_denorm[:, -1]))

                results.append({
                    "ticker": int(ticker_id),
                    "ticker_name": ticker_name,
                    "mae": mae,
                    "mape": mape,
                    "accuracy": acc,
                    "actual_last": actual_last,
                    "predicted_last": predicted_last
                })

        df_results = pd.DataFrame(results)
        print(df_results.sort_values("accuracy", ascending=False).to_string(index=False))

        print("\n🔹 Overall Statistics:")
        print(f"Average accuracy:      {df_results['accuracy'].mean():.2f}%")
        print(f"Highest accuracy:      {df_results['accuracy'].max():.2f}%")
        print(f"Lowest accuracy:       {df_results['accuracy'].min():.2f}%")
        print(f"Average MAE:           {df_results['mae'].mean():.4f}")
        print(f"Average MAPE:          {df_results['mape'].mean():.2f}%")

        return df_results

    def save_artifacts(self, dir_path="lstm_model/"):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{dir_path}/lstm_weights.pth")
        import joblib
        joblib.dump(self.scaler, f"{dir_path}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{dir_path}/label_encoder.pkl")
        print(f"Artifacts saved to {dir_path}")

    def training_pipeline(self, epochs=10, batch_size=64, lr=0.001):
        print("1. Loading CSV")
        df = self.load_data()

        print("2. Preprocessing data")
        X_train, y_train, X_test, y_test = self.preprocess_data(df,test_examples=5)

        print(f"3. Train shape: X={X_train.shape}, y={y_train.shape}")

        print("4. Starting model training")
        self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)

        print("5. Saving model and artifacts")
        self.save_artifacts()

    def testing_pipeline(self, model_path="lstm_model/lstm_weights.pth"):
        print("1. Loading CSV")
        df = self.load_data()

        print("2. Preprocessing data (only test set)")
        _, _, X_test, y_test = self.preprocess_data(df,test_examples=5)
        print(f"   ➤ X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

        print("3. Building model")
        input_size = X_test.shape[2]
        self.model = LSTMShortTermModel(input_size=input_size, output_size=self.pred_horizon, device=self.device)

        print("4. Loading trained weights")
        self.load_model(path=model_path)

        print("5. Evaluating model per ticker")
        self.evaluate_by_ticker(X_test, y_test)

if __name__ == "__main__":
    trainer = LSTMTrainer(csv_path="data/short_term.csv")
    trainer.training_pipeline(epochs=30, batch_size=32, lr=0.0005)
    #trainer.testing_pipeline()
