# Stock Predictor
## Overview  
Stock Predictor is a desktop application for forecasting stock prices using neural network models based on LSTM and TCN architectures.
## Project Structure (Data Science related)
- `model_dev/` — full code for model development and initial data collection for training.

- `app/` — main application:  
  - `core/data_collector.py` — module responsible for collecting up-to-date stock data via Yahoo Finance API.
  - `model/short_term_model.py` — pretrained LSTM-based model for short-term stock price forecasting. 
  - `model/mid_term_model.py` — pretrained TCN-based model for mid-term stock price forecasting.
    
- `stock-predictor-demo.mp4` — video demonstration of the desktop application.
