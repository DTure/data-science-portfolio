from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                              QPushButton, QFrame, QSpacerItem, QSizePolicy)
from PySide6.QtGui import QFont, QIcon, QPainter, QColor
from PySide6.QtCore import Qt, QSize
from ui.styles import AppStyles
from ui.stock_plot import StockPlot
from datetime import datetime, timedelta
from core.app_logic import AppLogic
import numpy as np

class CompanyPage(QWidget):
    app_logic = None
    def __init__(self, company_name, company_ticker):
        super().__init__()
        self.app_logic = AppLogic()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Верхня панель
        self.info_panel = QFrame()
        self.info_panel.setStyleSheet(AppStyles.COMPANY_HEADER)
        self.info_layout = QHBoxLayout(self.info_panel)
        self.info_layout.setContentsMargins(20, 15, 20, 15)
        self.info_layout.setSpacing(15)

        # Інформація про компанію (ліва частина)
        self.company_info = QWidget()
        self.company_layout = QHBoxLayout(self.company_info)
        self.company_layout.setContentsMargins(0, 0, 0, 0)
        self.company_layout.setSpacing(10)

        self.company_name = QLabel(company_name)
        self.company_name.setFont(QFont("Arial", 16, QFont.Bold))

        self.ticker_label = QLabel("Ticker:")
        self.ticker_label.setFont(QFont("Arial", 12))
        
        self.company_ticker = QLabel(company_ticker)
        self.company_ticker.setFont(QFont("Arial", 12, QFont.Bold))

        self.company_layout.addWidget(self.company_name)
        self.company_layout.addWidget(self.ticker_label)
        self.company_layout.addWidget(self.company_ticker)
        self.company_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Налаштування прогнозу (права частина)
        self.settings_panel = QWidget()
        self.settings_layout = QHBoxLayout(self.settings_panel)
        self.settings_layout.setSpacing(10)

        # Блок з терміном прогнозу
        self.term_block = QWidget()
        self.term_layout = QHBoxLayout(self.term_block)
        self.term_layout.setContentsMargins(0, 0, 0, 0)
        self.term_layout.setSpacing(5)

        self.term_label = QLabel("Forecast mode:")
        self.term_label.setFont(QFont("Arial", 10))
        self.term_label.setContentsMargins(0, 0, 0, 0)
        
        self.forecast_term = QComboBox()
        self.forecast_term.addItems(["Short-term", "Medium-term"])
        self.forecast_term.setFont(QFont("Arial", 12))
        self.forecast_term.setStyleSheet(AppStyles.FORECAST_COMBOBOX)
        
        self.term_layout.addWidget(self.term_label)
        self.term_layout.addWidget(self.forecast_term)

        # Кнопки
        self.predict_button = QPushButton("Predict")
        self.predict_button.setStyleSheet(AppStyles.BUTTON_PRIMARY)
        self.predict_button.setFixedWidth(120)

        self.save_button = QPushButton()
        self.save_button.setIcon(QIcon("assets/collection_icon.png"))
        self.save_button.setIconSize(QSize(24, 24)) 
        self.save_button.setStyleSheet(AppStyles.BUTTON_SECONDARY)
        self.save_button.setFixedSize(40, 40)

        self.is_saved = False
        self.save_button.clicked.connect(self.toggle_save_state)

        self.settings_layout.addWidget(self.term_block)
        self.settings_layout.addWidget(self.predict_button)
        self.settings_layout.addWidget(self.save_button)

        # Додаємо обидві частини до головного layout
        self.info_layout.addWidget(self.company_info)
        self.info_layout.addWidget(self.settings_panel)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(20, 20, 20, 10)
        self.results_layout.setSpacing(5)
        
        self.empty_label = QLabel("Prediction results will appear here")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("background-color: white; border: 1px solid #ddd; padding: 20px; font-size: 16px;")
        
        self.stock_plot = StockPlot()
        self.stock_plot.setVisible(False)
        
        self.results_layout.addWidget(self.empty_label)
        self.results_layout.addWidget(self.stock_plot)
        
        self.layout.addWidget(self.info_panel)
        self.layout.addWidget(self.results_container, stretch=1)
        
        self.predict_button.clicked.connect(self.on_predict_clicked)

    def toggle_save_state(self):
        self.is_saved = not self.is_saved
        if self.is_saved:
            self.save_button.setIcon(QIcon("assets/collection_saved_icon.png"))
        else:
            self.save_button.setIcon(QIcon("assets/collection_icon.png"))

    def on_predict_clicked(self):
        ticker = self.company_ticker.text()
        mode = self.forecast_term.currentText()
        
        # Отримуємо дані з app_logic
        historical, predicted = CompanyPage.app_logic.run_forecast(ticker, mode)
        
        if not historical or not predicted:
            print("Не вдалося отримати дані для графіка")
            return
        
        # Оновлюємо графік
        self.stock_plot.plot_data(historical, predicted)
        self.empty_label.setVisible(False)
        self.stock_plot.setVisible(True)
