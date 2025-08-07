from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QStackedWidget, QLabel,
QSpacerItem, QSizePolicy, QPushButton, QCompleter, QListWidgetItem)
from PySide6.QtCore import Qt, QSize, QStringListModel, QTimer
from PySide6.QtGui import QIcon
from ui.sidebar import Sidebar
from ui.company_page import CompanyPage
from ui.styles import AppStyles
from core.app_logic import AppLogic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("assets/app_icon.png"))
        if hasattr(self, 'setWindowIcon'):
            self.setWindowIcon(QIcon("assets/app_icon.png"))
        self.setWindowTitle("Stock Prediction App")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(AppStyles.MAIN_WINDOW_STYLE)

        self.app_logic = AppLogic()
        CompanyPage.app_logic = self.app_logic
        self.app_logic.recommendations_updated.connect(self.update_recommendations)
        
        # Головний контейнер
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основний layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Бокова панель
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Контентна область
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(10)
        
        # Пошук
        self.search_panel = QWidget()
        search_layout = QHBoxLayout(self.search_panel)
        search_layout.setContentsMargins(0, 0, 0, 0)

        # Розпірка зліва
        left_spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        search_layout.addItem(left_spacer)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search for companies...")
        self.search_field.setStyleSheet(AppStyles.SEARCH)
        search_layout.addWidget(self.search_field)

        self.search_button = QPushButton()
        self.search_button.setIcon(QIcon("assets/search_icon.png"))
        self.search_button.setStyleSheet(AppStyles.SEARCH_BUTTON)  
        self.search_button.setIconSize(QSize(20, 20))
        self.search_button.setCursor(Qt.PointingHandCursor)  
        search_layout.addWidget(self.search_button)

        # Розпірка справа
        right_spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        search_layout.addItem(right_spacer)

        # Задаємо stretch factor
        search_layout.setStretch(0, 1)  
        search_layout.setStretch(1, 2)
        search_layout.setStretch(2, 0)  
        search_layout.setStretch(3, 1) 

        content_layout.addWidget(self.search_panel)
        self.completer = QCompleter(self) 
        self.completer.popup().setStyleSheet(AppStyles.COMPLETER)
        self.search_field.setCompleter(self.completer)
         
        
        # Стек сторінок
        self.stacked_widget = QStackedWidget()
        
        # Домашня сторінка
        self.home_page = QWidget()
        home_layout = QVBoxLayout(self.home_page)
        self.home_label = QLabel("Welcome to Stock Prediction App")
        self.home_label.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(self.home_label)
        
        self.stacked_widget.addWidget(self.home_page)
        content_layout.addWidget(self.stacked_widget)
        
        main_layout.addWidget(content_area, 1) 
        
        # Словник сторінок
        self.pages = {"home": self.home_page}
        
        # Підключення сигналів
        self._connect_signals()
        
        # Встановлюємо початковий стан
        self.sidebar.home_button.setText("Home")
        self.sidebar.collection_button.setText("Collection")
        self.sidebar.history_button.setText("History")
        self.sidebar.home_button.setProperty("text", "Home")
        self.sidebar.collection_button.setProperty("text", "Collection")
        self.sidebar.history_button.setProperty("text", "History")
    
    def _connect_signals(self):
        self.sidebar.toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar.home_button.clicked.connect(self.show_home_page)
        self.sidebar.collection_button.clicked.connect(self.show_collection)
        self.sidebar.history_button.clicked.connect(self.show_history)
        self.search_field.textChanged.connect(self.show_recommendations)
        self.sidebar.bookmark_list.itemClicked.connect(self.open_bookmark_page)
        self.sidebar.bookmark_deleted.connect(self.remove_company_page)
        self.completer.highlighted.connect(self.on_completer_selected)
        self.search_button.clicked.connect(self.on_search_button_clicked)

    
    def on_completer_selected(self, text):
        self.search_field.textChanged.disconnect(self.show_recommendations)
        self.search_field.setText(text)
        self.search_field.textChanged.connect(self.show_recommendations)
        self.open_company_page(QListWidgetItem(text))
  
    def toggle_sidebar(self):
        current_width = self.sidebar.width()
        if current_width > 100:  
            self.sidebar.set_expanded(False)
        else:  
            self.sidebar.set_expanded(True)

# LOGIC METHODS FOR THE INTERFACE

    def open_company_page(self, item):
        QTimer.singleShot(0, lambda: self.search_field.setText(""))
        text = item.text() 
        parts = text.split(" - ")
        ticker = parts[0] 

        ticker, full_name = self.app_logic.search_company(ticker)
        if not ticker:  
            return

        display_name = f"{ticker} - {full_name}" 

        if ticker not in self.pages:
            new_page = CompanyPage(full_name, ticker)
            new_page.company_name = full_name 
            self.stacked_widget.addWidget(new_page)
            self.pages[ticker] = new_page  

            current_items = [self.sidebar.bookmark_list.item(i).text() for i in range(self.sidebar.bookmark_list.count())]
            if display_name not in current_items:
                current_items.append(display_name)
                self.sidebar.update_bookmarks("bookmarks", current_items)
        else:
            if hasattr(self.pages[ticker], 'is_saved'):
                if self.pages[ticker].is_saved:
                    self.pages[ticker].save_button.setIcon(QIcon("assets/collection_saved_icon.png"))
                else:
                    self.pages[ticker].save_button.setIcon(QIcon("assets/collection_icon.png"))

        self.stacked_widget.setCurrentWidget(self.pages[ticker])
    
    def show_recommendations(self, text):
        recommendations = self.app_logic.get_recommendations(text)
        model = QStringListModel(recommendations)
        self.completer.setModel(model)
        self.completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.complete(self.search_field.rect())
    
    def update_recommendations(self, recommendations):
        model = QStringListModel(recommendations)
        self.completer.setModel(model)
          
    def open_bookmark_page(self, item):
        text = item.text()  
        parts = text.split(" - ")
        ticker = parts[0] 

        if ticker in self.pages:
            self.stacked_widget.setCurrentWidget(self.pages[ticker])
         
    def show_home_page(self):
        self.stacked_widget.setCurrentWidget(self.pages["home"])
    
    def show_history(self):
        open_pages = []
        for name, page in self.pages.items():
            if name != "home":
                full_name = getattr(page, "company_name", "")
                display_name = f"{name} - {full_name}"
                open_pages.append(display_name)
        self.sidebar.update_bookmarks("history", open_pages)

    def show_collection(self):
        saved_items = []
        for ticker, page in self.pages.items():
            if hasattr(page, 'is_saved') and page.is_saved:
                full_name = getattr(page, "company_name", "")
                display_name = f"{ticker} - {full_name}"
                saved_items.append(display_name)
        self.sidebar.update_bookmarks("collection", saved_items)
    
    def remove_company_page(self, company_name):
        parts = company_name.split(" - ")
        ticker = parts[0] 

        if ticker in self.pages:
            page = self.pages.pop(ticker)
            self.stacked_widget.removeWidget(page)
            page.deleteLater()
    
    def on_search_button_clicked(self):
        text = self.search_field.text().strip()
        if not text:
            return
        ticker, name = self.app_logic.exact_match(text)
        if ticker and name:
            display_text = f"{ticker} - {name}"
            item = QListWidgetItem(display_text)
            self.open_company_page(item)



    