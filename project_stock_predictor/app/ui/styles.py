class AppStyles:

    MAIN_BACKGROUND = "#dcddde"
    DARK_BG = "#202123"
    LIGHT_BG = "#f8f9fa"
    WHITE = "white"
    ACCENT_GREEN = "#4CAF50"
    ACCENT_GRAY = "#202123"  
    HOVER_GREEN = "#45a049"
    HOVER_GRAY = "#343541"  

    MAIN_WINDOW_STYLE = f"""
        QMainWindow {{
            background-color: {MAIN_BACKGROUND};
        }}
        QWidget#centralWidget {{
            background-color: {MAIN_BACKGROUND};
        }}
    """

    SIDEBAR = f"""
        QFrame {{
            background-color: {DARK_BG};
        }}
        QPushButton {{
            text-align: left;
            padding: 8px;
            color: {WHITE};
            border: none;
        }}
        QListWidget {{
            background-color: {DARK_BG};
            color: {WHITE};
            border: none;
            outline: none;
        }}
        QListWidget::item {{
            padding: 8px;
        }}
        QListWidget::item:hover {{
            background-color: #343541;
        }}
    """

    SEARCH = """
        QLineEdit {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 12px;
            font-size: 14px;
        }
        QListWidget {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 4px;
        }
        QListWidget::item {
            padding: 8px;
        }
        QListWidget::item:hover {
            background-color: #f0f0f0;
        }
    """

    SEARCH_BUTTON = f"""
        QPushButton {{
            background-color: {ACCENT_GRAY};
            color: {WHITE};
            border: none;
            padding: 8px;
            border-radius: 16px;
            min-width: 0;
            min-height: 0;
            margin: 0;
        }}
        QPushButton:hover {{
            background-color: {HOVER_GRAY};
        }}
    """

    BUTTON_PRIMARY = f"""
        QPushButton {{
            background-color: {ACCENT_GREEN};
            color: {WHITE};
            border: none;
            min-height: 26px;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {HOVER_GREEN};
        }}
    """

    BUTTON_SECONDARY = f"""
        QPushButton {{
            background-color: {ACCENT_GRAY};
            color: {WHITE};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {HOVER_GRAY};
        }}
    """

    COMPANY_HEADER = f"""
        QFrame {{
            background-color: {LIGHT_BG};
            padding: 5px;
            border-radius: 20px;
        }}
        QLabel {{
            font-size: 16px;
            font-weight: bold;
        }}
    """

    FORECAST_COMBOBOX = f"""
    QComboBox {{
        font-size: 14px;
        padding: 8px 12px;  
        min-height: 12px;   
        border: 1px solid #343541;
        border-radius: 4px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 24px;
    }}
    """

    SIDEBAR_BUTTON = """
        QPushButton {
            color: white;
            text-align: left;
            padding-left: 10px;
            border: none;
        }
        QPushButton:hover {
            background-color: #44475a;
        }
        """

    SIDEBAR_BUTTON_ICON_ONLY = """
        QPushButton {
            color: white;
            text-align: center;
            border: none;
        }
        QPushButton:hover {
            background-color: #44475a;
        }
        """
    
    SIDEBAR_BOOKMARK_LABEL = """
        QLabel {
            color: #CCCCCC;
            padding-left: 10px;
            padding-top: 5px;
            font-size: 12px;
            
        }     
        """
    
    COMPLETER = """
    QListView {
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: white;
        padding: 6px;
        font-size: 14px;  
        min-width: 300px; 
    }
    QListView::item {
        padding: 8px 12px;  
        height: 28px;       
    }
    QListView::item:hover {
        background-color: #f0f0f0;
    }
    QScrollBar:vertical {
        width: 12px;        
    }
    """