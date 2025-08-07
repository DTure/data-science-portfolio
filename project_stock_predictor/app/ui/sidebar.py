from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel, QWidget, QListWidgetItem
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSize, Signal

from ui.styles import AppStyles

class Sidebar(QFrame):
    bookmark_deleted = Signal(str)
    def __init__(self, expanded=True):
        super().__init__()
        self.expanded = expanded
        self._init_ui()
        self.set_expanded(self.expanded)
        self.setStyleSheet(AppStyles.SIDEBAR)
        self.setMinimumWidth(60)
        self.setMaximumWidth(250)

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Верхня частина
        self.top_widget = QWidget()
        self.top_layout = QVBoxLayout(self.top_widget)
        self.top_layout.setContentsMargins(5, 5, 5, 5)
        self.top_layout.setSpacing(5)

        self.title_bar = QWidget()
        self.title_bar_layout = QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar_layout.setSpacing(5)

        self.app_name = QLabel("Stock Predictor")
        self.app_name.setStyleSheet("color: white; font-size: 16px;")
        self.app_name.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.toggle_button = QPushButton()
        self.toggle_button.setIcon(QIcon("assets/menu_icon.png"))
        self.toggle_button.setIconSize(QSize(28, 28))
        self.toggle_button.setFlat(True)
        self.toggle_button.setStyleSheet("QPushButton { padding: 0; }")

        self.title_bar_layout.addWidget(self.app_name)
        self.title_bar_layout.addStretch()
        self.title_bar_layout.addWidget(self.toggle_button)

        self.top_layout.addWidget(self.title_bar)
        self.layout.addWidget(self.top_widget)

        # Кнопки навігації
        self.nav_buttons = QWidget()
        self.nav_layout = QVBoxLayout(self.nav_buttons)
        self.nav_layout.setContentsMargins(5, 10, 5, 10)
        self.nav_layout.setSpacing(5)

        self.home_button = QPushButton()
        self.home_button.setIcon(QIcon("assets/home_icon.png"))
        self.home_button.setIconSize(QSize(24, 24))
        self.home_button.setFixedHeight(40)
        self.home_button.full_text = "Home"

        self.collection_button = QPushButton()
        self.collection_button.setIcon(QIcon("assets/collection_icon.png"))
        self.collection_button.setIconSize(QSize(24, 24))
        self.collection_button.setFixedHeight(40)
        self.collection_button.full_text = "Collection"

        self.history_button = QPushButton()
        self.history_button.setIcon(QIcon("assets/history_icon.png"))
        self.history_button.setIconSize(QSize(24, 24))
        self.history_button.setFixedHeight(40)
        self.history_button.full_text = "History"

        self.nav_layout.addWidget(self.home_button)
        self.nav_layout.addWidget(self.collection_button)
        self.nav_layout.addWidget(self.history_button)
        self.layout.addWidget(self.nav_buttons)

        # Контейнер для закладок
        self.bookmark_container = QWidget()
        self.bookmark_layout = QVBoxLayout(self.bookmark_container)
        self.bookmark_layout.setContentsMargins(5, 5, 5, 5)
        self.bookmark_layout.setSpacing(5)

        self.bookmark_label = QLabel("Recent:")
        self.bookmark_label.setStyleSheet(AppStyles.SIDEBAR_BOOKMARK_LABEL)

        self.bookmark_list = QListWidget()
        self.bookmark_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.bookmark_layout.addWidget(self.bookmark_label)
        self.bookmark_layout.addWidget(self.bookmark_list)
        self.layout.addWidget(self.bookmark_container, 1)

    def set_expanded(self, expanded):
        self.expanded = expanded
        self.setFixedWidth(250 if expanded else 60)

        if expanded:
            self.app_name.show()
            self.bookmark_label.setVisible(True)
            self.bookmark_list.setVisible(True)
            self.title_bar_layout.setAlignment(Qt.AlignVCenter)
            self.toggle_button.setFixedSize(24, 24)
        else:
            self.app_name.hide()
            self.bookmark_label.setVisible(False)
            self.bookmark_list.setVisible(False)
            self.title_bar_layout.setAlignment(Qt.AlignCenter)
            self.toggle_button.setFixedSize(40, 40)

        for btn in [self.home_button, self.collection_button, self.history_button]:
            if expanded:
                btn.setText(btn.full_text)
                btn.setStyleSheet(AppStyles.SIDEBAR_BUTTON)
            else:
                btn.setText("")
                btn.setStyleSheet(AppStyles.SIDEBAR_BUTTON_ICON_ONLY)

    def update_bookmarks(self, mode, items):
        if self.expanded:
            if mode == "collection":
                self.bookmark_label.setText("Collection:")
            elif mode == "history":
                self.bookmark_label.setText("Recent:")
            else:
                self.bookmark_label.setText("Recent:")
          
            self.bookmark_list.clear()
            for item_text in items:
                item = QListWidgetItem(item_text)
                delete_button = QPushButton(QIcon("assets/delete_icon.png"), "")
                delete_button.setFlat(True)
                delete_button.setStyleSheet("QPushButton { padding: 0; }")
                delete_button.clicked.connect(
                    lambda checked, text=item_text, it=item: self.remove_bookmark_item(text, it)
                )

                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_widget.setStyleSheet(AppStyles.SIDEBAR_BOOKMARK_LABEL)
                item_layout.addWidget(QWidget(), 1)
                item_layout.addWidget(delete_button)
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(5)

                self.bookmark_list.addItem(item)
                self.bookmark_list.setItemWidget(item, item_widget)

    def remove_bookmark_item(self, item_text, item):
        self.bookmark_list.takeItem(self.bookmark_list.row(item))
        self.bookmark_deleted.emit(item_text)