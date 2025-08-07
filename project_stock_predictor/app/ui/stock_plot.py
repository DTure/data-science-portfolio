from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib import dates as mdates

class StockPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100) 
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  
        layout.setSpacing(0) 
        layout.addWidget(self.canvas, stretch=1)  
        
        plt.style.use('seaborn-v0_8')
        self.show_empty_state()
        
        self.cursor = None
        self.annotation = None
        self.original_xlim = None 
        self.original_ylim = None  
        self.is_zoomed = False     

    def show_empty_state(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Click "Predict" to generate forecast',
               ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        self.canvas.draw()

    def plot_data(self, historical_data, predicted_data):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        hist_dates = [mdates.date2num(d[0]) for d in historical_data]
        hist_prices = [d[1] for d in historical_data]
        pred_dates = [mdates.date2num(d[0]) for d in predicted_data]
        pred_prices = [d[1] for d in predicted_data]

        bridge_dates = [hist_dates[-1], pred_dates[0]]
        bridge_prices = [hist_prices[-1], pred_prices[0]]

        self.ax.plot(hist_dates, hist_prices, 'b-o', label='Historical Price', markersize=5, linewidth=1.5)
        self.ax.plot(bridge_dates, bridge_prices, 'b-', linewidth=1.5, alpha=0.5)
        self.ax.plot(pred_dates, pred_prices, 'r--o', label='Predicted Price', markersize=5, linewidth=1.5)

        date_format = mdates.DateFormatter('%d.%m')
        self.ax.xaxis.set_major_formatter(date_format)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')

        self.ax.set_title('Stock Price Prediction')

        self.ax.text(
            -0.06, -0.08,  
            '$ \ date',    
            transform=self.ax.transAxes,  
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

        self.ax.legend()
        self.ax.grid(True)
        self.figure.tight_layout()

        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8)
        )
        self.annotation.set_visible(False)

        self.highlight_dot, = self.ax.plot([], [], 'o', markersize=10, zorder=5, visible=False)
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 3 and self.is_zoomed:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.is_zoomed = False
            self.canvas.draw()
            return
            
        if event.button == 1 and not self.is_zoomed:
            for line in self.ax.get_lines():
                if line.get_label() == 'Predicted Price':
                    cont, ind = line.contains(event)
                    if cont:
                        point_idx = ind['ind'][0]
                        x, y = line.get_data()
                        x_point = x[point_idx]
                        
                        one_week = 7  
                        x_min = x_point - one_week/2
                        x_max = x_point + one_week/2
                        
                        mask = (x >= x_min) & (x <= x_max)
                        y_in_range = y[mask]
                        if len(y_in_range) > 0:
                            y_pad = (max(y_in_range) - min(y_in_range)) * 0.1  
                            y_min = min(y_in_range) - y_pad
                            y_max = max(y_in_range) + y_pad
                        else:
                            y_min, y_max = self.original_ylim
                            
                        self.ax.set_xlim(x_min, x_max)
                        self.ax.set_ylim(y_min, y_max)
                        self.is_zoomed = True
                        self.canvas.draw()
                        break

    def hover(self, event):
        if event.inaxes != self.ax:
            self.annotation.set_visible(False)
            self.highlight_dot.set_visible(False)
            self.canvas.draw_idle()
            return

        visible = False

        for line in self.ax.get_lines():
            if line == self.highlight_dot:
                continue
            cont, ind = line.contains(event)
            if cont:
                point_idx = ind['ind'][0]
                x, y = line.get_data()
                x_point = x[point_idx]
                y_point = y[point_idx]

                xdate = mdates.num2date(x_point)
                text = f"{xdate.strftime('%Y-%m-%d')}\nPrice: ${y_point:.2f}"

                self.annotation.xy = (x_point, y_point)
                self.annotation.set_text(text)
                self.annotation.set_visible(True)

                self.highlight_dot.set_data([x_point], [y_point])
                self.highlight_dot.set_color(line.get_color())
                self.highlight_dot.set_visible(True)

                visible = True
                break

        if not visible:
            self.annotation.set_visible(False)
            self.highlight_dot.set_visible(False)

        self.canvas.draw_idle()