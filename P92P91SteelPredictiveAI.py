import sys
import numpy as np
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# Load the saved model and scaler
with open('best_life_cycle_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Function to predict life cycle using the input parameters
def predict_life_cycle(temperature, strain_amplitude, tensile_hold_time, compressive_hold_time):
    input_data = np.array([[temperature, strain_amplitude, tensile_hold_time, compressive_hold_time]])
    input_data_scaled = scaler.transform(input_data)
    predicted_cycles = model.predict(input_data_scaled)
    return predicted_cycles[0]


# Convert cycles to time (assuming strain rate is N cycles per second)
def convert_cycles_to_time(cycles, strain_rate):
    time_seconds = cycles / strain_rate
    time_minutes = time_seconds / 60
    return time_seconds, time_minutes


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.step = 1
        self.input_data = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Predictive AI - Life Cycle Prediction")
        self.setGeometry(100, 100, 600, 400)

        # Background image
        self.setStyleSheet(
            "background-image: url(AppBackground.jpg); background-repeat: no-repeat; background-position: center;")

        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Title
        self.title = QLabel("Predictive AI for 9–12% Cr Steel", self)
        self.title.setFont(QFont("Benguiat Bold", 20))
        self.title.setStyleSheet("color: yellow;")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        # Label for questions
        self.label = QLabel("Enter Temperature (°C):", self)
        self.label.setFont(QFont("Helvetica", 12))
        self.label.setStyleSheet("color: white;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Entry box
        self.entry = QLineEdit(self)
        self.entry.setFont(QFont("Helvetica", 10))
        self.entry.setStyleSheet("color: white;")
        self.entry.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.entry)

        # Button to proceed to next input
        self.button = QPushButton("Next", self)
        self.button.setFont(QFont("Helvetica", 12))
        self.button.setStyleSheet("background-color: darkblue; color: white;")
        self.button.clicked.connect(self.next_step)
        layout.addWidget(self.button)

    def next_step(self):
        if self.step == 1:
            self.input_data['temperature'] = float(self.entry.text())
            self.label.setText("Enter Strain Amplitude (%):")
        elif self.step == 2:
            self.input_data['strain_amplitude'] = float(self.entry.text())
            self.label.setText("Enter Tensile Hold Time (minutes):")
        elif self.step == 3:
            self.input_data['tensile_hold_time'] = float(self.entry.text())
            self.label.setText("Enter Compressive Hold Time (minutes):")
        elif self.step == 4:
            self.input_data['compressive_hold_time'] = float(self.entry.text())
            result = predict_life_cycle(self.input_data['temperature'], self.input_data['strain_amplitude'],
                                        self.input_data['tensile_hold_time'], self.input_data['compressive_hold_time'])
            strain_rate = 0.5  # Assuming 0.5 cycles per second
            time_seconds, time_minutes = convert_cycles_to_time(result, strain_rate)
            self.label.setText(
                f"Predicted Life Cycle: {result:.2f} Cycles\nTime to Failure: {time_seconds:.2f} seconds ({time_minutes:.2f} minutes)")
            self.entry.setVisible(False)
            self.button.setVisible(False)
            return

        # Clear the entry box for next input
        self.entry.clear()
        self.step += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
