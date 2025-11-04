# chronix_v4_pro_fixed.py — cleaned, fixed, working
import sys
import os
import time
import random
from collections import deque

import psutil
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QPushButton, QFileDialog, QMessageBox,
    QTabWidget, QFormLayout, QLineEdit, QCheckBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer

# Matplotlib canvas for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Configuration
# ---------------------------
DEBUG = False  # set True to see debug prints for model predictions
EMA_ALPHA = 0.25  # smoothing for failure % (0=no smoothing, 1=no memory)

# ---------------------------
# Styling (QSS) — Neon Blue Theme
# ---------------------------
QSS = """
QMainWindow {
    background-color: #000000;
}
QWidget {
    color: #00b7ff;
    font-family: 'Consolas';
}
QLabel#Title {
    color: #00eaff;
    font-size: 22pt;
    font-weight: bold;
}
QFrame.Card {
    background-color: #0a0a0a;
    border: 1px solid #00b7ff;
    border-radius: 10px;
    padding: 10px;
}
QPushButton {
    background-color: #00111a;
    border: 1px solid #00b7ff;
    border-radius: 6px;
    color: #00b7ff;
    font-weight: bold;
    padding: 6px;
}
QPushButton:hover {
    background-color: #002a3d;
    color: #00ffff;
}
QCheckBox { padding: 4px; color: #00b7ff; }
QLineEdit, QTextEdit {
    background-color: #0b0b0b;
    border: 1px solid #00b7ff;
    color: #00eaff;
    border-radius: 4px;
    padding: 5px;
}
QTabWidget::pane {
    border: 1px solid #00b7ff;
    background: #000000;
}
QTabBar::tab {
    background: #0b0b0b;
    color: #00b7ff;
    padding: 6px 12px;
    border: 1px solid #00b7ff;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background: #00293d;
    color: #00ffff;
}
"""

# ---------------------------
# Helper: Matplotlib Live Plot
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

# ---------------------------
# Dashboard Page (cards + graphs)
# ---------------------------
class DashboardPage(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app

        # Data queues for live graphs
        self.max_len = 120
        self.time_q = deque(maxlen=self.max_len)
        self.cpu_q = deque(maxlen=self.max_len)
        self.ram_q = deque(maxlen=self.max_len)
        self.temp_q = deque(maxlen=self.max_len)
        self.rpm_q = deque(maxlen=self.max_len)
        self.torque_q = deque(maxlen=self.max_len)
        self.failure_q = deque(maxlen=self.max_len)

        self.last_snapshot = None
        self._ema_failure = None  # for smoothing failure%

        self._build_ui()

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live)
        self.timer.start(1000)  # 1s refresh

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel("CHRONIX - MVP")
        title.setObjectName("Title")
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        # Cards
        card_frame = QGridLayout()
        card_frame.setSpacing(12)
        self.card_cpu = self.make_card("CPU %", "—")
        self.card_ram = self.make_card("RAM %", "—")
        self.card_temp = self.make_card("Air Temp [K]", "—")
        self.card_rpm = self.make_card("RPM", "—")
        self.card_torque = self.make_card("Torque [Nm]", "—")
        self.card_failure = self.make_card("Failure Status", "—")

        card_frame.addWidget(self.card_cpu[0], 0, 0)
        card_frame.addWidget(self.card_ram[0], 0, 1)
        card_frame.addWidget(self.card_temp[0], 0, 2)
        card_frame.addWidget(self.card_rpm[0], 1, 0)
        card_frame.addWidget(self.card_torque[0], 1, 1)
        card_frame.addWidget(self.card_failure[0], 1, 2)
        root.addLayout(card_frame)

        # Graph canvases (6)
        graphs = QHBoxLayout()
        self.canvas_cpu = MplCanvas()
        self.canvas_ram = MplCanvas()
        self.canvas_temp = MplCanvas()
        self.canvas_rpm = MplCanvas()
        self.canvas_torque = MplCanvas()
        self.canvas_failure = MplCanvas()

        graphs.addWidget(self.canvas_cpu)
        graphs.addWidget(self.canvas_ram)
        graphs.addWidget(self.canvas_temp)
        graphs.addWidget(self.canvas_rpm)
        graphs.addWidget(self.canvas_torque)
        graphs.addWidget(self.canvas_failure)
        root.addLayout(graphs)

        # Controls
        controls = QHBoxLayout()
        self.btn_compare = QPushButton("Compare Changes (since last click)")
        self.btn_compare.clicked.connect(self.compare_changes)
        controls.addWidget(self.btn_compare)

        self.chk_use_custom = QCheckBox("Use uploaded dataset model (if available)")
        self.chk_use_custom.stateChanged.connect(self.toggle_custom_model)
        controls.addWidget(self.chk_use_custom)

        self.failure_pct_label = QLabel("Failure %: —")
        controls.addWidget(self.failure_pct_label)

        controls.addStretch(1)
        root.addLayout(controls)

        # Live updates text box
        self.live_box = QTextEdit()
        self.live_box.setReadOnly(True)
        self.live_box.setFixedHeight(120)
        root.addWidget(self.live_box)

    def make_card(self, label_text, value_text):
        frame = QFrame()
        frame.setObjectName("Card")
        frame.setLayout(QVBoxLayout())
        frame.layout().setContentsMargins(8, 8, 8, 8)
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #bcd9ff; font-size: 9pt;")
        val = QLabel(value_text)
        val.setStyleSheet("color: #6fb8ff; font-size: 18pt; font-weight: bold;")
        frame.layout().addWidget(lbl)
        frame.layout().addWidget(val)
        frame.layout().addStretch(1)
        return frame, val

    def build_sample_for_model(self, model, raw_dict):
        """
        Build a DataFrame that matches model.feature_names_in_ if available.
        raw_dict: dict with observed features (air/process temp, rpm, torque, tool wear, target)
        """
        if model is None:
            return pd.DataFrame([raw_dict])

        # If model exposes feature_names_in_ (sklearn >=0.24), use that order / columns
        try:
            feat_names = getattr(model, "feature_names_in_", None)
            if feat_names is not None:
                row = {}
                for fn in feat_names:
                    if fn in raw_dict:
                        row[fn] = raw_dict[fn]
                    else:
                        # fill sensible default
                        if "temp" in fn.lower():
                            row[fn] = raw_dict.get('Air temperature [K]', 298.0)
                        elif "rpm" in fn.lower():
                            row[fn] = raw_dict.get('Rotational speed [rpm]', 1500)
                        elif "torque" in fn.lower():
                            row[fn] = raw_dict.get('Torque [Nm]', 0.0)
                        else:
                            row[fn] = 0.0
                return pd.DataFrame([row])
        except Exception as e:
            if DEBUG:
                print("build_sample_for_model error:", e)

        # fallback: return raw dict as DataFrame
        return pd.DataFrame([raw_dict])

    def compute_failure_from_model(self, model, sample_df):
        """
        Returns failure_pct (0-100) extracted from a model/DF robustly.
        """
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample_df)[0]
                classes = list(model.classes_)
                pred = model.predict(sample_df)[0] if hasattr(model, "predict") else None
                if DEBUG:
                    print("MODEL classes:", classes, "probs:", probs, "pred:", pred)

                # treat any class that is not 'no failure' as failure
                failure_indices = []
                for i, c in enumerate(classes):
                    name = str(c).lower()
                    if name in ("no failure", "no_failure", "none", "0", "no"):
                        continue
                    failure_indices.append(i)

                if failure_indices:
                    failure_prob = sum(probs[i] for i in failure_indices)
                else:
                    # fallback: if binary, take probability of the non-first class as failure
                    if len(probs) >= 2:
                        failure_prob = probs[-1]  # last class
                    else:
                        failure_prob = 0.0
                return max(0.0, min(100.0, float(failure_prob * 100.0)))
            else:
                pred = model.predict(sample_df)[0]
                if DEBUG:
                    print("MODEL predict (no proba):", pred)
                if str(pred).lower() in ("no failure", "no_failure", "none", "0", "no"):
                    return 0.0
                return 100.0
        except Exception as e:
            if DEBUG:
                print("compute_failure_from_model error:", e)
            return None

    def update_live(self):
        """
        Single, clean, robust update loop:
        - collects live metrics
        - updates cards & queues
        - gets active model and computes failure %
        - updates live box and redraws plots
        """
        # Gather metrics
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        air_temp = 298.0 + (cpu / 100.0) * 5.0
        process_temp = air_temp + random.uniform(0.5, 1.5)
        rpm = 1000 + int(psutil.cpu_freq().current) if psutil.cpu_freq() else 1500
        torque = round(random.uniform(3.5, 6.5), 2)
        wear = random.randint(100, 160)
        target = 1 if cpu > 80 or air_temp > 302 else 0

        # Append to queues
        ts = time.time()
        self.time_q.append(ts)
        self.cpu_q.append(cpu)
        self.ram_q.append(ram)
        self.temp_q.append(air_temp)
        self.rpm_q.append(rpm)
        self.torque_q.append(torque)

        # Update cards
        self.card_cpu[1].setText(f"{cpu:.1f} %")
        self.card_ram[1].setText(f"{ram:.1f} %")
        self.card_temp[1].setText(f"{air_temp:.1f} K")
        self.card_rpm[1].setText(str(rpm))
        self.card_torque[1].setText(f"{torque:.2f} Nm")

        # Prepare raw dict and sample_df for model
        raw = {
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rpm,
            'Torque [Nm]': torque,
            'Tool wear [min]': wear,
            'Target': target,
            'CPU %': cpu,
            'RAM %': ram
        }

        model = self.parent_app.get_active_model()

        failure_pct = None
        failure_status = "—"

        try:
            if model is not None:
                # Build sample DataFrame that matches model features (robust)
                sample_df = self.build_sample_for_model(model, raw)
                failure_pct = self.compute_failure_from_model(model, sample_df)
            else:
                # no model: produce a dynamic heuristic-based probability
                # normalize factors
                cpu_f = min(cpu / 100.0, 1.0)
                ram_f = min(ram / 100.0, 1.0)
                temp_f = min(max((air_temp - 295.0) / 20.0, 0.0), 1.0)  # 295 baseline
                torque_f = min(torque / 10.0, 1.0)
                rpm_f = min(max((rpm - 1000) / 4000.0, 0.0), 1.0)

                # weighted sum
                score = (0.30 * cpu_f + 0.20 * ram_f + 0.25 * temp_f + 0.15 * torque_f + 0.10 * rpm_f)
                # scale to percent and add some randomness
                failure_pct = min(max(score * 100.0 + random.uniform(-3.0, 3.0), 0.0), 99.9)

            # smoothing with EMA to avoid jitter
            if failure_pct is None:
                failure_pct = 0.0
            if self._ema_failure is None:
                self._ema_failure = float(failure_pct)
            else:
                self._ema_failure = EMA_ALPHA * float(failure_pct) + (1 - EMA_ALPHA) * self._ema_failure

            failure_pct_smoothed = round(self._ema_failure, 1)
            failure_status = "FAILURE" if failure_pct_smoothed > 50.0 else "Nominal"

        except Exception as e:
            if DEBUG:
                print("Prediction error:", repr(e))
            failure_status = "Error"
            failure_pct_smoothed = None

        # Update failure UI
        self.card_failure[1].setText(f"{failure_status}")
        if failure_pct_smoothed is not None:
            self.failure_pct_label.setText(f"Failure %: {failure_pct_smoothed:.1f}")
        else:
            self.failure_pct_label.setText("Failure %: —")

        # Save failure pct for plotting (use 0.0 when None)
        self.failure_q.append(failure_pct_smoothed if failure_pct_smoothed is not None else 0.0)

        # Live box
        now_str = time.strftime("%H:%M:%S")
        ftext = f"{failure_pct_smoothed:.1f}" if failure_pct_smoothed is not None else "—"
        line = f"[{now_str}] CPU={cpu:.1f}% | RAM={ram:.1f}% | Temp={air_temp:.1f}K | RPM={rpm} | Torque={torque:.2f}Nm | Failure%={ftext}\n"
        self.live_box.append(line)

        # Redraw plots
        self.redraw_plots()

    def redraw_plots(self):
        # CPU
        self.canvas_cpu.axes.cla()
        self.canvas_cpu.axes.plot(list(self.cpu_q), label="CPU %")
        self.canvas_cpu.axes.set_title("CPU % (last N)")
        self.canvas_cpu.axes.grid(True)
        self.canvas_cpu.draw()

        # RAM
        self.canvas_ram.axes.cla()
        self.canvas_ram.axes.plot(list(self.ram_q), label="RAM %")
        self.canvas_ram.axes.set_title("RAM % (last N)")
        self.canvas_ram.axes.grid(True)
        self.canvas_ram.draw()

        # Temp
        self.canvas_temp.axes.cla()
        self.canvas_temp.axes.plot(list(self.temp_q), label="Temp K")
        self.canvas_temp.axes.set_title("Air Temp [K]")
        self.canvas_temp.axes.grid(True)
        self.canvas_temp.draw()

        # RPM
        self.canvas_rpm.axes.cla()
        self.canvas_rpm.axes.plot(list(self.rpm_q), label="RPM")
        self.canvas_rpm.axes.set_title("Rotational Speed [RPM]")
        self.canvas_rpm.axes.grid(True)
        self.canvas_rpm.draw()

        # Torque
        self.canvas_torque.axes.cla()
        self.canvas_torque.axes.plot(list(self.torque_q), label="Torque")
        self.canvas_torque.axes.set_title("Torque [Nm]")
        self.canvas_torque.axes.grid(True)
        self.canvas_torque.draw()

        # Failure %
        self.canvas_failure.axes.cla()
        self.canvas_failure.axes.plot(list(self.failure_q), label="Failure %")
        self.canvas_failure.axes.set_title("Failure % (trend)")
        self.canvas_failure.axes.grid(True)
        self.canvas_failure.draw()

    def compare_changes(self):
        if not self.time_q:
            QMessageBox.information(self, "Compare", "No live data yet.")
            return

        current = {
            'CPU': self.cpu_q[-1],
            'RAM': self.ram_q[-1],
            'Temp': self.temp_q[-1],
            'RPM': self.rpm_q[-1],
            'Torque': self.torque_q[-1],
            'Failure': self.failure_q[-1] if self.failure_q else 0.0
        }

        if self.last_snapshot is None:
            self.last_snapshot = current
            QMessageBox.information(self, "Compare Snapshot", "Snapshot saved. Press Compare again later to see changes.")
            return

        parts = []
        for k in ['Temp', 'CPU', 'RAM', 'RPM', 'Torque', 'Failure']:
            diff = current[k] - self.last_snapshot.get(k, 0)
            if k == 'Temp' and abs(diff) >= 0.1:
                parts.append(f"Temp {'increased' if diff>0 else 'decreased'} by {diff:.2f} K")
            if k == 'CPU' and abs(diff) >= 0.1:
                parts.append(f"CPU {'up' if diff>0 else 'down'} by {diff:.1f}%")
            if k == 'RAM' and abs(diff) >= 0.1:
                parts.append(f"RAM {'up' if diff>0 else 'down'} by {diff:.1f}%")
            if k == 'RPM' and abs(diff) >= 1:
                parts.append(f"RPM {'up' if diff>0 else 'down'} by {int(diff)}")
            if k == 'Torque' and abs(diff) >= 0.01:
                parts.append(f"Torque {'increased' if diff>0 else 'decreased'} by {diff:.2f} Nm")
            if k == 'Failure' and abs(diff) >= 0.1:
                parts.append(f"Failure % {'increased' if diff>0 else 'decreased'} by {abs(diff):.1f}%")

        self.last_snapshot = current
        msg = "\n".join(parts) if parts else "No significant changes detected."
        QMessageBox.information(self, "Compare Results", msg)
        self.live_box.append(f"[COMPARE] {msg}\n")

    def toggle_custom_model(self, state):
        self.parent_app.use_uploaded_model = bool(state)

# ---------------------------
# Dataset Trainer Page
# ---------------------------
class TrainerPage(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.loaded_df = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Dataset Trainer")
        title.setStyleSheet("font-size: 14pt; color: #6fb8ff; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        self.path_input = QLineEdit()
        btn_browse = QPushButton("Browse CSV")
        btn_browse.clicked.connect(self.browse_csv)
        form.addRow("CSV path:", self.path_input)
        form.addRow("", btn_browse)

        self.drop_input = QLineEdit("Failure Type, UDI, Product ID, Type")
        form.addRow("Columns to drop (comma sep):", self.drop_input)
        layout.addLayout(form)

        btn_train = QPushButton("Train & Save Model (save as custom_model.joblib)")
        btn_train.clicked.connect(self.train_model)
        layout.addWidget(btn_train)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(160)
        layout.addWidget(self.log)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", os.getcwd(), "CSV Files (*.csv)")
        if path:
            self.path_input.setText(path)
            try:
                self.loaded_df = pd.read_csv(path)
                self.log.append(f"Loaded CSV: {path} (shape: {self.loaded_df.shape})")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def train_model(self):
        path = self.path_input.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Missing CSV", "Please select a valid CSV file first.")
            return
        try:
            df = pd.read_csv(path)
            drops = [c.strip() for c in self.drop_input.text().split(",") if c.strip()]
            if 'Failure Type' not in df.columns:
                QMessageBox.warning(self, "Missing Label", "CSV must contain 'Failure Type' column as label.")
                return
            X = df.drop(columns=drops)
            Y = df['Failure Type']
            model = DecisionTreeClassifier()
            model.fit(X, Y)
            joblib.dump(model, "custom_model.joblib")
            self.parent_app.custom_model_path = "custom_model.joblib"
            self.log.append(f"Trained DecisionTree on {X.shape[1]} features. Saved as custom_model.joblib")
            QMessageBox.information(self, "Trained", "Model trained and saved to custom_model.joblib")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self.log.append("Error: " + str(e))

# ---------------------------
# About Page
# ---------------------------
class AboutPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("About Chronix v4 Pro")
        title.setStyleSheet("font-size: 14pt; color: #6fb8ff; font-weight: bold;")
        layout.addWidget(title)
        about_text = QLabel(
            "Chronix v4 Pro — AI Hardware Predictor and Live Dashboard\n\n"
            "Features:\n"
            "- Live hardware sim + predictive model\n"
            "- Live graphs: CPU, RAM, Temp, RPM, Torque, Failure%\n"
            "- Upload and train your own dataset (DecisionTree)\n"
            "- Compare changes (snapshots)\n\n"
            "Built for: The Smiley Moon\n"
        )
        about_text.setWordWrap(True)
        layout.addWidget(about_text)
        layout.addStretch(1)

# ---------------------------
# Main Application Window
# ---------------------------
class ChronixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chronix v4 Pro")
        self.resize(1300, 760)
        self.setStyleSheet(QSS)

        self.default_model_path = "predictive_model.joblib"
        self.custom_model_path = None
        self.use_uploaded_model = False
        self.default_model = None

        if os.path.exists(self.default_model_path):
            try:
                self.default_model = joblib.load(self.default_model_path)
                if DEBUG:
                    print("Loaded default model.")
            except Exception as e:
                print("Could not load default model:", e)

        # Build tabs
        container = QWidget()
        root = QVBoxLayout(container)
        self.tab_widget = QTabWidget()
        root.addWidget(self.tab_widget)

        self.dashboard_page = DashboardPage(self)
        self.trainer_page = TrainerPage(self)
        self.about_page = AboutPage()

        self.tab_widget.addTab(self.dashboard_page, "Dashboard")
        self.tab_widget.addTab(self.trainer_page, "Dataset Trainer")
        self.tab_widget.addTab(self.about_page, "About")

        self.setCentralWidget(container)

    def get_active_model(self):
        if self.use_uploaded_model and self.custom_model_path and os.path.exists(self.custom_model_path):
            try:
                return joblib.load(self.custom_model_path)
            except Exception as e:
                if DEBUG:
                    print("Error loading custom model:", e)
                return self.default_model
        return self.default_model

# ---------------------------
# Run the app
# ---------------------------
def main():
    app = QApplication(sys.argv)
    window = ChronixApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
