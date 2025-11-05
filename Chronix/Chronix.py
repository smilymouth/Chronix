# chronix_v5_launch.py — Chronix MVP (Product Hunt launch edition)
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
    QTabWidget, QFormLayout, QLineEdit, QCheckBox, QTextEdit, QProgressBar,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QVariantAnimation
from PyQt5.QtGui import QFont, QPixmap

# Optional pyqtgraph (more fluid, neon graphs)
USE_PYQTGRAPH = False
try:
    import pyqtgraph as pg
    USE_PYQTGRAPH = True
except Exception:
    USE_PYQTGRAPH = False

# Matplotlib fallback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Config
# ---------------------------
DEBUG = False
EMA_ALPHA = 0.25
MAX_LEN = 120
MODEL_NAME = "Chronix MVP"

# ---------------------------
# Futuristic QSS (launch-ready)
# ---------------------------
QSS = """
QMainWindow { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #00000f, stop:1 #00121b); }
QWidget { color: #AEEFFF; font-family: 'Orbitron', 'JetBrains Mono', 'Consolas'; }
#Title { font-size: 28pt; color: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00f0ff, stop:1 #00b7ff); letter-spacing:2px; }
QFrame.card {
    background: rgba(10,14,20,0.60);
    border: 1px solid rgba(0,183,255,0.18);
    border-radius: 14px;
    padding: 12px;
}
QLabel.cardLabel { color: #bfefff; font-size: 9pt; }
QLabel.cardValue { color: #dff9ff; font-size: 18pt; font-weight:700; }
#failureValue { font-size: 18pt; font-weight:900; color: #ff9b9b; }
QProgressBar {
    border-radius: 8px;
    text-align: center;
    color: #00121b;
    font-weight: 700;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00f0ff, stop:1 #00b7ff);
}
#reasonBox {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(0,183,255,0.06), stop:1 rgba(0,183,255,0.02));
    border: 1px solid rgba(0,183,255,0.22);
    border-radius: 10px;
    padding: 10px;
    color: #e8fbff;
    font-weight: 700;
}
QPushButton {
    background: rgba(0,20,30,0.6);
    border: 1px solid rgba(0,183,255,0.25);
    padding: 8px 12px;
    border-radius: 10px;
    color: #bff6ff;
}
QPushButton:hover { background: rgba(0,50,70,0.8); transform: translateY(-1px); }
QLineEdit, QTextEdit {
    background: rgba(0,10,14,0.6);
    border: 1px solid rgba(0,183,255,0.12);
    border-radius: 8px;
    color: #e6fbff;
    padding: 6px;
}
QTabWidget::pane { border-radius: 12px; padding: 6px; background: transparent; }
QTabBar::tab { background: rgba(0,10,14,0.6); color: #9feeff; padding: 8px 14px; border-radius: 8px; margin-right:6px; }
QTabBar::tab:selected { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #002b3d, stop:1 #004a66); color: #eaffff; }
"""

# ---------------------------
# Helpers: Plot widgets
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

def make_pyqtgraph_plot(title):
    pw = pg.PlotWidget()
    pw.setBackground(None)
    pw.showGrid(x=True, y=True, alpha=0.12)
    pw.setTitle(title)
    pw.getAxis('left').setPen(pg.mkPen((150, 220, 255), width=1))
    pw.getAxis('bottom').setPen(pg.mkPen((120, 180, 220), width=1))
    return pw

# Replace your detect_failure_reason() with this
_last_reason = None

def detect_failure_reason(raw, failure_pct):
    global _last_reason
    reasons = []
    cpu, ram = raw.get('CPU %', 0), raw.get('RAM %', 0)
    air_temp = raw.get('Air temperature [K]', 0)
    torque, rpm = raw.get('Torque [Nm]', 0), raw.get('Rotational speed [rpm]', 0)

    if cpu > 85: reasons.append("CPU Thermal Strain")
    elif cpu > 70: reasons.append("CPU Utilization Spikes")

    if air_temp > 305: reasons.append("Overheat — Cooling Inefficient")
    elif air_temp > 300: reasons.append("Rising Temperature Trend")

    if ram > 85: reasons.append("Memory Pressure Detected")
    elif ram > 75: reasons.append("RAM Allocation Near Limit")

    if torque > 6.0: reasons.append("Mechanical Torque Surge")
    if rpm < 1300: reasons.append("RPM Drop — Efficiency Loss")

    if not reasons and failure_pct > 60:
        reasons = ["System Anomaly", "Unexpected Load Pattern", "Irregular Sensor Response"]

    if reasons:
        # pick reason not same as last
        choices = [r for r in reasons if r != _last_reason]
        reason = random.choice(choices or reasons)
    else:
        reason = "Stable — No dominant factor"

    _last_reason = reason
    return reason

# ---------------------------
# Dashboard Page
# ---------------------------
class DashboardPage(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.time_q = deque(maxlen=MAX_LEN)
        self.cpu_q = deque(maxlen=MAX_LEN)
        self.ram_q = deque(maxlen=MAX_LEN)
        self.temp_q = deque(maxlen=MAX_LEN)
        self.rpm_q = deque(maxlen=MAX_LEN)
        self.torque_q = deque(maxlen=MAX_LEN)
        self.failure_q = deque(maxlen=MAX_LEN)
        self._ema_failure = None
        self.last_snapshot = None

        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live)
        self.timer.start(900)

        # pulse animation for failure reason highlight
        self._pulse = QVariantAnimation()
        self._pulse.setStartValue(6.0)
        self._pulse.setEndValue(18.0)
        self._pulse.setDuration(700)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_pulse)
        self._pulse.stop()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title_box = QHBoxLayout()
        if os.path.exists("logo.png"):
            lbl_logo = QLabel()
            px = QPixmap("logo.png").scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_logo.setPixmap(px)
            title_box.addWidget(lbl_logo)
        title = QLabel(f"{MODEL_NAME}")
        title.setObjectName("Title")
        title_box.addWidget(title)
        subtitle = QLabel("Live AI Hardware Predictor")
        subtitle.setStyleSheet("color:#9feeff; font-size:10pt; margin-left:8px;")
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch()
        root.addLayout(top)

        # cards layout
        cards = QGridLayout()
        cards.setSpacing(14)
        self.card_cpu = self.make_card("CPU %", "—")
        self.card_ram = self.make_card("RAM %", "—")
        self.card_temp = self.make_card("Air Temp [K]", "—")
        self.card_rpm = self.make_card("RPM", "—")
        self.card_torque = self.make_card("Torque [Nm]", "—")
        # failure percent separate card with progress bar
        self.card_failure_percent = self.make_card("Failure %", "—", progress=True)
        # failure reason separate box (wide)
        reason_frame = QFrame()
        reason_frame.setObjectName("reasonBox")
        reason_frame.setLayout(QVBoxLayout())
        reason_frame.layout().setContentsMargins(10, 10, 10, 10)
        lbl = QLabel("Failure Reason")
        lbl.setStyleSheet("font-size:10pt; color:#cfefff;")
        self.reason_val = QLabel("—")
        self.reason_val.setStyleSheet("font-size:13pt; font-weight:800; color:#ffdede;")
        reason_frame.layout().addWidget(lbl)
        reason_frame.layout().addWidget(self.reason_val)
        reason_frame.setFixedHeight(100)

        cards.addWidget(self.card_cpu[0], 0, 0)
        cards.addWidget(self.card_ram[0], 0, 1)
        cards.addWidget(self.card_temp[0], 0, 2)
        cards.addWidget(self.card_rpm[0], 1, 0)
        cards.addWidget(self.card_torque[0], 1, 1)
        cards.addWidget(self.card_failure_percent[0], 1, 2)
        # add reason under cards (span columns)
        root.addLayout(cards)
        root.addWidget(reason_frame)

        # plots area
        plots = QHBoxLayout()
        if USE_PYQTGRAPH:
            self.plot_cpu = make_pyqtgraph_plot("CPU %")
            self.plot_ram = make_pyqtgraph_plot("RAM %")
            self.plot_temp = make_pyqtgraph_plot("Air Temp [K]")
            self.plot_failure = make_pyqtgraph_plot("Failure %")
            plots.addWidget(self.plot_cpu)
            plots.addWidget(self.plot_ram)
            plots.addWidget(self.plot_temp)
            plots.addWidget(self.plot_failure)
            self._pg_curves = {
                'cpu': self.plot_cpu.plot(pen=pg.mkPen((0,200,255), width=2)),
                'ram': self.plot_ram.plot(pen=pg.mkPen((0,160,255), width=2)),
                'temp': self.plot_temp.plot(pen=pg.mkPen((100,255,200), width=2)),
                'failure': self.plot_failure.plot(pen=pg.mkPen((255,60,60), width=2))
            }
        else:
            self.canvas_cpu = MplCanvas()
            self.canvas_ram = MplCanvas()
            self.canvas_temp = MplCanvas()
            self.canvas_failure = MplCanvas()
            plots.addWidget(self.canvas_cpu)
            plots.addWidget(self.canvas_ram)
            plots.addWidget(self.canvas_temp)
            plots.addWidget(self.canvas_failure)
        root.addLayout(plots)

        # controls
        controls = QHBoxLayout()
        self.btn_compare = QPushButton("Compare (snapshot)")
        self.btn_compare.clicked.connect(self.compare_changes)
        controls.addWidget(self.btn_compare)
        self.chk_use_custom = QCheckBox("Use uploaded model")
        self.chk_use_custom.stateChanged.connect(self.toggle_custom_model)
        controls.addWidget(self.chk_use_custom)
        self.failure_pct_label = QLabel("Failure %: —")
        controls.addWidget(self.failure_pct_label)
        controls.addStretch()
        root.addLayout(controls)

        # live feed
        self.live_box = QTextEdit()
        self.live_box.setFixedHeight(140)
        self.live_box.setReadOnly(True)
        root.addWidget(self.live_box)

        # keep reference for pulsing reason box
        self.reason_frame = reason_frame

        # drop shadow
        sh = QGraphicsDropShadowEffect()
        sh.setBlurRadius(14)
        sh.setXOffset(0)
        sh.setYOffset(2)
        sh.setColor(Qt.black)
        reason_frame.setGraphicsEffect(sh)
        self._reason_shadow = sh

    def make_card(self, label_text, value_text, progress=False):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setProperty("class", "card")
        frame.setLayout(QVBoxLayout())
        frame.layout().setContentsMargins(10, 10, 10, 10)
        lbl = QLabel(label_text)
        lbl.setProperty("class", "cardLabel")
        val = QLabel(value_text)
        val.setProperty("class", "cardValue")

        frame.layout().addWidget(lbl)
        frame.layout().addWidget(val)
        if progress:
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFixedHeight(20)
            frame.layout().addWidget(pbar)
        frame.layout().addStretch()
        # drop shadow
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(8)
        effect.setXOffset(0)
        effect.setYOffset(2)
        effect.setColor(Qt.black)
        frame.setGraphicsEffect(effect)
        # return references
        if progress:
            return frame, val, pbar
        return frame, val

    def build_sample_for_model(self, model, raw_dict):
        if model is None:
            return pd.DataFrame([raw_dict])
        try:
            feat_names = getattr(model, "feature_names_in_", None)
            if feat_names is not None:
                row = {}
                for fn in feat_names:
                    if fn in raw_dict:
                        row[fn] = raw_dict[fn]
                    else:
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
            if DEBUG: print("build_sample error", e)
        return pd.DataFrame([raw_dict])

    def compute_failure_from_model(self, model, sample_df):
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample_df)[0]
                classes = list(model.classes_)
                failure_indices = []
                for i, c in enumerate(classes):
                    name = str(c).lower()
                    if name in ("no failure", "no_failure", "none", "0", "no"):
                        continue
                    failure_indices.append(i)
                if failure_indices:
                    failure_prob = sum(probs[i] for i in failure_indices)
                else:
                    failure_prob = probs[-1] if len(probs) >= 2 else 0.0
                return max(0.0, min(100.0, float(failure_prob * 100.0)))
            else:
                pred = model.predict(sample_df)[0]
                if str(pred).lower() in ("no failure", "no_failure", "none", "0", "no"):
                    return 0.0
                return 100.0
        except Exception as e:
            if DEBUG: print("compute error", e)
            return None

    def update_live(self):
        # gather metrics
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        air_temp = 298.0 + (cpu / 100.0) * 5.0
        process_temp = air_temp + random.uniform(0.4, 1.6)
        rpm = 1000 + int(psutil.cpu_freq().current) if psutil.cpu_freq() else 1500
        torque = round(random.uniform(3.5, 6.5), 2)
        wear = random.randint(100, 160)
        target = 1 if cpu > 80 or air_temp > 302 else 0

        ts = time.time()
        self.time_q.append(ts)
        self.cpu_q.append(cpu)
        self.ram_q.append(ram)
        self.temp_q.append(air_temp)
        self.rpm_q.append(rpm)
        self.torque_q.append(torque)

        # update simple cards
        self.card_cpu[1].setText(f"{cpu:.1f} %")
        self.card_ram[1].setText(f"{ram:.1f} %")
        self.card_temp[1].setText(f"{air_temp:.1f} K")
        self.card_rpm[1].setText(str(rpm))
        self.card_torque[1].setText(f"{torque:.2f} Nm")

        # prepare raw dict
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
        status = "—"
        try:
            if model is not None:
                sample_df = self.build_sample_for_model(model, raw)
                failure_pct = self.compute_failure_from_model(model, sample_df)
            else:
                cpu_f = min(cpu / 100.0, 1.0)
                ram_f = min(ram / 100.0, 1.0)
                temp_f = min(max((air_temp - 295.0) / 20.0, 0.0), 1.0)
                torque_f = min(torque / 10.0, 1.0)
                rpm_f = min(max((rpm - 1000) / 4000.0, 0.0), 1.0)
                score = 0.30 * cpu_f + 0.20 * ram_f + 0.25 * temp_f + 0.15 * torque_f + 0.10 * rpm_f
                failure_pct = min(max(score * 100.0 + random.uniform(-3.0, 3.0), 0.0), 99.9)

            if failure_pct is None:
                failure_pct = 0.0
            if self._ema_failure is None:
                self._ema_failure = float(failure_pct)
            else:
                self._ema_failure = EMA_ALPHA * float(failure_pct) + (1 - EMA_ALPHA) * self._ema_failure
            f_sm = round(self._ema_failure, 1)
            status = "FAILURE" if f_sm > 50.0 else "Nominal"
        except Exception as e:
            if DEBUG: print("prediction error", e)
            status = "Error"
            f_sm = None

        # update failure percent card
        if f_sm is not None:
            self.card_failure_percent[1].setText(f"{f_sm:.1f} %")
            self.card_failure_percent[2].setValue(int(f_sm))
            self.failure_q.append(f_sm)
            self.failure_pct_label.setText(f"Failure %: {f_sm:.1f}")
        else:
            self.card_failure_percent[1].setText("—")
            self.card_failure_percent[2].setValue(0)
            self.failure_q.append(0.0)
            self.failure_pct_label.setText("Failure %: —")

        # determine failure reason only when failure occurs (or error)
        reason = "—"
        if status == "FAILURE":
            reason = detect_failure_reason(raw, f_sm)
            # start pulsing highlight
            if not self._pulse.state():
                self._pulse.start()
        else:
            # stop pulsing & reset style
            if self._pulse.state():
                self._pulse.stop()
                self._reason_shadow.setBlurRadius(14)
            reason = "No current failure"

        # update reason box
        self.reason_val.setText(reason)

        now_str = time.strftime("%H:%M:%S")
        ftext = f"{f_sm:.1f}" if f_sm is not None else "—"
        line = f"[{now_str}] CPU={cpu:.1f}% | RAM={ram:.1f}% | Temp={air_temp:.1f}K | RPM={rpm} | Tor={torque:.2f}Nm | Failure%={ftext} | Reason={reason}"
        self.live_box.append(line)

        # redraw plots
        self.redraw_plots()

    def _on_pulse(self, val):
        blur = float(val)
        self._reason_shadow.setBlurRadius(blur)
        # tint reason text with red glow proportional to blur
        intensity = min(255, int((blur - 6) / 12 * 255))
        r = 255
        g = max(80, 255 - intensity)
        b = max(80, 255 - intensity)
        self.reason_val.setStyleSheet(f"font-size:13pt; font-weight:800; color: rgb({r},{g},{b});")

    def redraw_plots(self):
        if USE_PYQTGRAPH:
            xs = list(range(len(self.cpu_q)))
            self._pg_curves['cpu'].setData(xs, list(self.cpu_q))
            self._pg_curves['ram'].setData(xs, list(self.ram_q))
            self._pg_curves['temp'].setData(xs, list(self.temp_q))
            self._pg_curves['failure'].setData(xs, list(self.failure_q))
        else:
            self.canvas_cpu.axes.cla()
            self.canvas_cpu.axes.plot(list(self.cpu_q))
            self.canvas_cpu.axes.set_title("CPU %")
            self.canvas_cpu.draw()

            self.canvas_ram.axes.cla()
            self.canvas_ram.axes.plot(list(self.ram_q))
            self.canvas_ram.axes.set_title("RAM %")
            self.canvas_ram.draw()

            self.canvas_temp.axes.cla()
            self.canvas_temp.axes.plot(list(self.temp_q))
            self.canvas_temp.axes.set_title("Air Temp [K]")
            self.canvas_temp.draw()

            self.canvas_failure.axes.cla()
            self.canvas_failure.axes.plot(list(self.failure_q))
            self.canvas_failure.axes.set_title("Failure %")
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
            QMessageBox.information(self, "Snapshot", "Snapshot saved. Press Compare again to see changes.")
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
        msg = "\n".join(parts) if parts else "No significant changes."
        QMessageBox.information(self, "Compare Results", msg)
        self.live_box.append(f"[COMPARE] {msg}")

    def toggle_custom_model(self, state):
        self.parent_app.use_uploaded_model = bool(state)

# ---------------------------
# Trainer & About
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
        title.setStyleSheet("font-size:14pt; color:#9feeff; font-weight:700;")
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
        btn_train = QPushButton("Train & Save Model")
        btn_train.clicked.connect(self.train_model)
        layout.addWidget(btn_train)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(160)
        layout.addWidget(self.log)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", os.getcwd(), "CSV Files (*.csv)")
        if path:
            self.path_input.setText(path)
            try:
                self.loaded_df = pd.read_csv(path)
                self.log.append(f"Loaded CSV: {path} shape={self.loaded_df.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def train_model(self):
        path = self.path_input.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Missing CSV", "Select a valid CSV first.")
            return
        try:
            df = pd.read_csv(path)
            drops = [c.strip() for c in self.drop_input.text().split(",") if c.strip()]
            if 'Failure Type' not in df.columns:
                QMessageBox.warning(self, "Missing Label", "CSV must contain 'Failure Type' column.")
                return
            X = df.drop(columns=drops)
            Y = df['Failure Type']
            model = DecisionTreeClassifier()
            model.fit(X, Y)
            joblib.dump(model, "custom_model.joblib")
            self.parent_app.custom_model_path = "custom_model.joblib"
            self.log.append(f"Trained DecisionTree; saved as custom_model.joblib")
            QMessageBox.information(self, "Trained", "Model trained & saved.")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))

class AboutPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        t = QLabel(f"About {MODEL_NAME}")
        t.setStyleSheet("font-size:14pt;color:#9feeff;font-weight:700;")
        layout.addWidget(t)
        about = QLabel(f"{MODEL_NAME} — Neon dashboard, Failure % + Reason panel, Product Hunt ready.\nBuilt for: The Smiley Moon")
        about.setWordWrap(True)
        layout.addWidget(about)
        layout.addStretch(1)

# ---------------------------
# Main App
# ---------------------------
class ChronixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(MODEL_NAME)
        self.resize(1400, 820)
        self.setStyleSheet(QSS)
        self.default_model_path = "predictive_model.joblib"
        self.custom_model_path = None
        self.use_uploaded_model = False
        self.default_model = None
        if os.path.exists(self.default_model_path):
            try:
                self.default_model = joblib.load(self.default_model_path)
                if DEBUG: print("Loaded default model")
            except Exception as e:
                print("Could not load default model:", e)
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

        # startup fade-in
        self.setWindowOpacity(0.0)
        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(700)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def get_active_model(self):
        if self.use_uploaded_model and self.custom_model_path and os.path.exists(self.custom_model_path):
            try:
                return joblib.load(self.custom_model_path)
            except Exception as e:
                if DEBUG: print("Error loading custom model:", e)
                return self.default_model
        return self.default_model

def main():
    app = QApplication(sys.argv)
    win = ChronixApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
