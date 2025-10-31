<h1 align="center">
  ⚡ CHRONIX — AI Hardware Predictor  
</h1>

<p align="center">
  <img src="925f5c74-2d8f-45b5-9344-f237cfdc8823.png" width="200" alt="Chronix Logo"/>
</p>

<h3 align="center">
  🧠 Predict. Detect. Prevent.  
</h3>

---

### 🧩 Overview  
**CHRONIX** is an **AI-powered hardware health prediction system** designed for **real-time monitoring** and **predictive maintenance**.  
It continuously analyzes **CPU, RAM, Temperature, Torque, and RPM** using **Machine Learning** to forecast potential failures **before they occur**.  
You can share your feedbacks in
https://discord.gg/tZ28bE8RN
🌀 Featuring a **dark futuristic GUI**, **auto-refresh every second**, and **interactive live graphs** — fully **VIBE CODED** using **PyQt5**, **psutil**, **pandas**, **scikit-learn**, and **matplotlib**.
> ⚠️ *Note: The AI ML Bot is externally integrated.*


---

### 🚀 Features  
- 🔍 Real-time CPU, RAM, and Temperature tracking  
- 🤖 AI-based hardware failure prediction (`predictive_maintenance.csv`)  
- 📊 Live graph visualization (CPU, RAM, Temp, RPM, Torque)  
- ⚙️ Manual dataset loader for training custom AI models  
- 💡 Compare last vs current system states  
- 🌑 Smooth futuristic dark UI  

---

### 💻 Run Chronix MVP  

Clone the repository and install all required dependencies:

```bash
git clone https://github.com/smilymouth/Chronix-MVP.git
cd Chronix-MVP
pip install -r requirements.txt
python Chronix.py
If requirements.txt is missing, install manually:

bash
Copy code
pip install pyqt5 psutil pandas scikit-learn matplotlib
🧠 Predictive Model
Chronix uses a Random Forest Classifier trained on a predictive maintenance dataset to estimate real-time failure probabilities.
The model adapts dynamically to your system’s live stats, analyzing parameters like:

CPU Load

RAM Usage

Temperature (Kelvin)

Rotational Speed (RPM)

Torque (Nm)

📦 Folder Structure
Copy code
Chronix-MVP/
│
├── Chronix_MVP.py
├── predictive_maintenance.csv
├── requirements.txt
├── 925f5c74-2d8f-45b5-9344-f237cfdc8823.png
└── README.md
🔥 VIBE CODED
This entire MVP (excluding the AI ML bot) is VIBE CODED — blending art, code, and intelligence into one ecosystem.
Every pixel and prediction is tuned to reflect The Smiley Moon’s futuristic design philosophy.

🧰 Tech Stack
Module	Purpose
PyQt5	GUI Framework
psutil	Hardware Data Monitoring
pandas	Data Processing
scikit-learn	AI/ML Model
matplotlib	Real-time Graphs

📥 Download
Grab the latest ZIP release directly from GitHub:
🔗 Download Chronix-MVP

👨‍💻 Developer
The Smiley Moon
Founder — VIBE CODED
💻 Ethical Hacker | ML Developer | Creator of Chronix

🛡️ License
Released under the MIT License — free to use, modify, and share with proper credit.

<p align="center"> <b>⚡ VIBE CODED | Predict. Detect. Prevent. ⚡</b> </p> ```
