<h1 align="center">⚡ CHRONIX — AI Hardware Health Predictor</h1>

<p align="center">
  <img src="logo.png" alt="Chronix Logo" width="220">
</p>

<p align="center">
  <b>Predict • Detect • Prevent</b>
</p>

<hr>

<h2>🧠 Overview</h2>

<p>
<b>CHRONIX</b> is an AI-powered hardware monitoring and predictive maintenance system designed to analyze system health in real time.
</p>

<p>
The application continuously monitors system metrics such as <b>CPU usage, RAM usage, temperature, rotational speed (RPM), and torque</b>.  
Using a Machine Learning model, Chronix predicts potential hardware failures before they occur.
</p>

<hr>

<h2>🚀 Features</h2>

<ul>
<li>📊 Real-time hardware monitoring</li>
<li>🤖 Machine Learning based failure prediction</li>
<li>📈 Live graphical visualization of system metrics</li>
<li>🔁 Auto-refresh monitoring every second</li>
<li>🧪 Custom dataset loader for training models</li>
<li>📉 Compare previous vs current system states</li>
<li>🌙 Modern dark themed interface</li>
</ul>

<hr>

<h2>🧰 Tech Stack</h2>

<table>
<tr>
<th>Technology</th>
<th>Purpose</th>
</tr>

<tr>
<td>PyQt5</td>
<td>Graphical User Interface</td>
</tr>

<tr>
<td>psutil</td>
<td>Hardware monitoring</td>
</tr>

<tr>
<td>pandas</td>
<td>Data processing</td>
</tr>

<tr>
<td>scikit-learn</td>
<td>Machine learning model</td>
</tr>

<tr>
<td>matplotlib</td>
<td>Real-time graphs</td>
</tr>

</table>

<hr>

<h2>🧠 Predictive Model</h2>

<p>
Chronix uses a <b>Random Forest Classifier</b> trained on a predictive maintenance dataset.
</p>

<p>The model analyzes:</p>

<ul>
<li>CPU Load</li>
<li>RAM Usage</li>
<li>Temperature (Kelvin)</li>
<li>Rotational Speed (RPM)</li>
<li>Torque (Nm)</li>
</ul>

<p>
Based on these metrics, the model estimates the probability of potential hardware failures.
</p>

<hr>

<h2>📦 Project Structure</h2>

<pre>
Chronix/
│
├── Chronix.py
├── predictive_maintenance.csv
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>💻 Installation</h2>

<pre>
git clone https://github.com/smilymouth/Chronix.git
cd Chronix
pip install -r requirements.txt
python Chronix.py
</pre>

<hr>

<h2>💬 Community</h2>

<p>
Join the Discord community for feedback and discussions:
</p>

<p>
<a href="https://discord.gg/tZ28bE8RN">Join Discord</a>
</p>

<hr>

<h2>👨‍💻 Developer</h2>

<p>
<b>The Smiley Moon</b><br>
Ethical Hacker • Developer • Creator of Chronix
</p>

<hr>

<h2>🛡 License</h2>

<p>
Released under the <b>MIT License</b>.  
Free to use, modify, and distribute with attribution.
</p>

<hr>

<p align="center">
<b>⚡ CHRONIX — AI Powered Hardware Prediction ⚡</b>
</p>
