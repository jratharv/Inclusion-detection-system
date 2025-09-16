Real-Time Inclusion Detection System 🚨🤖

An AI-powered real-time monitoring system that uses an LSTM deep learning model to detect inclusions (abnormal patterns) in synthetic industrial sensor data.

🔹 Features

Flask + Socket.IO Backend (app.py)

LSTM model training with synthetic data

Real-time data stream simulation

REST APIs for training, start/stop detection, and fetching live results

WebSocket events for real-time updates

Interactive Web Dashboard (index.html, app.js)

Train and start/stop detection with a click

Real-time charts (sensor signal & confidence) using Plotly

Live status indicators and detection logs

Alerts with confidence levels when inclusions are detected

Standalone Console Version (inclusion_detection.py)

CLI-based detection system with Matplotlib visualizations

Real-time logs and alerts in terminal

🛠️ Tech Stack

Backend: Python, Flask, Flask-SocketIO, TensorFlow/Keras, NumPy

Frontend: HTML, Bootstrap 5, JavaScript, Plotly, Socket.IO

Visualization: Real-time signal and confidence plots

🚀 How It Works

Train Model: Generates synthetic sensor data and trains an LSTM-based anomaly detector.

Start Detection: Simulates live sensor data in real-time.

Monitor Dashboard: View signals, confidence scores, and inclusion alerts instantly.

📂 Project Structure
├── app.py                  # Flask backend with LSTM model and APIs
├── inclusion_detection.py   # Standalone console-based version
├── templates/
│   └── index.html           # Web dashboard UI
├── static/js/
│   └── app.js               # Frontend logic (charts, events, WebSocket)
└── static/css/
    └── style.css            # Styling for dashboard

🎯 Use Cases

Industrial process monitoring

Real-time anomaly detection

AI-powered safety systems


TO INSTALL OTHER MODULE USE 
pip install flask flask-socketio eventlet tensorflow numpy matplotlib
