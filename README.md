Household Power Load Forecasting & Power Cut Prediction
📌 Overview
This project is a full-stack machine learning application that forecasts household electricity load and predicts potential power cuts based on historical consumption, weather, and voltage data.
It combines deep learning (LSTM) for time-series forecasting with Random Forest classification for outage prediction, and delivers results through a Django-powered web interface.

🚀 Features
Next-Day Load Forecasting using LSTM trained on 460,000+ historical records.

Power Cut Prediction using Random Forest with 91%+ accuracy.

Weather & Voltage Integration for more reliable predictions.

Django Web Interface for personalized, date-specific results.

User Authentication with household-specific data history.

Scalable Deployment with model persistence (.h5 and .joblib files).

🛠 Tech Stack
Machine Learning

LSTM (Keras/TensorFlow) for load forecasting

Random Forest Classifier (scikit-learn) for power cut prediction

MinMaxScaler for feature scaling

Backend

Python, Django

Pandas, NumPy for data processing

Frontend

HTML, CSS, Bootstrap

Deployment & Storage

Model files (.h5, .joblib) stored locally / in cloud

CSV-based dataset ingestion

📊 Dataset
Household electricity usage data (2023–2024)

Weather metrics: temperature, humidity, precipitation, wind speed

Voltage readings

~460,000 total time-series records

📈 Results
LSTM Model: <1.2 kW Mean Absolute Error (MAE) for next-day load prediction.

Random Forest Classifier: 91%+ accuracy for binary power cut detection.

End-to-End Prediction Time: <2 seconds per request.

🔍 How It Works
Data Preprocessing

Load, clean, and scale input features.

Prepare 30-day historical sequences for LSTM.

Load Forecasting

LSTM predicts next-day household electricity consumption.

Power Cut Prediction

Random Forest predicts outage probability using forecasted load + weather & voltage.

Web Output

Display predictions and probability scores to the authenticated household user.

📂 Project Structure
php
Copy
Edit
.
├── pow_app/
│   ├── views.py              # Django views for prediction
│   ├── templates/            # HTML templates
│   ├── static/                # CSS/JS files
│   ├── model/                # Trained models (.h5, .joblib, scalers) and datasets
│   └── Merged_Load_Weather_2019_2020.csv
├── manage.py
├── requirements.txt
└── README.md
⚡ Installation
bash
Copy
Edit
# Clone repository
git clone https://github.com/your-username/power-forecasting.git
cd power-forecasting

# Install dependencies
pip install -r requirements.txt

# Run Django server
python manage.py runserver
📌 Usage
Log in to your household account.

Select forecast type: today or future.

View predicted load (kW) and power cut probability.

Make informed decisions on electricity usage.

🏆 Achievements
Built & deployed a production-ready ML pipeline for real-time predictions.

Integrated deep learning & traditional ML models in a single application.

Reduced prediction error to <1.2 kW and achieved 91% accuracy in outage prediction.