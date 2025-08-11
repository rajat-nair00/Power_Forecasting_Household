from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.hashers import check_password, make_password
from django.contrib.auth.decorators import login_required
from django.contrib.auth import  logout
from keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from joblib import load
import tensorflow as tf
from joblib import load
import pickle
from sklearn.preprocessing import MinMaxScaler
from django.contrib import messages
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
import logging
from .models import Users
from django.conf import settings

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'pow_app', 'model')

# Load weather dataset once
weather_data_path = os.path.join(MODEL_DIR, 'All_India_Weather_2023_2024.csv')
weather_df = pd.read_csv(weather_data_path)
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df['date'] = weather_df['time'].dt.date

# Load models once globally (outside the view)
lstm_model = load_model(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\lstm_model.h5',compile=False)
rf_model = joblib.load(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\rf_model.joblib')
power_cut_classifier = joblib.load(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\power_cut_classifier.joblib')
scaler = joblib.load(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\scaler.joblib')
label_encoder_state = joblib.load(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\state_label_encoder.joblib')

# Logger setup
logger = logging.getLogger(__name__)

# --- Views ---

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == "POST":
        name1 = request.POST.get("name")
        gender1 = request.POST.get("gender")
        dob1 = request.POST.get("dob")
        email1 = request.POST.get("email")
        state1 = request.POST.get("state")
        household_id1 = request.POST.get("household_id")
        

        if not (name1 and gender1 and dob1 and email1 and state1 and household_id1):
            messages.error(request, "All fields are required.")
            return render(request, 'register.html')

        user = Users(Name=name1, Gender=gender1, Dob=dob1, Email=email1, State=state1,Household_ID=household_id1)
        user.save()
        request.session['email'] = email1
        request.session['household_id'] = household_id1
        return redirect('createlogin')

    return render(request, 'register.html')

def createlogin(request):
    if request.method == 'POST':
        email = request.POST.get("email")
        pass1 = request.POST.get("password1")
        pass2 = request.POST.get("password2")

        if not email:
            messages.error(request, "Email is required.")
            return render(request, 'Createlogin.html')

        try:
            user = Users.objects.get(Email=email)
        except Users.DoesNotExist:
            messages.error(request, "User not found. Please register first.")
            return render(request, 'Createlogin.html')

        if pass1 != pass2:
            messages.error(request, "Passwords do not match!")
            return render(request, 'Createlogin.html')

        user.Pwd = make_password(pass1)
        user.save()
        messages.success(request, "Login credentials created. Please login.")
        return redirect('custom_login')

    return render(request, 'Createlogin.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')

        user = Users.objects.filter(Email=email).first()
        if not user or not check_password(password, user.Pwd):
            messages.error(request, "Invalid username or password")
            return render(request, 'login.html')
        response = redirect('dashboard')
        
        max_age = 30 * 24 * 60 * 60 if remember_me else 60 * 60
        response.set_cookie('logged_in_user', email, max_age=max_age)

        return response

    return render(request, 'login.html')

def dashboard(request):
    email = request.session.get('email')
    if not email:
        return redirect('custom_login')

    user = Users.objects.filter(Email=email).first()
    return render(request, 'dashboard.html', {'user': user})

def predict(request):
    return render(request, 'predict.html')

def get_weather_data_from_api(city):
    api_key = "f093470b8ce5c04ac0f132c471f4a95a"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature_2m_max': data['main']['temp_max'],
            'temperature_2m_min': data['main']['temp_min'],
            'precipitation_sum': 0.0,  # API doesn't return precipitation_sum, can be adjusted
            'wind_speed_10m_max': data['wind']['speed']
        }
    else:
        raise Exception("Weather API error: Unable to fetch data.")

def result(request):
    if request.method != 'GET':
        return HttpResponse("Invalid request method.", status=405)

    forecast_type = request.GET.get('forecast_type')
    state_name = request.session.get('state')

    if not state_name:
        return render(request, 'error.html', {'message': "User state not found in session."})

    try:
        encoded_state = label_encoder_state.transform([state_name])[0]
    except ValueError:
        return render(request, 'error.html', {'message': "Invalid state name."})

    try:
        if forecast_type == 'today':
            now = datetime.now()
            year, month, day = now.year, now.month, now.day

            try:
                weather_data = get_weather_data_from_api(state_name)
            except Exception as e:
                return render(request, 'error.html', {'message': f"Weather API error: {str(e)}"})

        elif forecast_type == 'future':
            try:
                year = int(request.GET.get('year'))
                month = int(request.GET.get('month'))
                day = int(request.GET.get('day'))
            except (TypeError, ValueError):
                return render(request, 'error.html', {'message': "Invalid date input."})

            # Load weather scaler and LSTM model
            scaler_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'weather_scaler.joblib')
            weather_scaler = load(scaler_path)
            weather_lstm = load_model(os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'lstm_weather_forecast_model.h5'))

            time_steps = 10
            recent_weather = weather_df[['temperature_2m_max', 'temperature_2m_min',
                                         'precipitation_sum', 'wind_speed_10m_max']].values[-time_steps:]

            scaled_recent_weather = weather_scaler.transform(recent_weather)
            lstm_input_weather = scaled_recent_weather.reshape(1, time_steps, -1)

            predicted_scaled_weather = weather_lstm.predict(lstm_input_weather, verbose=0)[0]
            predicted_weather = weather_scaler.inverse_transform([predicted_scaled_weather])[0]

            weather_data = {
                'temperature_2m_max': predicted_weather[0],
                'temperature_2m_min': predicted_weather[1],
                'precipitation_sum': predicted_weather[2],
                'wind_speed_10m_max': predicted_weather[3]
            }

        else:
            return render(request, 'error.html', {'message': "Invalid forecast type."})

        # ====== Predict usage using LSTM + RF ======

        # Load scaler for weather features (again if needed)
        scaler_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'weather_scaler.joblib')
        weather_scaler = load(scaler_path)

        # STEP 1: Scale ONLY weather features
        weather_features = np.array([[weather_data['temperature_2m_max'],
                                      weather_data['temperature_2m_min'],
                                      weather_data['precipitation_sum'],
                                      weather_data['wind_speed_10m_max']]])
        scaled_weather = weather_scaler.transform(weather_features)[0]

        # STEP 2: Add encoded_state to scaled input
        lstm_combined_input = np.append(scaled_weather, encoded_state)

        # STEP 3: Repeat for TIME_STEPS to feed into LSTM
        TIME_STEPS = 10
        lstm_input = np.tile(lstm_combined_input, (TIME_STEPS, 1)).reshape(1, TIME_STEPS, -1)

        # LSTM prediction
        predicted_lstm_usage = lstm_model.predict(lstm_input, verbose=0)[0][0]

        # STEP 4: Random Forest input
        rf_reg_input = np.array([[predicted_lstm_usage,
                          weather_data['temperature_2m_max'],
                          weather_data['temperature_2m_min'],
                          weather_data['precipitation_sum'],
                          weather_data['wind_speed_10m_max']]])

        predicted_final_usage = rf_model.predict(rf_reg_input)[0]
        power_cut_prob = power_cut_classifier.predict_proba(rf_reg_input)[0][1]
        power_cut_pred = power_cut_classifier.predict(rf_reg_input)[0]
        status = "Power Cut Likely" if power_cut_pred == 1 else "Power Supply Stable"

        context = {
                    'forecast_type': forecast_type,
                    'year': year,
                    'month': month,
                    'day': day,
                    'predicted_load': f"{predicted_final_usage:.2f}",
                    'power_cut_probability': f"{power_cut_prob:.2f}",
                    'status': status,
                    'state': state_name,
                    'weather_data': weather_data
                }

        return render(request, 'Results.html', context)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render(request, 'error.html', {'message': f"Prediction error: {str(e)}"})

def household_predict(request):
    if request.method == 'POST':
        try:
            household_id = request.session.get('household_id')
            if not household_id:
                messages.error(request, "Household ID not found. Please log in.")
                return redirect('login_page')

            # Extract inputs from POST data
            load_kw = float(request.POST.get('load_kw'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            voltage = float(request.POST.get('voltage'))

            # Prepare input features
            features = np.array([[load_kw, temperature, humidity, voltage]])

            # Predict power cut
            powercut_prob = power_cut_classifier.predict_proba(features)[0][1]
            powercut_pred = power_cut_classifier.predict(features)[0]

            status = "Power Cut Likely" if powercut_pred == 1 else "Power Supply Stable"

            context = {
                'household_id': household_id,
                'load_kw': load_kw,
                'temperature': temperature,
                'humidity': humidity,
                'voltage': voltage,
                'power_cut_probability': f"{powercut_prob:.2f}",
                'status': status
            }

            return render(request, 'household_results.html', context)

        except Exception as e:
            logger.error(f"Household powercut prediction error: {e}")
            messages.error(request, "Invalid input or prediction error.")
            return redirect('household_powercut')
    else:
        return render(request, 'household_powercut.html')
    
def prediction_options(request):
    return render(request, 'predict_options.html')

load_model_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'load_predictor_lstm_v2.h5')
powercut_model_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'household_power_cut_classifier.joblib')
load_scaler_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'load_scaler_v2.joblib')
powercut_scaler_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'household_powercut_scaler.joblib')
with open(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\best_threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)


# load_model = load_model(load_model_path, compile=True, custom_objects={'mse': MeanSquaredError()})
load_predictor = load_model(load_model_path, compile=True, custom_objects={'mse': MeanSquaredError()})
powercut_model = joblib.load(powercut_model_path)
load_scaler = load(load_scaler_path)
powercut_scaler = joblib.load(powercut_scaler_path)

from datetime import timedelta

household_df = pd.read_csv(r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\household_power_data_2023_2024.csv')
household_df['Date'] = pd.to_datetime(household_df['Date']).dt.date  # Ensure date format

def household_result_view(request):
    forecast_type = request.GET.get('forecast_type')
    state = request.session.get('state', 'Unknown')
    household_id = request.session.get('household_id')

    if not household_id:
        messages.error(request, "Household ID not found. Please log in.")
        return redirect('login_page')

    # Determine target date
    if forecast_type == 'today':
        target_date = datetime.now().date()
    elif forecast_type == 'future':
        year = request.GET.get('year')
        month = request.GET.get('month')
        day = request.GET.get('day')

        if not (year and month and day):
            messages.error(request, "Please enter a valid year, month, and day.")
            return redirect('household_powercut')

        try:
            target_date = datetime(int(year), int(month), int(day)).date()
        except ValueError:
            messages.error(request, "Invalid date format. Please enter a valid date.")
            return redirect('household_powercut')
    else:
        messages.error(request, "Invalid forecast type selected.")
        return redirect('household_powercut')

    try:
        # Load data
        csv_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'household_power_data_2023_2024.csv')
        household_df = pd.read_csv(csv_path)
        household_df['Date'] = pd.to_datetime(household_df['Date']).dt.date

        # Get user name
        try:
            household_obj = Users.objects.get(Household_ID=household_id)
            user_name = household_obj.Name
        except Users.DoesNotExist:
            user_name = "Unknown"

        # Get last 30 days of data before target date
        history = household_df[
            (household_df['Household_ID'] == household_id) & 
            (household_df['Date'] < target_date)
        ].sort_values('Date', ascending=False).head(30)

        if history.shape[0] < 30:
            messages.error(request, "Not enough historical data available for prediction.")
            return redirect('household_powercut')

        history = history.sort_values('Date')

        # Extract and scale features for load model
        load_features = history[['Load_kW', 'Temperature', 'Humidity', 'Voltage']]
        scaled_array = load_scaler.transform(load_features)  # Assuming load_scaler is loaded globally
        scaled_input = np.expand_dims(scaled_array, axis=0)  # shape: (1, 30, 4)

        # Predict scaled load
        predicted_load_scaled = load_predictor.predict(scaled_input)[0][0]

        # Inverse transform to get actual load kW
        dummy_row = np.zeros((1, 4))
        dummy_row[0, 0] = predicted_load_scaled  # load_kW is first feature in scaler
        predicted_load_actual = load_scaler.inverse_transform(dummy_row)[0, 0]

        # Prepare input for power cut prediction
        latest_row = history.iloc[-1]
        powercut_input = np.array([[predicted_load_actual, latest_row['Temperature'], latest_row['Humidity'], latest_row['Voltage']]])
        powercut_input_scaled = powercut_scaler.transform(powercut_input)

        # Predict power cut probability
        powercut_prob = float(powercut_model.predict_proba(powercut_input_scaled)[0][1])
        prediction = "Power Cut Expected" if powercut_prob >= 0.5 else "No Power Cut"

        context = {
            'household_id': household_id,
            'user_name': user_name,
            'state': state,
            'date': target_date,
            'predicted_load': round(predicted_load_actual, 2),
            'power_cut_probability': f"{powercut_prob:.2f}",
            'prediction': prediction,
        }

        return render(request, 'household_results.html', context)

    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        messages.error(request, "Prediction failed due to an internal error.")
        return redirect('household_powercut')

def logout(request):
    # ✅ Log out the user
    logout(request)

    # ✅ Clear all session data
    request.session.flush()

    # ✅ Redirect to Home Page instead of Login Page
    response = redirect('home')  # ✅ Redirecting to home

    # ✅ Remove Remember Me cookies
    response.delete_cookie('logged_in_user')

    print("✅ [LOGOUT] User successfully logged out. Redirecting to Home Page.")

    return response

# def household_result_view(request):
#     forecast_type = request.GET.get('forecast_type')
#     state = request.session.get('state', 'Unknown')
#     household_id = request.session.get('household_id')

#     if not household_id:
#         messages.error(request, "Household ID not found. Please log in.")
#         return redirect('login_page')

#     # Determine target date
#     if forecast_type == 'today':
#         target_date = datetime.now().date()
#     elif forecast_type == 'future':
#         year = request.GET.get('year')
#         month = request.GET.get('month')
#         day = request.GET.get('day')

#         if not (year and month and day):
#             messages.error(request, "Please enter a valid year, month, and day.")
#             return redirect('household_powercut')

#         try:
#             target_date = datetime(int(year), int(month), int(day)).date()
#         except ValueError:
#             messages.error(request, "Invalid date format. Please enter a valid date.")
#             return redirect('household_powercut')
#     else:
#         messages.error(request, "Invalid forecast type selected.")
#         return redirect('household_powercut')

#     try:
#         # Load data
#         csv_path = os.path.join(settings.BASE_DIR, 'pow_app', 'model', 'household_power_data_2023_2024.csv')
#         household_df = pd.read_csv(csv_path)
#         household_df['Date'] = pd.to_datetime(household_df['Date']).dt.date

#         # Get user name
#         try:
#             household_obj = Users.objects.get(Household_ID=household_id)
#             user_name = household_obj.Name
#         except Users.DoesNotExist:
#             user_name = "Unknown"

#         # Get last 30 days of data
#         history = household_df[
#             (household_df['Household_ID'] == household_id) & 
#             (household_df['Date'] < target_date)
#         ].sort_values('Date', ascending=False).head(30)

#         if history.shape[0] < 30:
#             messages.error(request, "Not enough historical data available for prediction.")
#             return redirect('household_powercut')

#         history = history.sort_values('Date')

#         # Extract and scale features for load model
#         load_features = history[['Load_kW', 'Temperature', 'Humidity', 'Voltage']]
#         scaled_array = load_scaler.transform(load_features)  # shape: (30, 4)
#         scaled_input = np.expand_dims(scaled_array, axis=0)  # shape: (1, 30, 4)

#         # Predict load
#         predicted_load = float(load_predictor.predict(scaled_input)[0][0])
        
#         print("Scaled input shape:", scaled_input.shape)
        
#         # Prepare input for power cut prediction
#         latest_row = history.iloc[-1]
#         # print("[DEBUG] Scaler expects:", powercut_scaler.feature_names_in_)
#         # print("[DEBUG] Model expects:", powercut_model.feature_names_in_)
#         # print("[DEBUG] Latest row values:", latest_row[['Temperature', 'Humidity', 'Voltage']])
#         powercut_input = np.array([[predicted_load, latest_row['Temperature'], latest_row['Humidity'], latest_row['Voltage']]])
#         powercut_input_scaled = powercut_scaler.transform(powercut_input)

#         # Predict power cut
#         # powercut_prob = float(powercut_model.predict(powercut_input_scaled)[0][0])
#         powercut_prob = float(powercut_model.predict_proba(powercut_input_scaled)[0][1])
#         prediction = "Power Cut Expected" if powercut_prob >= 0.5 else "No Power Cut"
        
        

#         context = {
#             'household_id': household_id,
#             'user_name': user_name,
#             'state': state,
#             'date': target_date,
#             'predicted_load': round(predicted_load, 2),
#             'power_cut_probability': f"{powercut_prob:.2f}",
#             'prediction': prediction,
#         }

#         return render(request, 'household_results.html', context)

#     except Exception as e:
#         logger.error(f"[ERROR] Prediction failed: {e}")
#         messages.error(request, "Prediction failed due to an internal error.")
#         return redirect('household_powercut')
    
# Create your views here