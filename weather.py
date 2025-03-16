import requests  # For weather data
import time
from PIL import Image
import numpy as np
import cv2

# Function to get weather data
def get_weather(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        
        # Extract weather data
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        pressure = data['main']['pressure']
        wind_speed = data['wind']['speed']
        sunrise = time.strftime('%H:%M', time.gmtime(data['sys']['sunrise'] + data['timezone']))
        sunset = time.strftime('%H:%M', time.gmtime(data['sys']['sunset'] + data['timezone']))
        aqi = get_air_quality(data['coord']['lat'], data['coord']['lon'], api_key)  # AQI based on coordinates
        
        # Check for weather alerts
        alerts = data.get('alerts', [])
        severe_weather_alert = alerts[0]['description'] if alerts else None
        
        return weather_description, temperature, humidity, pressure, wind_speed, sunrise, sunset, aqi, severe_weather_alert
    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"[ERROR] An error occurred: {err}")
    
    return None, None, None, None, None, None, None, None, None

# Function to get air quality index
def get_air_quality(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        aqi = data['list'][0]['main']['aqi']
        return aqi
    except Exception as err:
        print(f"[ERROR] An error occurred while fetching AQI: {err}")
        return None

# Function to load weather icon based on condition
def load_weather_icon(condition):
    if 'clear' in condition:
        icon_path = r"C:\Users\ravit\Downloads\sun.png"
    elif 'cloud' in condition:
        icon_path = r"C:\Users\ravit\Downloads\hail.png"
    elif 'rain' in condition:
        icon_path = r"C:\Users\ravit\Downloads\rain.png"
    else:
        icon_path = r"C:\Users\ravit\Downloads\default.png"  # Default icon

    icon = Image.open(icon_path).resize((50, 50))  # Resize icon
    return np.array(icon)

# Weather API setup
api_key = "b8a7f19a29a407272c311f1f8c86a270"  # Replace with your actual OpenWeatherMap API key
city = "Greater Noida"  # Change city name if needed
weather_description, temperature, humidity, pressure, wind_speed, sunrise, sunset, aqi, severe_weather_alert = get_weather(api_key, city)

# Print weather data for debugging
print(f"Weather: {weather_description}, Temp: {temperature}, Humidity: {humidity}, Pressure: {pressure}, Wind Speed: {wind_speed}, AQI: {aqi}, Sunrise: {sunrise}, Sunset: {sunset}")

# Create a window for displaying weather conditions
weather_window_name = "Weather Conditions"
weather_window = np.zeros((400, 400, 3), dtype=np.uint8)  # Blank window
weather_window[:] = (173, 216, 230)  # Light blue background color

# Prepare weather information for display
weather_window[:] = (173, 216, 230)  # Clear the weather window with light blue color
if weather_description and temperature is not None:
    # Load and display weather icon
    weather_icon = load_weather_icon(weather_description)
    weather_icon_bgr = cv2.cvtColor(weather_icon, cv2.COLOR_RGBA2BGR)
    weather_window[10:60, 10:60] = weather_icon_bgr  # Icon position

    # Display weather information
    cv2.putText(weather_window, f"Location: {city}", (70, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Weather: {weather_description.capitalize()}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Temp: {temperature} °C", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Humidity: {humidity}%", (10, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Pressure: {pressure} hPa", (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Wind Speed: {wind_speed} m/s", (10, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"AQI: {aqi}", (10, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Sunrise: {sunrise}", (10, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Sunset: {sunset}", (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Get current time and date
    current_time = time.strftime('%H:%M:%S', time.localtime())
    current_date = time.strftime('%Y-%m-%d', time.localtime())
    cv2.putText(weather_window, f"Current Date: {current_date}", (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(weather_window, f"Current Time: {current_time}", (10, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display severe weather alert if available
    if severe_weather_alert:
        cv2.putText(weather_window, f"Weather Alert: {severe_weather_alert}", (10, 410), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Show the weather window
cv2.imshow(weather_window_name, weather_window)
cv2.waitKey(0)
cv2.destroyAllWindows()
