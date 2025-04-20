from scipy.spatial import distance
import numpy as np
import cv2
import os
import pygame
import pyttsx3  # For voice alert
import matplotlib.pyplot as plt
from collections import deque
import mediapipe as mp
import time
import requests  # For weather data
from PIL import Image

# Function to log data to CSV
def log_data(event, ear_value):
    with open("drowsiness_log.csv", "a") as log_file:
        log_file.write(f"{time.time()},{event},{ear_value}\n")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

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

# Constants
threshold_value = 0.18  # EAR threshold for detecting closed eyes
alarm_path = "C:/Users/ravit/Downloads/DriverDrowsinessDetection-main/DriverDrowsinessDetection-main/alarm.wav"  # Updated alarm path
alert_message = "Wake up! Stay alert! and Drive Carefully"
screenshot_folder = "drowsy_screenshots"
ear_values = deque(maxlen=50)  # Stores the last 50 EAR values for real-time graph
blink_times = []  # List to store timestamps of blinks

# Ensure screenshot directory exists
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# Initialize Mediapipe face detection and landmark detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.1)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3)

# Initialize pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load(alarm_path)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Start video capture
cap = cv2.VideoCapture(0)
time.sleep(2)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Initialize fatigue level
fatigue_level = "Green"  # Initial fatigue level

# Define eye landmark indices
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Indices for left eye
right_eye_indices = [362, 385, 387, 263, 373, 380]  # Indices for right eye

# Initialize Matplotlib for real-time graph
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
ax.set_ylim(0, 0.5)  # Set y-axis limits for EAR
ear_plot, = ax.plot([], [], 'g', label="EAR")
ax.set_xlim(0, 50)  # Set x-axis limit for the number of points
plt.xlabel("Frames")
plt.ylabel("EAR")
plt.title("Eye Aspect Ratio Over Time")
plt.legend()

# Weather API setup
api_key = "b8a7f19a29a407272c311f1f8c86a270"  # Replace with your actual OpenWeatherMap API key
city = "Greater Noida"  # Change city name if needed
weather_description, temperature, humidity, pressure, wind_speed, sunrise, sunset, aqi, severe_weather_alert = get_weather(api_key, city)

# Print weather data for debugging
print(f"Weather: {weather_description}, Temp: {temperature}, Humidity: {humidity}, Pressure: {pressure}, Wind Speed: {wind_speed}, AQI: {aqi}, Sunrise: {sunrise}, Sunset: {sunset}")

# Timer initialization
start_time = time.time()  # Record the start time

# Create a window for displaying weather conditions
weather_window_name = "Weather Conditions"
weather_window = np.zeros((400, 400, 3), dtype=np.uint8)  # Blank window
weather_window[:] = (173, 216, 230)  # Light blue background color

# Main loop for video processing
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not captured.")
            break

        # Convert frame to RGB and process
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        results = mp_face_detection.process(frame)
        face_mesh_results = mp_face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Check if any faces are detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Process face landmarks if detected
                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        # Extract eye landmarks
                        left_eye = np.array([(landmark.x * w, landmark.y * h) for idx in left_eye_indices for landmark in [face_landmarks.landmark[idx]]])
                        right_eye = np.array([(landmark.x * w, landmark.y * h) for idx in right_eye_indices for landmark in [face_landmarks.landmark[idx]]])

                        # Calculate EAR for both eyes
                        leftEAR = eye_aspect_ratio(left_eye)
                        rightEAR = eye_aspect_ratio(right_eye)

                        # Calculate combined EAR
                        EAR = (leftEAR + rightEAR) / 2.0
                        ear_values.append(EAR)  # Update EAR list

                        # Draw eye contours
                        cv2.polylines(frame, [left_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                        cv2.polylines(frame, [right_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

                        # Check EAR for blinks
                        if EAR < threshold_value:
                            # Record the current time of the blink
                            blink_times.append(time.time())

                        # Remove timestamps older than 15 seconds
                        blink_times = [t for t in blink_times if time.time() - t <= 15]

                        # Check if there are 10 blinks in the last 15 seconds
                        if len(blink_times) >= 10:
                            fatigue_level = "Red"  # Trigger alarm
                        else:
                            fatigue_level = "Green"

                        # Trigger alarm if fatigue level is Red
                        if fatigue_level == "Red":
                            cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            if not pygame.mixer.music.get_busy():
                                pygame.mixer.music.play()
                                engine.say(alert_message)
                                engine.runAndWait()
                                log_data("Drowsiness Detected", EAR)

                        # Display Blink Count and Fatigue Level
                        cv2.putText(frame, f"Blinks in last 15s: {len(blink_times)}", (10, 420), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, f"Fatigue Level: {fatigue_level}", (10, 450), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if fatigue_level == "Green" else (0, 0, 255))

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)

        # Display the elapsed time on the frame
        cv2.putText(frame, f"Active Time: {elapsed_minutes:02}:{elapsed_seconds:02}", (10, 480), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the processed frame
        cv2.imshow("Frame", frame)

        # Update EAR graph
        ear_plot.set_xdata(range(len(ear_values)))
        ear_plot.set_ydata(ear_values)
        ax.set_xlim(0, len(ear_values))
        plt.draw()
        plt.pause(0.001)

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

            # Get current time
            current_time = time.strftime('%H:%M:%S', time.localtime())
            cv2.putText(weather_window, f"Current Time: {current_time}", (10, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display severe weather alert if available
            if severe_weather_alert:
                cv2.putText(frame, f"Weather Alert: {severe_weather_alert}", (10, 500), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Additional temperature checks
            if temperature > 35:  # High temperature alert
                cv2.putText(frame, "Alert: High Temperature!", (10, 520), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif temperature < 0:  # Low temperature alert
                cv2.putText(frame, "Alert: Low Temperature!", (10, 540), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Air Quality Alert
            if aqi == 4 or aqi == 5:  # Unhealthy or Very Unhealthy
                cv2.putText(frame, "Alert: Unhealthy Air Quality!", (10, 560), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print("[INFO] Weather data not available.")

        # Show the weather window
        cv2.imshow(weather_window_name, weather_window)

        # Check for user input to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break

except Exception as e:
    print(f"[ERROR] An exception occurred: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    plt.close()
    print("[INFO] Resources released and program exited safely.")