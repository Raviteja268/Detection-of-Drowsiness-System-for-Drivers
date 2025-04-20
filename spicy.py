# from scipy.spatial import distance
# from imutils.video import VideoStream
# import numpy as np
# import cv2
# import os
# import pygame
# import pyttsx3  # For voice alert
# import matplotlib.pyplot as plt
# from collections import deque
# import mediapipe as mp
# import time
# import pandas as pd
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import requests  # For mobile notifications

# # Function to log data to CSV
# def log_data(event, ear_value):
#     with open("drowsiness_log.csv", "a") as log_file:
#         log_file.write(f"{time.time()},{event},{ear_value}\n")

# # Function to log driving history
# def log_driving_history(session_duration, drowsiness_events):
#     with open("driving_history_log.csv", "a") as history_file:
#         history_file.write(f"{time.time()},{session_duration},{drowsiness_events}\n")

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     EAR = (A + B) / (2.0 * C)
#     return EAR

# # Define thresholds for blink counts
# blink_thresholds = {
#     "Green": 20,   # Alert
#     "Yellow": 10,  # Caution
#     "Red": 0       # Drowsy
# }

# # Function to determine fatigue level based on blink count
# def fatigue_level_indicator(blink_count):
#     if blink_count >= blink_thresholds["Green"]:
#         return "Green"  # Alert
#     elif blink_count >= blink_thresholds["Yellow"]:
#         return "Yellow"  # Caution
#     else:
#         return "Red"  # Drowsy

# # Define colors for fatigue levels
# fatigue_colors = {
#     "Green": (0, 255, 0),   # Alert
#     "Yellow": (0, 255, 255),  # Caution
#     "Red": (0, 0, 255)      # Drowsy
# }

# # Constants
# threshold_value = 0.25
# frames = 20
# alarm_path = "D:/drowsiness detection/alarm.wav"  # Updated path
# alert_message = "Wake up! Stay alert! and Drive Carefully"
# screenshot_folder = "drowsy_screenshots"
# blink_count = 0
# ear_values = deque(maxlen=50)  # Stores the last 50 EAR values for real-time graph
# drowsiness_events_count = 0  # Count of drowsiness events

# # Ensure screenshot directory exists
# if not os.path.exists(screenshot_folder):
#     os.makedirs(screenshot_folder)

# # Initialize mediapipe face detection and landmark detection
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh

# # Initialize pygame for alarm
# pygame.mixer.init()
# pygame.mixer.music.load(alarm_path)

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed of speech

# # Initialize Matplotlib for real-time graph
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_ylim(0, 0.5)
# ear_plot, = ax.plot([], [], 'g', label="EAR")
# plt.legend()

# # Start video capture
# cap = cv2.VideoCapture(0)
# time.sleep(2)

# if not cap.isOpened():
#     print("[ERROR] Could not open webcam.")
#     exit()

# flag = 0
# session_start_time = time.time()  # Start timer for driving session

# # Define eye landmark indices
# left_eye_indices = [33, 160, 158, 133, 153, 144]  # Indices for left eye
# right_eye_indices = [362, 385, 387, 263, 373, 380]  # Indices for right eye

# # Create a simple GUI
# def open_file_dialog():
#     global alarm_path
#     alarm_path = filedialog.askopenfilename(title="Select Alarm Sound", filetypes=[("WAV files", "*.wav")])
#     pygame.mixer.music.load(alarm_path)
#     messagebox.showinfo("Info", "Alarm sound updated!")

# def report_false_positive():
#     response = messagebox.askyesno("Feedback", "Was the alert a false positive?")
#     log_data("False Positive" if response else "False Negative", None)

# def send_mobile_notification():
#     # Replace with your mobile notification service API details
#     url = "https://api.your-notification-service.com/send"
#     payload = {"message": "Drowsiness detected! Please take a break."}
#     requests.post(url, json=payload)

# def notify_emergency_contact():
#     # Replace with your emergency contact notification logic
#     url = "https://api.your-notification-service.com/emergency"
#     payload = {"message": "Driver is drowsy. Immediate assistance needed!"}
#     requests.post(url, json=payload)

# root = tk.Tk()
# root.title("Drowsiness Detection System")
# tk.Button(root, text="Change Alarm Sound", command=open_file_dialog).pack(pady=10)
# tk.Button(root, text="Report Feedback", command=report_false_positive).pack(pady=10)
# root.withdraw()  # Hide the main window

# with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
#      mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Frame not captured.")
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame.flags.writeable = False

#         results = face_detection.process(frame)
#         face_mesh_results = face_mesh.process(frame)

#         frame.flags.writeable = True
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # Check if any faces are detected
#         if results.detections:
#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 h, w, _ = frame.shape
#                 x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
#                 cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

#                 if face_mesh_results.multi_face_landmarks:
#                     for face_landmarks in face_mesh_results.multi_face_landmarks:
#                         # Extract eye landmarks using the correct indices
#                         left_eye = np.array([(landmark.x * w, landmark.y * h) for idx in left_eye_indices for landmark in [face_landmarks.landmark[idx]]])
#                         right_eye = np.array([(landmark.x * w, landmark.y * h) for idx in right_eye_indices for landmark in [face_landmarks.landmark[idx]]])

#                         leftEAR = eye_aspect_ratio(left_eye)
#                         rightEAR = eye_aspect_ratio(right_eye)

#                         EAR = (leftEAR + rightEAR) / 2.0
#                         ear_values.append(EAR)  # Update EAR list

#                         # Draw eye contours
#                         cv2.polylines(frame, [left_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
#                         cv2.polylines(frame, [right_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

#                         if EAR < threshold_value:
#                             flag += 1
#                             if flag >= frames:
#                                 cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (50, 50), 
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
#                                 if not pygame.mixer.music.get_busy():
#                                     pygame.mixer.music.play()
#                                     engine.say(alert_message)
#                                     engine.runAndWait()
#                                     log_data("Drowsiness Detected", EAR)  # Log event
#                                     send_mobile_notification()  # Send notification
#                                     drowsiness_events_count += 1  # Increment drowsiness event count

#                                     # Notify emergency contact if drowsiness detected multiple times
#                                     if drowsiness_events_count >= 3:
#                                         notify_emergency_contact()
                                
#                                 # Save Screenshot
#                                 screenshot_path = os.path.join(screenshot_folder, f"drowsy_{time.time()}.jpg")
#                                 cv2.imwrite(screenshot_path, frame)
#                                 print(f"[INFO] Screenshot saved: {screenshot_path}")

#                         else:
#                             if flag > 0:
#                                 blink_count += 1  # Count blinks when EAR recovers
#                             flag = 0
#                             pygame.mixer.music.stop()

#         # Display Blink Count, Session Duration, and Fatigue Level
#         session_duration = time.time() - session_start_time
#         fatigue_level = fatigue_level_indicator(blink_count)

#         # Get the corresponding color for the fatigue level
#         fatigue_color = fatigue_colors[fatigue_level]

#         cv2.putText(frame, f"Blinks: {blink_count}", (10, 420), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.putText(frame, f"Duration: {int(session_duration)}s", (10, 450), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.putText(frame, f"Fatigue Level: {fatigue_level}", (10, 480), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)  # Use the color from the dictionary

#         cv2.imshow("Frame", frame)

#         # Update EAR real-time graph
#         ear_plot.set_xdata(range(len(ear_values)))
#         ear_plot.set_ydata(ear_values)
#         ax.set_xlim(0, len(ear_values))
#         plt.draw()
#         plt.pause(0.001)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord("q"):  # Quit
#             break
#         elif key == ord("u"):  # Increase sensitivity
#             threshold_value += 0.01
#             print(f"Threshold Increased: {threshold_value:.2f}")
#         elif key == ord("d"):  # Decrease sensitivity
#             threshold_value -= 0.01
#             print(f"Threshold Decreased: {threshold_value:.2f}")

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()
# plt.close()
# root.destroy()

# # Log driving history at the end of the session
# session_duration = time.time() - session_start_time
# log_driving_history(session_duration, drowsiness_events_count)


# from flask import Flask, render_template, Response
# import cv2
# import time
# import pygame
# import pyttsx3
# import mediapipe as mp
# import numpy as np
# from scipy.spatial import distance

# app = Flask(__name__)

# # Initialize pygame for audio alerts and TTS engine
# pygame.mixer.init()
# default_alarm_path = "D:/drowsiness detection/alarm.wav"  # Update if needed
# pygame.mixer.music.load(default_alarm_path)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
# alert_message = "Wake up! Stay alert! and Drive Carefully"

# # Detection parameters
# threshold_value = 0.25
# flag = 0
# blink_count = 0
# session_start_time = time.time()

# # Mediapipe initialization
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# left_eye_indices = [33, 160, 158, 133, 153, 144]
# right_eye_indices = [362, 385, 387, 263, 373, 380]

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# def generate_frames():
#     global flag, blink_count, threshold_value
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
#          mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert frame and process with mediapipe
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_rgb.flags.writeable = False
#             results = face_detection.process(frame_rgb)
#             face_mesh_results = face_mesh.process(frame_rgb)
#             frame_rgb.flags.writeable = True
#             frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#             h, w, _ = frame_bgr.shape

#             if results.detections:
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
#                     width, height = int(bboxC.width * w), int(bboxC.height * h)
#                     cv2.rectangle(frame_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
#                     if face_mesh_results.multi_face_landmarks:
#                         for face_landmarks in face_mesh_results.multi_face_landmarks:
#                             # Extract eye landmarks
#                             left_eye = np.array([(face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h) 
#                                                  for idx in left_eye_indices])
#                             right_eye = np.array([(face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h) 
#                                                  for idx in right_eye_indices])
#                             leftEAR = eye_aspect_ratio(left_eye)
#                             rightEAR = eye_aspect_ratio(right_eye)
#                             EAR = (leftEAR + rightEAR) / 2.0

#                             # Draw eye contours
#                             cv2.polylines(frame_bgr, [left_eye.astype(np.int32)], True, (0, 255, 0), 1)
#                             cv2.polylines(frame_bgr, [right_eye.astype(np.int32)], True, (0, 255, 0), 1)

#                             # Drowsiness detection logic
#                             if EAR < threshold_value:
#                                 flag += 1
#                                 if flag >= 20:
#                                     cv2.putText(frame_bgr, "ALERT! DROWSINESS DETECTED!", (50, 50), 
#                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                                     if not pygame.mixer.music.get_busy():
#                                         pygame.mixer.music.play()
#                                         engine.say(alert_message)
#                                         engine.runAndWait()
#                                         blink_count += 1
#                             else:
#                                 if flag > 0:
#                                     blink_count += 1
#                                 flag = 0

#             # Overlay session stats
#             session_duration = int(time.time() - session_start_time)
#             cv2.putText(frame_bgr, f"Blinks: {blink_count}", (10, 420), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#             cv2.putText(frame_bgr, f"Duration: {session_duration}s", (10, 450), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame_bgr)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     cap.release()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import time
import pygame
import pyttsx3
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from collections import deque
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = "mysecretkey"  # Set a secret key for session management

# Connect to MongoDB
client = MongoClient("mongodb+srv://21bq1a05o2:Venky630335@cluster0.7xwmt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["drowsiness"]
users_collection = db["users"]

# Initialize pygame for audio alerts and TTS engine
pygame.mixer.init()
default_alarm_path = "C:/Users/ravit/Downloads/DriverDrowsinessDetection-main/DriverDrowsinessDetection-main/alarm.wav"  # Update if needed
pygame.mixer.music.load(default_alarm_path)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
alert_message = "Wake up! Stay alert! and Drive Carefully"

# Detection parameters
threshold_value = 0.25
flag = 0
blink_count = 0
session_start_time = time.time()

# For storing EAR values for graphing (last 50 values)
ear_values = deque(maxlen=50)

# Global flag to control detection
detection_running = True

# Thresholds for fatigue level (blink counts)
blink_thresholds = {
    "Green": 20,   # Alert
    "Yellow": 10,  # Caution
    "Red": 0       # Drowsy
}

# Fatigue colors (web hex values)
fatigue_colors = {
    "Green": "#00FF00",
    "Yellow": "#FFFF00",
    "Red": "#FF0000"
}

def fatigue_level_indicator(blink_count):
    if blink_count >= blink_thresholds["Green"]:
        return "Green"  # Alert
    elif blink_count >= blink_thresholds["Yellow"]:
        return "Yellow"  # Caution
    else:
        return "Red"  # Drowsy

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mediapipe initialization
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

def generate_frames():
    global flag, blink_count, threshold_value, detection_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = face_detection.process(frame_rgb)
            face_mesh_results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w, _ = frame_bgr.shape

            if detection_running:
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
                        width, height = int(bboxC.width * w), int(bboxC.height * h)
                        cv2.rectangle(frame_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        if face_mesh_results.multi_face_landmarks:
                            for face_landmarks in face_mesh_results.multi_face_landmarks:
                                # Extract eye landmarks
                                left_eye = np.array([
                                    (face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h)
                                    for idx in left_eye_indices
                                ])
                                right_eye = np.array([
                                    (face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h)
                                    for idx in right_eye_indices
                                ])
                                leftEAR = eye_aspect_ratio(left_eye)
                                rightEAR = eye_aspect_ratio(right_eye)
                                EAR = (leftEAR + rightEAR) / 2.0
                                ear_values.append(EAR)  # Save for real-time graph

                                # Draw eye contours
                                cv2.polylines(frame_bgr, [left_eye.astype(np.int32)], True, (0, 255, 0), 1)
                                cv2.polylines(frame_bgr, [right_eye.astype(np.int32)], True, (0, 255, 0), 1)

                                # Drowsiness detection logic
                                if EAR < threshold_value:
                                    flag += 1
                                    if flag >= 20:
                                        cv2.putText(frame_bgr, "ALERT! DROWSINESS DETECTED!", (50, 50), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        if not pygame.mixer.music.get_busy():
                                            pygame.mixer.music.play()
                                            engine.say(alert_message)
                                            engine.runAndWait()
                                            blink_count += 1
                                else:
                                    if flag > 0:
                                        blink_count += 1
                                    flag = 0
            else:
                cv2.putText(frame_bgr, "DETECTION STOPPED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Overlay session stats on the frame
            session_duration = int(time.time() - session_start_time)
            cv2.putText(frame_bgr, f"Blinks: {blink_count}", (10, 420), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Duration: {session_duration}s", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# ---------- User Authentication Routes ----------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get registration form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        # Check if user already exists
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return render_template('register.html', error="Username already exists!")
        # For production, hash the password before storing.
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password
        })
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = users_collection.find_one({"username": username, "password": password})
        if user:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials!")
    return render_template('login.html')

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------- Detection and Streaming Routes ----------

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to provide EAR values as JSON for the frontend chart
@app.route('/ear_data')
def ear_data():
    return jsonify(ear_values=list(ear_values))

# Endpoint to provide blink count, session duration, and fatigue level (with color)
@app.route('/stats')
def stats():
    session_duration = int(time.time() - session_start_time)
    fatigue_level = fatigue_level_indicator(blink_count)
    return jsonify({
        'blink_count': blink_count,
        'duration': session_duration,
        'fatigue_level': fatigue_level,
        'color': fatigue_colors[fatigue_level]
    })

# Endpoint to stop detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    detection_running = False
    return jsonify({'status': 'detection stopped'})

# Endpoint to start detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running
    detection_running = True
    return jsonify({'status': 'detection started'})

if __name__ == '__main__':
    app.run(debug=True)
