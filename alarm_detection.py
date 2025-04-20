from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os

# Hardcoded paths for shape predictor and alarm sound
predictor_path = r"C:\Users\ravit\Downloads\shape_predictor_68_face_landmarks.dat"
alarm_path = r"C:\Users\ravit\Downloads\DriverDrowsinessDetection-main\DriverDrowsinessDetection-main\alarm.wav"

def sound_alarm(path):
    # Play an alarm sound whenever the function is invoked
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear

# Construct the argument parse and parse the arguments one by one
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-file", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default=alarm_path,
    help="path alarm .WAV file")  # Default alarm path is the one you specified
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

# Define two constants: one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30

# Initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# Check if the shape predictor file exists
if not os.path.isfile(args["shape_file"]):
    print("[ERROR] The shape predictor file does not exist. Please provide a valid path.")
    exit()

# Check if the alarm sound file exists, if provided
if args["alarm"] != "" and not os.path.isfile(args["alarm"]):
    print("[ERROR] The alarm sound file does not exist. Please provide a valid path.")
    exit()

# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor(args["shape_file"])
except Exception as e:
    print(f"[ERROR] Unable to load shape predictor: {e}")
    exit()

# Grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Check if video stream started successfully
if vs.stream is None:
    print("[ERROR] Unable to access webcam. Please check your webcam settings.")
    exit()

# Loop over frames from the video stream
while True:
    try:
        # Grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

                # Draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    except Exception as e:
        print(f"[ERROR] An error occurred while processing the video stream: {e}")
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
