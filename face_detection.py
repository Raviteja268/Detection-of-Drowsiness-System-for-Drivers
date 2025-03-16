import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Use MediaPipe Face Detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not captured.")
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Perform face detection
        results = face_detection.process(rgb_frame)

        # Convert the frame back to BGR
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw rectangles around detected faces
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Face Detection", frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
