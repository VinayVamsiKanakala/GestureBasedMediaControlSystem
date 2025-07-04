import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the number of data points to collect per gesture
DATA_POINTS_PER_GESTURE = 100
OUTPUT_CSV_FILE = 'gesture_data.csv'

# Ensure the CSV file exists or create it with headers
if not os.path.exists(OUTPUT_CSV_FILE):
    with open(OUTPUT_CSV_FILE, 'w') as f:
        f.write('label,' + ','.join([f'x{i},y{i},z{i}' for i in range(21)]) + '\n')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

gesture_label = None
data_count = 0

print("Instructions:")
print("Press '1' and show one finger.")
print("Press '2' and show two fingers.")
print("Press '3' and show three fingers.")
print("Press '4' and show four fingers.")
print("Press '5' and show five fingers (play/pause).")
print("Press 'q' to quit data collection.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Flip the frame horizontally to remove mirroring
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if gesture_label is not None and data_count < DATA_POINTS_PER_GESTURE:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmark_str = ','.join(map(str, landmarks))

                with open(OUTPUT_CSV_FILE, 'a') as f:
                    f.write(f'{gesture_label},{landmark_str}\n')
                data_count += 1
                print(f"Collected {data_count}/{DATA_POINTS_PER_GESTURE} data points for gesture {gesture_label}")

                if data_count == DATA_POINTS_PER_GESTURE:
                    print(f"Finished collecting data for gesture {gesture_label}")
                    gesture_label = None
                    data_count = 0

    cv2.putText(frame, f"Collecting for: {gesture_label if gesture_label else 'None'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        gesture_label = 1
        data_count = 0
    elif key == ord('2'):
        gesture_label = 2
        data_count = 0
    elif key == ord('3'):
        gesture_label = 3
        data_count = 0
    elif key == ord('4'):
        gesture_label = 4
        data_count = 0
    elif key == ord('5'):
        gesture_label = 5
        data_count = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
