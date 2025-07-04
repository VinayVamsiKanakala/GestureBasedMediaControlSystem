import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import time

# Load the trained model
MODEL_FILE = 'gesture_model.pkl'
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)

# Ask user which media player to control
def select_media_player():
    print("\nSelect a media player to control:")
    print("1. YouTube")
    print("2. Spotify")
    print("3. VLC Player")
    print("4. MX Player")
    choice = input("Enter your choice (1/2/3/4): ").strip()
    return {
        '1': 'youtube',
        '2': 'spotify',
        '3': 'vlc',
        '4': 'mx'
    }.get(choice, 'youtube')  # Default to YouTube

media_player = select_media_player()
print(f"\n🎯 Controlling: {media_player.capitalize()}")

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.8,
                      min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Cooldown to prevent rapid key presses
last_action_time = time.time()
cooldown = 1.0  # 1-second cooldown

gesture_names = {
    1: "One Finger (Next)",
    2: "Two Fingers (Previous)",
    3: "Three Fingers (Vol Up)",
    4: "Four Fingers (Vol Down)",
    5: "Five Fingers (Play/Pause)"
}

def calculate_features(landmarks):
    features = []
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized = [(x - wrist_x, y - wrist_y, z - wrist_z)
                  for x, y, z in zip(landmarks[::3], landmarks[1::3], landmarks[2::3])]
    features.extend([v for triple in normalized for v in triple])

    for i in [8, 12, 16, 20]:
        tip = np.array([landmarks[i*3], landmarks[i*3+1], landmarks[i*3+2]])
        wrist = np.array([wrist_x, wrist_y, wrist_z])
        features.append(np.linalg.norm(tip - wrist))

    for (i, j) in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip = np.array([landmarks[i*3], landmarks[i*3+1], landmarks[i*3+2]])
        pip = np.array([landmarks[j*3], landmarks[j*3+1], landmarks[j*3+2]])
        features.append(np.linalg.norm(tip - pip))

    return np.array(features)

def control_media(gesture):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time < cooldown:
        return

    print(f"🎯 Detected: {gesture_names[gesture]}")

    if gesture == 5:  # Play/Pause
        pyautogui.press('space')

    elif gesture == 1:  # Next
        if media_player == 'youtube':
            pyautogui.hotkey('shift', 'n')
        elif media_player == 'spotify':
            pyautogui.hotkey('ctrl', 'right')
        elif media_player == 'vlc':
            pyautogui.press('n')
        elif media_player == 'mx':
            pyautogui.press('right')  # example

    elif gesture == 2:  # Previous
        if media_player == 'youtube':
            pyautogui.hotkey('shift', 'p')
        elif media_player == 'spotify':
            pyautogui.hotkey('ctrl', 'left')
        elif media_player == 'vlc':
            pyautogui.press('p')
        elif media_player == 'mx':
            pyautogui.press('left')  # example

    elif gesture == 3:
        pyautogui.press('volumeup')

    elif gesture == 4:
        pyautogui.press('volumedown')

    last_action_time = current_time

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            if landmarks:
                try:
                    features = calculate_features(landmarks).reshape(1, -1)
                    gesture = model.predict(features)[0]
                    cv2.putText(frame, f"Gesture: {gesture_names[gesture]}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    control_media(gesture)
                except Exception as e:
                    print(f"Prediction error: {e}")

    cv2.imshow('Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
