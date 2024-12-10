from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import base64
import pyautogui
import time

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Gesture Variables
last_action_time = 0
gesture_hold_time = 0.8  # Minimum time between gestures

# Helper function for gesture recognition
def detect_gesture(hand_landmarks):
    global last_action_time
    screen_width, screen_height = pyautogui.size()

    # Extract landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Cursor movement
    cursor_x = int(index_tip.x * screen_width)
    cursor_y = int(index_tip.y * screen_height)
    pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

    # Click gesture (index and thumb close)
    current_time = time.time()
    if abs(index_tip.x - thumb_tip.x) < 0.03 and abs(index_tip.y - thumb_tip.y) < 0.03:
        if current_time - last_action_time > gesture_hold_time:
            pyautogui.click()
            last_action_time = current_time
    return "Gesture Detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])

    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(frame_rgb)
    response = {"gesture": "No gesture detected"}
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            response["gesture"] = detect_gesture(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
