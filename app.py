from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Screen Dimensions
screen_width, screen_height = pyautogui.size()

# Gesture Variables
last_action_time = 0  # Timer for gesture breaks
gesture_hold_time = 0.8  # Minimum time between gestures

# Helper Functions
def distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks."""
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

def detect_gesture(hand_landmarks, hand_label):
    global last_action_time

    # Get relevant landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Current time for gesture breaks
    current_time = time.time()

    # Cursor Movement (Middle Finger)
    cursor_x = int(middle_tip.x * screen_width)
    cursor_y = int(middle_tip.y * screen_height)
    pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

    # Click Gesture (Index and Thumb Tip Close)
    if distance(index_tip, thumb_tip) < 0.09:
        if current_time - last_action_time > gesture_hold_time:
            pyautogui.click()
            last_action_time = current_time

    # Scroll Gesture (Thumb Up or Down)
    if thumb_tip.y < index_tip.y and thumb_tip.y < pinky_tip.y:  # Thumb Up
        if current_time - last_action_time > gesture_hold_time:
            pyautogui.scroll(60)
            last_action_time = current_time
    elif thumb_tip.y > index_tip.y and thumb_tip.y > pinky_tip.y:  # Thumb Down
        if current_time - last_action_time > gesture_hold_time:
            pyautogui.scroll(-60)
            last_action_time = current_time

# Video Capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip and Convert Frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detect_gesture(hand_landmarks, "Right")  # Assuming single-hand input

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main Entry
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
