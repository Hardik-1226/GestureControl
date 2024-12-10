from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import base64

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    if frame is None:
        return jsonify({"error": "Frame not decoded correctly"})

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        return jsonify({"gesture": "Hand detected"})
    else:
        return jsonify({"gesture": "No hand detected"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
