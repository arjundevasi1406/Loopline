from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load YOLO model
model = YOLO('model/yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    image_data = data['image'].split(",")[1]  # Extract base64 data
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = np.array(image)

    # Perform object detection
    results = model.predict(image)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return jsonify(detections=detections)

if __name__ == '_main_':
    app.run(debug=True)