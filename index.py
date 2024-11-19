from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import threading

app = Flask(__name__)

# Global variables
uploaded_image = None
detector = cv2.ORB_create()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
kmeans = None  # Placeholder for K-Means clustering
live_frame = None
frame_lock = threading.Lock()

# Function to extract ORB features from an image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

# Process uploaded image
def process_uploaded_image(image):
    global kmeans
    _, descriptors = extract_features(image)
    if descriptors is not None:
        # Cluster features with K-Means
        kmeans = cv2.kmeans(
            descriptors.astype(np.float32), 
            2, 
            None, 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
            attempts=10, 
            flags=cv2.KMEANS_RANDOM_CENTERS
        )[2]

# Match and detect objects in live feed
def detect_object(frame):
    global uploaded_image, kmeans
    if uploaded_image is None or kmeans is None:
        return frame  # No uploaded image to compare

    keypoints, descriptors = extract_features(frame)
    if descriptors is None:
        return frame

    # Match features between live frame and uploaded image
    matches = bf_matcher.match(kmeans, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw bounding box if matches found
    if len(matches) > 10:  # Threshold for good matches
        points = [keypoints[m.trainIdx].pt for m in matches]
        points = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Background thread for live feed processing
def video_stream():
    global live_frame, frame_lock
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with frame_lock:
            live_frame = detect_object(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_image
    file = request.files['image']
    if file:
        # Read the uploaded image
        uploaded_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        process_uploaded_image(uploaded_image)
        return "Image uploaded and processed successfully!"
    return "Failed to upload image!"

# Stream live video feed to the browser
@app.route('/video_feed')
def video_feed():
    def generate():
        global live_frame, frame_lock
        while True:
            with frame_lock:
                if live_frame is not None:
                    _, buffer = cv2.imencode('.jpg', live_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the background thread for video processing
video_thread = threading.Thread(target=video_stream)
video_thread.daemon = True
video_thread.start()

if __name__ == '__main__':
    app.run(debug=True)