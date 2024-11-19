import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from sklearn.cluster import KMeans

app = Flask(__name__)

# Initialize variables
reference_image = None
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
tracking_box = None
tracker = cv2.TrackerCSRT_create()  # Robust tracker

# Parameters
matching_threshold = 40  # Minimum matches
tracking_enabled = False

def extract_features(image):
    """Extract ORB features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Upload reference image."""
    global reference_image, ref_keypoints, ref_descriptors

    file = request.files['image']
    if file:
        img = np.frombuffer(file.read(), np.uint8)
        reference_image = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Extract features from the uploaded image
        ref_keypoints, ref_descriptors = extract_features(reference_image)
        return "Image uploaded and processed successfully!"
    return "Failed to upload image!", 400
    
app=app

def detect_object(frame):
    """Detect object in the frame."""
    global reference_image, ref_keypoints, ref_descriptors, tracking_box, tracker, tracking_enabled

    if reference_image is None:
        return frame

    # Convert frame to grayscale and extract features
    frame_keypoints, frame_descriptors = extract_features(frame)

    if ref_descriptors is not None and frame_descriptors is not None:
        # Match features
        matches = bf.match(ref_descriptors, frame_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter matches using K-Means clustering
        if len(matches) >= matching_threshold:
            src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Compute homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = reference_image.shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Initialize tracker for detected object
                tracking_box = cv2.boundingRect(dst)
                tracker.init(frame, tracking_box)
                tracking_enabled = True

    # If tracking is enabled, update the tracker
    if tracking_enabled:
        success, tracking_box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, tracking_box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return frame

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Capture video frames and process."""
    global tracking_box, tracking_enabled

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = detect_object(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)