import cv2
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Global variables for ORB detector, feature matcher, and clicked object
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
clicked_image = None  # The image of the clicked object
keypoints1, descriptors1 = None, None  # Features of the clicked object


def match_and_track(frame):
    """Match the clicked object features and draw a single accurate box."""
    global clicked_image, keypoints1, descriptors1

    if clicked_image is None or descriptors1 is None:
        return frame  # No object clicked yet

    # Detect and compute ORB features in the current frame
    keypoints2, descriptors2 = orb.detectAndCompute(frame, None)

    if descriptors2 is not None:
        # Match features using BFMatcher
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter matches based on distance threshold
        good_matches = [m for m in matches if m.distance < 50]

        # Proceed only if sufficient good matches are found
        if len(good_matches) > 10:  # Minimum good matches for detection
            matched_points = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Cluster matched points using K-Means
            _, labels, centers = cv2.kmeans(
                matched_points.astype(np.float32),
                K=1,
                bestLabels=None,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                attempts=10,
                flags=cv2.KMEANS_RANDOM_CENTERS,
            )
            x, y = centers[0]
            w, h = 150, 150  # Fixed bounding box size
            bbox = (int(x - w / 2), int(y - h / 2), w, h)

            # Draw a single bounding box around the detected object
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Same Product Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/click', methods=['POST'])
def click_image():
    """Handle the image clicked by the user."""
    global clicked_image, keypoints1, descriptors1
    if 'image' in request.files:
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        clicked_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        keypoints1, descriptors1 = orb.detectAndCompute(clicked_image, None)
        return "Image uploaded successfully"
    return "No image uploaded"


def generate_frames():
    """Generate video frames for real-time detection."""
    global clicked_image

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Match and track objects in the frame
        frame = match_and_track(frame)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Provide video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
