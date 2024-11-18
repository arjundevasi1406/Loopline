const video = document.getElementById("camera-feed");
const canvas = document.getElementById("captured-image");
const captureBtn = document.getElementById("capture-btn");
const context = canvas.getContext("2d");

// Access the camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((error) => {
    console.error("Error accessing camera:", error);
  });

// Capture image
captureBtn.addEventListener("click", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL("image/jpeg");
  sendImageToServer(imageData);
});

// Send image to server
function sendImageToServer(imageData) {
  fetch("/capture", {
    method: "POST",
    body: JSON.stringify({ image: imageData }),
    headers: { "Content-Type": "application/json" },
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Server response:", data);
    })
    .catch((error) => console.error("Error:", error));
}