from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os
from tensorflow.keras.models import load_model

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder="templates")

# -------------------------
# Load emotion model
# -------------------------
MODEL_PATH = os.path.join("models", "emotion_cnn_model.keras")
model = load_model(MODEL_PATH)

# -------------------------
# Load Haar Cascade
# -------------------------
FACE_CASCADE_PATH = os.path.join(
    "models", "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

print("Cascade loaded:", not face_cascade.empty())

# -------------------------
# Emotion labels
# -------------------------
emotions = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# -------------------------
# Home route (UI)
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Prediction route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        # Decode base64 image
        img_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Resize frame for better detection (CRITICAL)
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Face detection (relaxed params)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(40, 40)
        )

        print("Faces detected:", len(faces))

        # No face found
        if len(faces) == 0:
            return jsonify({
                "emotion": "undefined",
                "confidence": 0.0
            })

        # Take first detected face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        # Preprocess for model
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict
        preds = model.predict(face, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(preds[0][idx])

        return jsonify({
            "emotion": emotions[idx],
            "confidence": confidence
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "emotion": "error",
            "confidence": 0.0
        })

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

