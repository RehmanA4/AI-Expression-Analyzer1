import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OLD_MODEL = os.path.join(BASE_DIR, "..", "models", "emotion_cnn_model.h5")
NEW_MODEL = os.path.join(BASE_DIR, "..", "models", "emotion_cnn_model_fixed.keras")

print("Loading old model...")
model = load_model(OLD_MODEL, compile=False)

print("Saving model in new Keras format...")
model.save(NEW_MODEL)

print("✅ Model successfully converted!")
print("New model path:", NEW_MODEL)


