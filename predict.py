import os
import base64
from io import BytesIO

import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = os.path.join("models", "mnist_cnn.h5")

# Load model once at startup
model = load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess uploaded image or canvas drawing to match MNIST format:
    - Convert to grayscale
    - Resize to 28x28
    - If background is light and digit is dark, invert colors
    - Normalize to [0,1]
    - Add batch + channel dimension
    """
    # Grayscale & resize
    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img).astype("float32")

    # If background is light (high mean), invert to match MNIST (white digit on dark bg)
    if img_array.mean() > 127:
        img_array = 255.0 - img_array

    img_array /= 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_digit(img: Image.Image):
    """Run the CNN model on a PIL image and return (digit, probabilities list)."""
    arr = preprocess_image(img)
    preds = model.predict(arr)
    digit = int(np.argmax(preds))
    probs = preds[0].tolist()
    return digit, probs


@app.route("/", methods=["GET"])
def home():
    """Render main UI page."""
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():
    """Handle normal file upload from the UI."""
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    try:
        img = Image.open(file.stream)
        digit, probs = predict_digit(img)
        best_conf = max(probs) if probs else 0.0

        return render_template(
            "index.html",
            prediction=digit,
            probabilities=probs,
            best_confidence=best_conf
        )
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """Handle mouse-drawn canvas image sent as base64 data URL."""
    data_url = request.form.get("canvas_data")
    if not data_url:
        return render_template("index.html", prediction="No drawing received")

    try:
        # data_url is like "data:image/png;base64,AAAA..."
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))

        digit, probs = predict_digit(img)
        best_conf = max(probs) if probs else 0.0

        return render_template(
            "index.html",
            prediction=digit,
            probabilities=probs,
            best_confidence=best_conf
        )
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check for local + docker + cloud."""
    return {"status": "ok", "model_loaded": True}, 200


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API endpoint that accepts an image file and returns:
    - prediction (digit 0–9)
    - confidence (max probability)
    - probabilities (list of 10 values)
    """
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "Empty filename"}, 400

    try:
        img = Image.open(file.stream)
        digit, probs = predict_digit(img)
        best_conf = max(probs) if probs else 0.0

        return {
            "prediction": digit,
            "confidence": best_conf,
            "probabilities": probs
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
