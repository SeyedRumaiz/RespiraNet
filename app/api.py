from flask import Flask, request, jsonify
from PIL import Image
import io   # For raw byte streams

from model import DenseNetModel121

app = Flask(__name__)

# Load model once when server starts
model = DenseNetModel121("../models/best_densenet.weights.h5")

# Front-end sends requests here
@app.route("/predict", methods=["POST"])
def predict():

    # Check if file uploaded to prevent crashes
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']

    # io Bytes wraps as a file-like stream, Image to decode, RGB for model
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    label, confidence = model.run(image)

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
