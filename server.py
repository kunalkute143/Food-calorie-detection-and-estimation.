"""
server.py
---------
Flask backend for the React frontend.

Run:  python server.py        (starts at http://localhost:5000)

API:
  GET  /api/health    -> model status + list of foods
  POST /api/predict   -> form-data: image, grams (optional), strictness (low|medium|high)
"""
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from calories import CALORIES_PER_100G, DISCLAIMER, estimate_calories
from predict import analyze_image, load_food_model, ood_available

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024   # 15 MB upload limit

_, CLASS_NAMES = load_food_model()   # load once at start-up


@app.get("/api/health")
def health():
    return jsonify(model_loaded=True, ood_ready=ood_available(), classes=CLASS_NAMES)


@app.post("/api/predict")
def predict_route():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded."), 400

    data = np.frombuffer(file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="Could not read this image file."), 400
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    grams = request.form.get("grams", type=float)
    strictness = request.form.get("strictness", "medium")

    result = analyze_image(rgb, strictness=strictness)
    result["calories"] = None
    result["grams_used"] = None
    result["portion_is_custom"] = bool(grams and grams > 0)
    result["disclaimer"] = DISCLAIMER

    if result["recognised"] and result["food"] in CALORIES_PER_100G:
        kcal, used = estimate_calories(result["food"], grams)
        result["calories"] = kcal
        result["grams_used"] = used
    return jsonify(result)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def frontend(path):
    """Serves the built React app (after `npm run build`)."""
    if os.path.isdir(DIST):
        if path and os.path.isfile(os.path.join(DIST, path)):
            return send_from_directory(DIST, path)
        return send_from_directory(DIST, "index.html")
    return ("API is running. Start the UI with:  cd frontend  &&  npm run dev  "
            "and open http://localhost:5173"), 200


if __name__ == "__main__":
    if not ood_available():
        print("WARNING: ood_stats.npz not found. Non-food images will NOT be rejected well.\n"
              "         Run:  python build_ood.py")
    app.run(host="127.0.0.1", port=5000, debug=False)
