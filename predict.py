"""
predict.py
----------
Loads food_model.h5, predicts the food and decides whether the image is
really one of the foods the model knows.

Command line:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg 250      (250 = portion in grams)

Other files import from here:
    server.py     -> analyze_image()
    build_ood.py  -> forward_batch(), mahalanobis_scores()
    app.py (old Streamlit UI) -> predict_food()
"""
import json
import os
import sys

import cv2
import numpy as np

from calories import DISCLAIMER, estimate_calories, pretty_name

MODEL_PATH = "food_model.h5"
CLASS_FILE = "class_names.json"
OOD_FILE = "ood_stats.npz"      # created by build_ood.py
IMG_SIZE = 128                  # must be the same size used in train.py

# Higher percentile = more relaxed. Lower = stricter (rejects more).
STRICTNESS_PERCENTILE = {"low": 99, "medium": 95, "high": 85}

_model = None
_class_names = None
_ood = None
_ood_checked = False


# ---------------- Loading ----------------
def load_food_model():
    """Load the trained model and class names (only once)."""
    global _model, _class_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH, compile=False)
        with open(CLASS_FILE) as f:
            _class_names = json.load(f)
    return _model, _class_names


def load_ood():
    """Load the 'known food' statistics made by build_ood.py (None if missing)."""
    global _ood, _ood_checked
    if not _ood_checked:
        _ood_checked = True
        if os.path.exists(OOD_FILE):
            d = np.load(OOD_FILE)
            _ood = {"means": d["means"], "prec": d["prec"], "val_scores": d["val_scores"]}
    return _ood


def ood_available():
    return load_ood() is not None


# ---------------- Image helpers ----------------
def read_image(path):
    """Read an image file with OpenCV and convert BGR -> RGB."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(img_rgb):
    """Resize to 128x128 and add a batch dimension.
    (Scaling to 0-1 is done inside the model by the Rescaling layer.)"""
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(resized.astype("float32"), axis=0)  # (1,128,128,3)


def photo_check(img_rgb):
    """
    Simple check to catch screenshots / documents: they have big flat areas
    (plain white or a single colour), real food photos have texture everywhere.
    Returns (looks_like_photo, flat_ratio, white_ratio).
    """
    small = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    flat_ratio = float((magnitude < 2).mean())
    white_ratio = float((small.min(axis=2) > 235).mean())
    ok = not (flat_ratio > 0.5 or white_ratio > 0.6)
    return ok, flat_ratio, white_ratio


# ---------------- Model forward pass ----------------
def forward_batch(batch):
    """
    Run images through the CNN and return BOTH:
      embeddings : output of the Dense(128) layer (what the model "understood")
      probs      : softmax probabilities for each food class
    """
    model, _ = load_food_model()
    h = batch
    for layer in model.layers[:-2]:      # everything up to Dense(128)
        h = layer(h, training=False)
    probs = model.layers[-1](h)          # final softmax layer
    return np.asarray(h), np.asarray(probs)


def mahalanobis_scores(embs, means, prec):
    """Distance of each embedding to the closest food class (smaller = more food-like)."""
    diff = embs[:, None, :] - means[None, :, :]                  # (N, C, D)
    d = np.einsum("ncd,de,nce->nc", diff, prec, diff, optimize=True)
    return d.min(axis=1)


# ---------------- Public prediction functions ----------------
def predict_food(img_rgb):
    """Old simple API: (food_name, confidence_percent, all_probabilities, class_names)."""
    _, class_names = load_food_model()
    _, probs = forward_batch(preprocess_image(img_rgb))
    probs = probs[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]) * 100, probs, class_names


def analyze_image(img_rgb, strictness="medium", min_conf=35.0, warn_conf=60.0):
    """
    Full analysis with rejection of non-food images.
    A prediction is accepted only if ALL checks pass:
      1. it looks like a photo (not a screenshot/document)
      2. its CNN features are close to the known foods (out-of-distribution check)
      3. the model confidence is at least min_conf %
    If the confidence is between min_conf and warn_conf, the best guess is still
    returned, but with low_confidence = True so the app can show a warning.
    """
    _, class_names = load_food_model()
    is_photo, flat_ratio, white_ratio = photo_check(img_rgb)

    emb, probs = forward_batch(preprocess_image(img_rgb))
    emb, probs = emb[0], probs[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    top3 = [{"name": class_names[i], "label": pretty_name(class_names[i]),
             "prob": float(probs[i]) * 100} for i in np.argsort(probs)[::-1][:3]]

    ood = load_ood()
    ood_score = ood_threshold = None
    looks_known = True
    if ood is not None:
        pct = STRICTNESS_PERCENTILE.get(strictness, 95)
        ood_threshold = float(np.percentile(ood["val_scores"], pct))
        ood_score = float(mahalanobis_scores(emb[None, :], ood["means"], ood["prec"])[0])
        looks_known = ood_score <= ood_threshold

    reason, message = None, ""
    if not is_photo:
        reason = "not_a_photo"
        message = "This looks like a screenshot or document, not a food photo."
    elif not looks_known:
        reason = "not_food"
        message = "This does not look like any of the foods this model was trained on."
    elif confidence < min_conf:
        reason = "low_confidence"
        message = "I am not confident enough to name this food. Try a clearer photo of a single dish."

    low_confidence = reason is None and confidence < warn_conf
    if low_confidence:
        message = "Low confidence: this is the best guess of the model and it may be wrong."

    return {
        "recognised": reason is None,
        "reason": reason,
        "low_confidence": low_confidence,
        "message": message,
        "food": class_names[idx],
        "food_label": pretty_name(class_names[idx]),
        "confidence": confidence,
        "top3": top3,
        "ood_checked": ood is not None,
        "debug": {"ood_score": ood_score, "ood_threshold": ood_threshold,
                  "flat_ratio": flat_ratio, "white_ratio": white_ratio},
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()
    grams = float(sys.argv[2]) if len(sys.argv) > 2 else None

    r = analyze_image(read_image(path))
    print("\n----- RESULT -----")
    if not r["recognised"]:
        print(f"Not recognised: {r['message']}")
        return
    print(f"Predicted food : {r['food_label']}")
    print(f"Confidence     : {r['confidence']:.2f}%")
    if r["low_confidence"]:
        print("Warning        : low confidence, the prediction may be wrong")
    result = estimate_calories(r["food"], grams)
    if result:
        kcal, used = result
        print(f"Portion used   : {used} g")
        print(f"Estimated kcal : {kcal}")
    else:
        print("No calorie data for this food. Add it to calories.py")
    print(f"\nNote: {DISCLAIMER}")


if __name__ == "__main__":
    main()