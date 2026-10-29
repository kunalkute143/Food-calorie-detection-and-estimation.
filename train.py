"""
train.py
--------
Trains a simple CNN on dataset/train and dataset/validation and saves:
  food_model.h5       -> the trained model
  class_names.json    -> list of class names (same order as model outputs)
  training_history.csv, training_plot.png -> accuracy/loss records

Run:  python train.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # save plots to a file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- Settings ----------------
TRAIN_DIR = os.path.join("dataset", "train")
VAL_DIR = os.path.join("dataset", "validation")
IMG_SIZE = (128, 128)      # every image is resized to 128 x 128
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = "food_model.h5"
CLASS_FILE = "class_names.json"

for folder in (TRAIN_DIR, VAL_DIR):
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder '{folder}' not found. Prepare the dataset first.")

# ---------------- 1. Load data ----------------
# Folder names become class labels automatically.
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=True, seed=42)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes found:", class_names)

# Save class names so predict.py / app.py use the exact same order
with open(CLASS_FILE, "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------- 2. Build the CNN ----------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    # Preprocessing: scale pixel values from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # Data augmentation (only active during training)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Convolution block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------------- 3. Train ----------------
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ---------------- 4. Save model ----------------
model.save(MODEL_PATH)
print(f"\nModel saved as {MODEL_PATH}")

# ---------------- 5. Save history and plots ----------------
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("training_history.csv", index=False)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_df["accuracy"], label="Train")
plt.plot(hist_df["val_accuracy"], label="Validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_df["loss"], label="Train")
plt.plot(hist_df["val_loss"], label="Validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
print("Saved training_plot.png and training_history.csv")
print(f"Final validation accuracy: {hist_df['val_accuracy'].iloc[-1] * 100:.2f}%")
