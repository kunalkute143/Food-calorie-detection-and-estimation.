"""
build_ood.py   (run ONCE, no re-training needed)
------------------------------------------------
Teaches the app what "a known food" looks like inside the CNN, so it can
refuse images that are not food (people, objects, screenshots ...).

How it works (simple version):
 1. Pass all training images through the trained CNN and take the Dense(128)
    features of each image.
 2. For each food class, find the average feature vector.
 3. For any new image, measure how far its features are from the closest
    food class (Mahalanobis distance).
 4. If the distance is larger than what real validation foods usually show,
    the image is rejected as "not food".

Run:  python build_ood.py
Optional: put some NON-food photos in a folder named  non_food_test/
          and this script will tell you how many get rejected.
"""
import os

import numpy as np

from predict import (OOD_FILE, STRICTNESS_PERCENTILE, forward_batch, load_food_model,
                     mahalanobis_scores, preprocess_image, read_image)

_, class_names = load_food_model()


def list_images(root):
    paths, labels = [], []
    for ci, cname in enumerate(class_names):
        folder = os.path.join(root, cname)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, fn))
                labels.append(ci)
    return paths, np.array(labels)


def embed_paths(paths):
    """Return embeddings and the indices of images that could be read."""
    embs, kept = [], []
    for start in range(0, len(paths), 64):
        batch, idxs = [], []
        for j, p in enumerate(paths[start:start + 64]):
            try:
                batch.append(preprocess_image(read_image(p))[0])
                idxs.append(start + j)
            except ValueError:
                pass
        if batch:
            e, _ = forward_batch(np.stack(batch))
            embs.append(e)
            kept.extend(idxs)
        print(f"  processed {min(start + 64, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(embs), np.array(kept)


for folder in ("dataset/train", "dataset/validation"):
    if not os.path.isdir(folder):
        raise SystemExit(f"'{folder}' not found. The dataset folder is needed once for this step.")

print("Reading training images...")
tr_paths, tr_labels = list_images("dataset/train")
tr_emb, kept = embed_paths(tr_paths)
tr_labels = tr_labels[kept]

print("Reading validation images...")
va_paths, _ = list_images("dataset/validation")
va_emb, _ = embed_paths(va_paths)

for c, name in enumerate(class_names):
    if not (tr_labels == c).any():
        raise SystemExit(f"No training images found for class '{name}'.")

means = np.stack([tr_emb[tr_labels == c].mean(axis=0) for c in range(len(class_names))])
centered = tr_emb - means[tr_labels]
cov = centered.T @ centered / len(centered)
dim = cov.shape[0]
cov = cov + np.eye(dim) * (1e-2 * np.trace(cov) / dim)   # keeps the matrix invertible
prec = np.linalg.inv(cov)

val_scores = mahalanobis_scores(va_emb, means, prec)
np.savez(OOD_FILE, means=means, prec=prec, val_scores=val_scores)
print(f"\nSaved {OOD_FILE}  (train images: {len(tr_emb)}, validation images: {len(va_emb)})")

print("\nReal validation foods accepted at each strictness level:")
for level, pct in STRICTNESS_PERCENTILE.items():
    thr = np.percentile(val_scores, pct)
    print(f"  {level:7s} -> {np.mean(val_scores <= thr) * 100:.0f}% accepted")

# Optional test with non-food images
if os.path.isdir("non_food_test"):
    files = [os.path.join("non_food_test", f) for f in os.listdir("non_food_test")
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if files:
        nf_emb, _ = embed_paths(files)
        nf_scores = mahalanobis_scores(nf_emb, means, prec)
        print(f"\nNon-food test images ({len(files)}): percent REJECTED")
        for level, pct in STRICTNESS_PERCENTILE.items():
            thr = np.percentile(val_scores, pct)
            print(f"  {level:7s} -> {np.mean(nf_scores > thr) * 100:.0f}% rejected")
