#!/usr/bin/env python3
"""
Classify PET plastic bottle images using a trained Keras CNN model.

Classification categories:
  no_label   - Colorless transparent PET bottle (ready for recycling)
  with_label - Colorless transparent PET bottle with label sticker (remove sticker first)
  bad        - Dirty, colored, non-transparent, not a bottle, etc.

Usage:
  # Single image
  python classify.py --model best_step6_vgg16_finetuned.keras --image photo.jpg

  # Directory of images (model stays loaded for speed)
  python classify.py --model best_step6_vgg16_finetuned.keras --image ./photos/

  # Show timing information
  python classify.py --model best_step6_vgg16_finetuned.keras --image ./photos/ --time
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ── Class mapping (alphabetical order from ImageDataGenerator.flow_from_directory) ──
CLASS_NAMES = ["bad", "no_label", "with_label"]

CLASS_DESCRIPTIONS = {
    "bad": "Bad (dirty/colored/not-a-bottle)",
    "no_label": "No label (transparent, recyclable as-is)",
    "with_label": "With label (remove sticker before recycling)",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def load_model(model_path: str):
    """Load and return a Keras model."""
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.\n")
    return model


def classify_single(model, img_path: str) -> tuple[str, float, np.ndarray]:
    """
    Classify a single image.

    Returns:
        (predicted_class, confidence, all_probabilities)
    """
    # Load and resize image to 224x224 (VGG16 input size)
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to numpy array and preprocess
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # rescale (matches training preprocessing)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    class_idx = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probabilities[class_idx])

    return class_name, confidence, probabilities


def collect_images(path: str) -> list[str]:
    """
    If path is a file, return [path].
    If path is a directory, return sorted list of image files within.
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        images = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        return images

    print(f"Error: '{path}' is not a valid file or directory.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Classify PET bottle images using a trained Keras CNN model."
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the .keras model file",
    )
    parser.add_argument(
        "-i", "--image",
        required=True,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "-t", "--time",
        action="store_true",
        help="Print total time consumed",
    )
    args = parser.parse_args()

    # Validate model path
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    # Collect images
    images = collect_images(args.image)

    if not images:
        print("No image files found.")
        sys.exit(1)

    # Load model (done once)
    total_start = time.perf_counter() if args.time else None

    model = load_model(args.model)

    model_load_time = time.perf_counter() - total_start if args.time else 0.0

    # Classify each image (model stays loaded for directory mode)
    counts = {name: 0 for name in CLASS_NAMES}

    for i, img_path in enumerate(images, 1):
        predict_start = time.perf_counter() if args.time else None

        class_name, confidence, probabilities = classify_single(model, img_path)

        predict_time = time.perf_counter() - predict_start if args.time else 0.0
        counts[class_name] += 1

        filename = os.path.basename(img_path)
        desc = CLASS_DESCRIPTIONS[class_name]

        print(f"[{i}/{len(images)}] {filename}  →  {class_name} ({confidence:.1%})")
        print(f"         {desc}")
        if args.time:
            print(f"         inference time: {predict_time:.4f}s")
        # Print per-class probabilities
        prob_str = "  ".join(
            f"{CLASS_NAMES[j]}: {probabilities[j]:.4f}" for j in range(len(CLASS_NAMES))
        )
        print(f"         probabilities: {prob_str}")

    # Summary
    print(f"\n{'='*60}")
    print("Classification Summary")
    print(f"{'='*60}")
    print(f"  Total images:  {len(images)}")
    for name in CLASS_NAMES:
        print(f"  {name:<12}:  {counts[name]:>5}  ({counts[name]/len(images)*100:.1f}%)")

    if args.time:
        total_time = time.perf_counter() - total_start
        inference_time = total_time - model_load_time
        print(f"\n  Timing:")
        print(f"    Model load time:    {model_load_time:.4f}s")
        print(f"    Total inference:    {inference_time:.4f}s")
        print(f"    Avg per image:      {inference_time/len(images)*1000:.2f}ms")
        print(f"    Total elapsed:      {total_time:.4f}s")


if __name__ == "__main__":
    main()