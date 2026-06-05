#!/usr/bin/env python3
"""
Classify "bad/other" (class 3) plastic bottle images into 3 subclasses using
OpenRouter Vision API.

Subclasses:
  1 - colored   : Colored / non-transparent bottle (blue, green, opaque, etc.)
  2 - dirty     : Dirty / contaminated bottle (filled with liquid, food residue, grime)
  3 - not_bottle : Not a bottle (other objects, garbage, non-bottle items)

Images are processed concurrently.
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an image classifier for plastic bottle images. Classify the given image into exactly ONE of these three subclasses:

1 - colored: A colored or non-transparent plastic bottle (blue, green, red, opaque, translucent, etc.)
2 - dirty: A dirty or contaminated plastic bottle (filled with liquid/food, has residue, grime, mold, or is visibly unclean)
3 - not_bottle: NOT a plastic bottle at all (other objects, trash, garbage, non-bottle items, or images where no bottle is clearly visible)

Respond with ONLY a single digit: 1, 2, or 3. No explanation, no punctuation, just the digit."""

USER_PROMPT = "Classify this image. Respond with only 1, 2, or 3."

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_to_base64_url(image_path: str) -> str:
    """Read an image file and return a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def classify_subclass(image_path: str, api_key: str, model: str, retries: int = 3) -> int:
    """Send a single image to OpenRouter and return the subclass (1, 2, or 3)."""
    image_url = image_to_base64_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 4,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()

            content = result["choices"][0]["message"]["content"].strip()

            # Extract the first digit found in the response
            for char in content:
                if char in ("1", "2", "3"):
                    return int(char)

            print(f"  WARNING: Unexpected response '{content}', defaulting to 3 (not_bottle)")
            return 3

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"  HTTP error {status} on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts, classifying as 3")
                return 3

        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
            print(f"  Error on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts, classifying as 3")
                return 3


def main():
    parser = argparse.ArgumentParser(
        description="Classify bad/other (class 3) images into subclasses via OpenRouter Vision API"
    )
    parser.add_argument(
        "-c", "--config",
        default="classify_subclass.yaml",
        help="Path to YAML config file (default: classify_subclass.yaml)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    api_key = cfg["api_key"]
    model = cfg["model"]
    input_dir = cfg["input_dir"]
    out_colored = cfg["output_colored_dir"]
    out_dirty = cfg["output_dirty_dir"]
    out_not_bottle = cfg["output_not_bottle_dir"]
    max_workers = cfg.get("max_workers", 1)

    # Create output directories
    for d in (out_colored, out_dirty, out_not_bottle):
        os.makedirs(d, exist_ok=True)

    # Collect image files
    images = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Model: {model}")
    print(f"Workers: {max_workers}")
    print("-" * 60)

    # Track stats
    counts = {1: 0, 2: 0, 3: 0}
    dest_map = {1: out_colored, 2: out_dirty, 3: out_not_bottle}
    label_map = {1: "colored", 2: "dirty", 3: "not_bottle"}

    # All output directories to check for already-classified files
    all_output_dirs = [out_colored, out_dirty, out_not_bottle]

    # Lock for thread-safe skip-check + file copy and stats update
    io_lock = threading.Lock()

    def already_classified(src_path: str, filename: str) -> bool:
        """Check if filename already exists in any output dir with matching size."""
        src_size = os.path.getsize(src_path)
        for d in all_output_dirs:
            out_path = os.path.join(d, filename)
            if os.path.exists(out_path) and os.path.getsize(out_path) == src_size:
                return True
        return False

    def process_image(i: int, filename: str):
        """Classify a single image and copy to the correct output dir. Thread-safe."""
        src_path = os.path.join(input_dir, filename)

        with io_lock:
            if already_classified(src_path, filename):
                print(f"[{i}/{len(images)}] {filename} ... skipped (already classified)")
                return "skipped"

        category = classify_subclass(src_path, api_key, model)

        with io_lock:
            dest_dir = dest_map[category]
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_dir, f"{base}_{i}{ext}")
            shutil.copy2(src_path, dest_path)

            counts[category] += 1
            label = label_map[category]
            print(f"[{i}/{len(images)}] {filename} ... {label}")
            return label

    skipped = 0
    if max_workers <= 1:
        # Sequential mode
        for i, filename in enumerate(images, 1):
            result = process_image(i, filename)
            if result == "skipped":
                skipped += 1
    else:
        # Concurrent mode
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, filename in enumerate(images, 1):
                future = executor.submit(process_image, i, filename)
                futures[future] = filename
            for future in as_completed(futures):
                result = future.result()
                if result == "skipped":
                    skipped += 1

    # Summary
    print("-" * 60)
    print("Subclass classification complete!")
    print(f"  1 (colored):     {counts[1]}")
    print(f"  2 (dirty):       {counts[2]}")
    print(f"  3 (not_bottle):  {counts[3]}")
    print(f"  Skipped:         {skipped}")
    print(f"  Total:           {sum(counts.values()) + skipped}")


if __name__ == "__main__":
    main()