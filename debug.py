# accuract_test_debug.py

import os
import cv2
from deepface import DeepFace

# 1. Configuration
TEST_DIR = "data/fer-2013/test"    # adjust this if your test images live elsewhere
VALID_EXTS = {".jpg", ".jpeg", ".png", ".pgm"}

# Debug: check that TEST_DIR exists
print(f">>> Looking for test images in: {TEST_DIR}")
if not os.path.isdir(TEST_DIR):
    print("❌ Test directory not found! Check your path.")
    exit(1)

# 2. Find all emotion subfolders
classes = [d for d in os.listdir(TEST_DIR)
           if os.path.isdir(os.path.join(TEST_DIR, d))]
print(f">>> Found emotion classes: {classes}")

total = 0
correct = 0
per_class = {}

# 3. Iterate and predict
for true_label in classes:
    class_dir = os.path.join(TEST_DIR, true_label)
    per_class.setdefault(true_label, {"correct": 0, "total": 0})

    files = [f for f in os.listdir(class_dir)
             if os.path.splitext(f)[1].lower() in VALID_EXTS]
    print(f">>> Checking {len(files)} images for class '{true_label}'")

    for fname in files:
        img_path = os.path.join(class_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ⚠️  Could not read {img_path}, skipping")
            continue

        # run DeepFace with OpenCV backend (avoids tf-keras error)
        # run DeepFace with OpenCV backend
        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        # DeepFace sometimes returns a list → grab the first dict
        if isinstance(result, list):
            result = result[0]

        pred = result["dominant_emotion"]


        # tally
        per_class[true_label]["total"] += 1
        total += 1
        if pred.lower() == true_label.lower():
            per_class[true_label]["correct"] += 1
            correct += 1

# 4. Final reporting
if total == 0:
    print("❌ No valid images processed! Check your folder structure and extensions.")
else:
    overall_acc = 100 * correct / total
    print(f"\n✅ Overall accuracy: {correct}/{total} = {overall_acc:.2f}%\n")
    print("Per-class accuracy:")
    for label, stats in per_class.items():
        t = stats["total"]
        c = stats["correct"]
        acc = 100 * c / t if t else 0
        print(f"  {label:>8}: {c}/{t} = {acc:.2f}%")
