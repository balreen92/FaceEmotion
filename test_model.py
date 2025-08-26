import os
import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained model
model = tf.keras.models.load_model("emotion_vision_model_from_dirs.h5")

# 2. Emotion labels
emotions = ["angry","disgust","fearful","happy","neutral","sad","surprised"]

# 3. Point to a test sub-folder — change “happy” to any class you want to test
test_class = "happy"
test_folder = os.path.join("data", "fer-2013", "test", test_class)

# 4. Grab the first image file we find there
files = [f for f in os.listdir(test_folder)
         if f.lower().endswith(('.png','.jpg','.jpeg','.pgm'))]
if not files:
    print(f"No image files found in {test_folder}")
    exit(1)

img_path = os.path.join(test_folder, files[0])
print(f"Testing on image: {img_path}")

# 5. Load & preprocess
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Failed to read image at {img_path}")
    exit(1)

img = cv2.resize(img, (48, 48)) / 255.0
inp = np.expand_dims(img, axis=(0, -1))  # shape → (1,48,48,1)

# 6. Predict
pred = model.predict(inp)
idx  = np.argmax(pred, axis=1)[0]
conf = pred[0][idx] * 100

print(f"Predicted emotion: {emotions[idx]} ({conf:.1f}% confidence)")
