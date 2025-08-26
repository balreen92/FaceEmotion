# vision/train_from_dirs.py

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2 as cv

# 1. Point to your data directories
train_dir = "data/fer-2013/train"
test_dir  = "data/fer-2013/test"

# 2. Create datasets from folders
IMG_SIZE = (48, 48)
BATCH_SIZE = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=["angry","disgust","fearful","happy","neutral","sad","surprised"],
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42,
)


def crop112(img, det, pad=0.35, align=False):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        # Fallback: Haar (frontal)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        haar = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        boxes = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(60, 60))
        if len(boxes) == 0:
            return None
        x, y, ww, hh = boxes[0]
    else:
        x, y, ww, hh = faces[0][:4].astype(int)

    cx, cy = x + ww / 2, y + hh / 2
    side = int(max(ww, hh) * (1 + 2 * pad))
    nx, ny = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    nxe, nye = min(w, nx + side), min(h, ny + side)
    crop = img[ny:nye, nx:nxe]
    if crop.size == 0:
        return None
    return cv.resize(crop, (112, 112), interpolation=cv.INTER_AREA)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=train_ds.class_names,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
)

# 3. Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# 4. Build a simple CNN (you can swap in transfer learning here)
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(7, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# 6. Save
model.save("emotion_vision_model_from_dirs.h5")
print("Saved model to emotion_vision_model_from_dirs.h5")
# After you have `model` defined and your `val_ds` created...

# 1. Evaluate on the validation (test) dataset
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

final_train_acc = history.history['accuracy'][-1]
final_val_acc   = history.history['val_accuracy'][-1]
print(f"Final train accuracy: {final_train_acc*100:.2f}%")
print(f"Final val   accuracy: {final_val_acc*100:.2f}%")
