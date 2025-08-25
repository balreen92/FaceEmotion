# vision/train_from_dirs.py

import tensorflow as tf
from tensorflow.keras import layers, models

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
    class_names=["angry","disgust","fear","happy","neutral","sad","surprise"],
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42,
)

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
