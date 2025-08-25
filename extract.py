import os
import cv2
import numpy as np
from tqdm import tqdm

# Your dataset path
dataset_path = "/Users/balreenkaur/Desktop/WIL/FaceExpression/fer2013/test"

# Lists to store images and labels
images = []
labels = []

# Loop through each emotion folder
for label_name in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label_name)
    if os.path.isdir(label_folder):
        for img_file in tqdm(os.listdir(label_folder), desc=f"Loading {label_name}"):
            img_path = os.path.join(label_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_name)
            else:
                print("Failed to read:", img_path)

images = np.array(images)
labels = np.array(labels)

print("Loaded image shape:", images.shape)
print("Labels:", set(labels))
