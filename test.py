
import sys
import os
sys.path.append(os.path.abspath('.'))

from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet

model = ResMaskNet()


import torch
import numpy as np
import cv2
model = torch.load("/Users/balreenkaur/Desktop/FaceEmotion/trained_models/ResMaskNet_Z_resmasking_dropout1_rot30.pth", map_location=torch.device('cpu'))
model.eval()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FERTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.label_map = {}  # label name to number
        self.transform = transform

        for idx, label_name in enumerate(sorted(os.listdir(root_dir))):
            self.label_map[label_name] = idx
            label_folder = os.path.join(root_dir, label_name)
            if os.path.isdir(label_folder):
                for img_file in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img_file)
                    self.images.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = np.expand_dims(image, axis=0)  # 1 channel
        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(self.labels[idx])


# Create the dataset and dataloader
dataset = FERTestDataset("/Users/balreenkaur/Desktop/WIL/FaceExpression/fer2013/test")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")






# import pandas as pd
# import numpy as np
# import torch
# import cv2
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'feat', 'feat')))

# from torchvision import transforms
# from emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
# from sklearn.metrics import accuracy_score
#
# # Load CSV
# csv_path = "/Users/balreenkaur/Desktop/WIL/FaceExpression/fer2013/test"
# df = pd.read_csv(csv_path)
#
# # Use only 'PublicTest' (for evaluation)
# df = df[df['Usage'] == 'PublicTest']
#
# # Prepare emotion labels
# emotion_labels = {
#     0: 'angry',
#     1: 'disgust',
#     2: 'fear',
#     3: 'happy',
#     4: 'sad',
#     5: 'surprise',
#     6: 'neutral'
# }
#
# # Transform
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
#
# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResMaskNet(device=device, pretrained="local")
# model.eval()
#
# y_true = []
# y_pred = []
#
# # Loop through test samples
# for index, row in df.iterrows():
#     emotion = int(row['emotion'])
#     pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
#     img = cv2.cvtColor(pixels.astype('uint8'), cv2.COLOR_GRAY2RGB)
#
#     # Apply transform
#     img_tensor = transform(img).unsqueeze(0).to(device)
#
#     # Predict
#     with torch.no_grad():
#         output = model.model(img_tensor)
#         predicted = torch.argmax(output, dim=1).item()
#
#     y_true.append(emotion)
#     y_pred.append(predicted)
#
# # Accuracy
# acc = accuracy_score(y_true, y_pred)
# print(f"\n✅ Model accuracy on FER-2013 PublicTest set: {acc * 100:.2f}%")