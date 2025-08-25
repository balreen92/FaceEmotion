# evaluate_transfer.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from sklearn.metrics import classification_report

# ─── Config ───────────────────────────────────────────────────────────
MODEL_PATH = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/model/ResMaskNet_Z_resmasking_dropout1_rot30.pth"
TEST_DIR   = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/vision/data/fer-2013/test"
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DataLoader ────────────────────────────────────────────────────────
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
ds = datasets.ImageFolder(TEST_DIR, transform=tfms)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
classes = ds.classes

# ─── Load model ────────────────────────────────────────────────────────
# Recreate the same architecture
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ─── Inference & Report ────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labs in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labs.tolist())

print("=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=classes))
