import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from sklearn.metrics import classification_report

# ─── Paths (adjust these!) ──────────────────────────────────────────
MODEL_PATH = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/model/ResMaskNet_Z_resmasking_dropout1_rot30.pth"
TEST_DIR   = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/vision/data/fer-2013/test"

# ─── Setup ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

tfms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])
test_ds = datasets.ImageFolder(TEST_DIR, transform=tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
class_names = test_ds.classes
print(f">>> Classes: {class_names}")

# ─── Load model ──────────────────────────────────────────────────────
print(f">>> Loading ResMaskNet from {MODEL_PATH}")
model = ResMaskNet().to(device)
sd = torch.load(MODEL_PATH, map_location=device)
clean_sd = {k.replace("module.", ""): v for k,v in sd.items()}
model.load_state_dict(clean_sd)
model.eval()
print("✅ Model loaded\n")

# ─── Evaluate ────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

total   = len(all_labels)
correct = sum(p==t for p,t in zip(all_preds, all_labels))
acc     = 100 * correct / total if total else 0.0
print(f"\nOverall Accuracy: {correct}/{total} = {acc:.2f}%\n")
print("Per-class report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
