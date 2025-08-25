# quick_check_accuracy.py

import torch

# 1. Point this at your .pth
MODEL_PATH = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/model/ResMaskNet_Z_resmasking_dropout1_rot30.pth"

# 2. Load the checkpoint dict
ckpt = torch.load(MODEL_PATH, map_location="cpu")

# 3. Print the stored metrics
print("✔️ Checkpoint keys:", list(ckpt.keys()))
print(f"Best train  accuracy: {ckpt.get('best_train_acc')}")
print(f"Best valid  accuracy: {ckpt.get('best_val_acc')}")
print(f"Test       accuracy: {ckpt.get('test_acc')}")
