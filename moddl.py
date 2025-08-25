import torch
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet

# 1. Instantiate the network architecture
model = ResMaskNet()

# 2. Load the checkpoint
checkpoint = torch.load("ResMaskNet_Z_resmaski…ot30.pth", map_location="cpu")
model.load_state_dict(checkpoint)

# 3. Switch to eval mode
model.eval()

# 4. (Optional) Print a summary
print(model)

