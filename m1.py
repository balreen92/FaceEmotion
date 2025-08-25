import feat
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
model = ResMaskNet()   # should no longer FileNotFoundError
print("✅ Loaded ResMaskNet with its JSON config!")
