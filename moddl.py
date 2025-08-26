# import torch
# from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
#
# # 1. Instantiate the network architecture
# model = ResMaskNet()
#
# # 2. Load the checkpoint
# checkpoint = torch.load("ResMaskNet_Z_resmaski…ot30.pth", map_location="cpu")
# model.load_state_dict(checkpoint)
#
# # 3. Switch to eval mode
# model.eval()
#
# # 4. (Optional) Print a summary
# print(model)
import cv2 as cv
for i in range(5):
    cap = cv.VideoCapture(i, cv.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Index {i}: {'OPEN' if ok else 'FAIL'}")
    if ok:
        ret, frame = cap.read()
        print("  Frame:", "OK" if ret else "NO")
        cap.release()
