# train_transfer.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # ─── 1. Configuration ────────────────────────────────────────────────
    DATA_DIR   = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/vision/data/fer-2013"
    MODEL_OUT  = "/Users/balreenkaur/Desktop/WIL_PRO/ai-kiosk-ml/model/ResMaskNet_Z_resmasking_dropout1_rot30.pth"
    BATCH_SIZE = 64
    NUM_EPOCHS_HEAD = 1    # try 1 for a quick sanity check
    NUM_EPOCHS_FINE = 1    # try 1 for a quick sanity check
    LR_HEAD    = 1e-3
    LR_FINE    = 1e-5
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4        # set to 0 if you hit multiprocessing issues

    print(f">>> Using device: {DEVICE}")

    # ─── 2. Data Transforms & Loaders ────────────────────────────────────
    train_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR,"test"),  transform=test_tfms)

    # QUICK SUBSET FOR SANITY TESTING (uncomment to use)
    # train_ds = Subset(train_ds, list(range(2000)))
    # test_ds  = Subset(test_ds,  list(range(1000)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    class_names = train_ds.dataset.classes if isinstance(train_ds, Subset) else train_ds.classes
    print(f">>> Found classes: {class_names}")

    # ─── 3. Build & Prepare Model ────────────────────────────────────────
    model = models.resnet50(pretrained=True)
    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    # Replace head
    num_ftrs = model.fc.in_features
    model.fc  = nn.Linear(num_ftrs, len(class_names))
    model = model.to(DEVICE)

    # ─── 4. Train Head ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    opt_head  = optim.Adam(model.fc.parameters(), lr=LR_HEAD)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_acc":[], "test_acc":[]}

    for epoch in range(NUM_EPOCHS_HEAD):
        # train
        model.train()
        all_preds, all_labels = [], []
        loop = tqdm(train_loader, desc=f"[Head] Epoch {epoch+1}/{NUM_EPOCHS_HEAD}", leave=False)
        for imgs, labs in loop:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labs)
            opt_head.zero_grad(); loss.backward(); opt_head.step()

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().tolist())

        train_acc = accuracy_score(all_labels, all_preds)

        # validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            loop = tqdm(test_loader, desc="  Validating", leave=False)
            for imgs, labs in loop:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                outs = model(imgs)
                preds = outs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs.cpu().tolist())

        test_acc = accuracy_score(all_labels, all_preds)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"[Head] Epoch {epoch+1}/{NUM_EPOCHS_HEAD}  train={train_acc:.3f}  test={test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # ─── 5. Fine-Tune Last Block ───────────────────────────────────────────
    # Unfreeze layer4 and fc
    for name, p in model.named_parameters():
        if "layer4" in name or "fc" in name:
            p.requires_grad = True

    opt_fine = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINE
    )

    for epoch in range(NUM_EPOCHS_FINE):
        model.train()
        all_preds, all_labels = [], []
        loop = tqdm(train_loader, desc=f"[Fine] Epoch {epoch+1}/{NUM_EPOCHS_FINE}", leave=False)
        for imgs, labs in loop:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labs)
            opt_fine.zero_grad(); loss.backward(); opt_fine.step()

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().tolist())

        train_acc = accuracy_score(all_labels, all_preds)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            loop = tqdm(test_loader, desc="  Validating", leave=False)
            for imgs, labs in loop:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                outs = model(imgs)
                preds = outs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs.cpu().tolist())

        test_acc = accuracy_score(all_labels, all_preds)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"[Fine] Epoch {epoch+1}/{NUM_EPOCHS_FINE}  train={train_acc:.3f}  test={test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f">>> Best test accuracy: {best_acc:.3f}")
    torch.save(best_model_wts, MODEL_OUT)
    print(f">>> Model saved to {MODEL_OUT}")

    # ─── 6. Plot training curves ─────────────────────────────────────────
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"],  label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
