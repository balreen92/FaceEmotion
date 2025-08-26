
#!/usr/bin/env python3

"""
Fine-tune a pre-trained facial expression classifier (7 classes) with strong tricks:
- Freezing then unfreezing backbone
- Class-balanced sampling / class weights
- Label smoothing
- Focal loss (optional)
- Mixup/CutMix (optional)
- Cosine LR schedule + warmup
- Test-time augmentation during validation (flip averaging)
- Exports best model to ONNX

Data layout (train & val):
data_root/
  train/
    angry/ ...
    disgust/ ...
    fearful/ ...
    happy/ ...
    neutral/ ...
    sad/ ...
    surprised/ ...
  val/
    angry/ ...
    ...

Usage (example):
  python fer_finetune.py --data_root /path/to/data --epochs 15 --batch_size 64 --mixup 0.2 --cutmix 0.2 --focal --label_smoothing 0.05

Author: ChatGPT
"""
import argparse, os, math, random, time, shutil, json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from torchvision import transforms, datasets, models

# Optional OpenCV for face-centric crop (preprocessing)
import cv2 as cv

# ----------------------------- Utils -----------------------------

SEVEN_LABELS = ["angry","disgust","fearful","happy","neutral","sad","surprised"]

def seed_all(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        num_classes = logits.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = F.one_hot(target, num_classes=num_classes).float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = -(true_dist * log_probs).sum(dim=-1)  # per-sample CE
        pt = (true_dist * probs).sum(dim=-1).clamp(1e-6, 1.0)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def get_loss_fn(use_focal: bool, label_smoothing: float, class_weights=None):
    if use_focal:
        return FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=label_smoothing)
    else:
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)
        else:
            cw = None
        return nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

def get_model(num_classes=7, freeze_backbone_epochs=3):
    # Use torchvision resnet18 as a strong baseline (fast on CPU, good accuracy)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = m.fc.in_features
    m.fc = nn.Linear(in_feat, num_classes)
    return m

# ------------------ OpenCV YuNet face crop (optional) ------------------
class YuNetCropper:
    def __init__(self, yunet_path=None, input_size=(320,240), score_thr=0.9, nms_thr=0.3, topk=5000, pad=0.25, align=False):
        self.pad = pad
        self.align = align
        if yunet_path is None:
            raise ValueError("Provide path to YuNet ONNX (face_detection_yunet_2023mar.onnx)")
        self.det = cv.FaceDetectorYN_create(yunet_path, "", input_size, score_thr, nms_thr, topk)

    def crop112(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[0]
        x, y, ww, hh = [float(v) for v in f[:4]]
        # landmarks
        le = (float(f[4]),  float(f[5]))
        re = (float(f[6]),  float(f[7]))

        cx, cy = x + ww/2.0, y + hh/2.0
        side = max(ww, hh) * (1.0 + 2*self.pad)
        nx = int(max(0, cx - side/2.0)); ny = int(max(0, cy - side/2.0))
        nxe = int(min(w, cx + side/2.0)); nye = int(min(h, cy + side/2.0))
        crop = img_bgr[ny:nye, nx:nxe]
        if crop.size == 0: return None

        if self.align:
            dy = re[1] - le[1]; dx = re[0] - le[0]
            angle = -np.degrees(np.arctan2(dy, dx))  # negative to correct tilt
            ch, cw = crop.shape[:2]
            M = cv.getRotationMatrix2D((cw/2.0, ch/2.0), angle, 1.0)
            crop = cv.warpAffine(crop, M, (cw, ch), flags=cv.INTER_LINEAR)

        face112 = cv.resize(crop, (112,112), interpolation=cv.INTER_AREA)
        return face112

# Custom Dataset that applies YuNet crop before transforms, with fallback
class FaceFolderDataset(Dataset):
    def __init__(self, root, class_to_idx, use_yunet=False, yunet_path=None, pad=0.25, align=False, transform=None):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
        self.transform = transform
        self.use_yunet = use_yunet
        self.cropper = YuNetCropper(yunet_path, pad=pad, align=align) if use_yunet else None
        exts = (".jpg",".jpeg",".png",".bmp",".webp")
        for cls in sorted(class_to_idx.keys()):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir): continue
            for fn in os.listdir(cdir):
                if fn.lower().endswith(exts):
                    self.samples.append((os.path.join(cdir, fn), class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = cv.imread(path)
        if img is None:
            # Return a dummy black image if unreadable
            img = np.zeros((112,112,3), dtype=np.uint8)
        if self.use_yunet and self.cropper is not None:
            crop = self.cropper.crop112(img)
            if crop is not None:
                img = crop
            else:
                img = cv.resize(img, (112,112), interpolation=cv.INTER_AREA)
        else:
            img = cv.resize(img, (112,112), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Convert to PIL-like tensor via torchvision transforms
        import PIL.Image as Image
        pil = Image.fromarray(img)
        if self.transform:
            pil = self.transform(pil)
        return pil, target

# ----------------------------- Mixup/CutMix -----------------------------
def do_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def do_cutmix(x, y, alpha=0.2):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=x.device)
    cx = np.random.randint(w); cy = np.random.randint(h)
    bw = int(w * np.sqrt(1 - lam)); bh = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - bw//2, 0, w); y1 = np.clip(cy - bh//2, 0, h)
    x2 = np.clip(cx + bw//2, 0, w); y2 = np.clip(cy + bh//2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x, (y_a, y_b), lam

def mix_criterion(criterion, preds, targets_mix):
    (y_a, y_b), lam = targets_mix
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

# ----------------------------- Training -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None, mixup=0.0, cutmix=0.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # Apply mixup or cutmix (one of them)
        targets_mix = None
        if cutmix > 0:
            imgs, targets_mix, lam = do_cutmix(imgs, labels, alpha=cutmix)
        elif mixup > 0:
            imgs, targets_mix, lam = do_mixup(imgs, labels, alpha=mixup)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            if targets_mix is not None:
                loss = mix_criterion(criterion, logits, targets_mix)
            else:
                loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, tta_flip=True):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        if tta_flip:
            logits_flip = model(torch.flip(imgs, dims=[3]))
            logits = (logits + logits_flip) / 2.0
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return loss_sum/total, correct/total

def build_loaders(data_root, batch_size, num_workers, use_yunet, yunet_path, pad, align, img_size=112):
    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    # Build datasets
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    # Discover classes from train dir
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted([c for c in classes if c in SEVEN_LABELS])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    # Warn if classes missing
    missing = [c for c in SEVEN_LABELS if c not in classes]
    if missing:
        print("[WARN] Missing classes in train set:", missing)

    train_ds = FaceFolderDataset(train_dir, class_to_idx, use_yunet, yunet_path, pad, align, transform=train_tf)
    val_ds   = FaceFolderDataset(val_dir,   class_to_idx, use_yunet, yunet_path, pad, align, transform=val_tf)

    # Class weights for imbalance (inverse frequency)
    counts = np.zeros(len(class_to_idx), dtype=np.float64)
    for _, t in train_ds.samples:
        counts[t] += 1
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()

    # Balanced sampler (optional) – here we use standard shuffle; provide flag to enable sampler if desired
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_to_idx, weights.tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="root with train/ and val/")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--freeze_epochs", type=int, default=3, help="freeze backbone for first N epochs")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--focal", action="store_true", help="use focal loss")
    p.add_argument("--use_yunet", action="store_true", help="use YuNet cropper in dataset")
    p.add_argument("--yunet_path", type=str, default="face_detection_yunet_2023mar.onnx")
    p.add_argument("--pad", type=float, default=0.25)
    p.add_argument("--align", action="store_true")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--export_onnx", action="store_true")
    args = p.parse_args()

    seed_all(1337)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx, class_weights = build_loaders(
        args.data_root, args.batch_size, args.num_workers,
        args.use_yunet, args.yunet_path, args.pad, args.align, img_size=args.img_size
    )
    print("[INFO] Classes:", class_to_idx)

    model = get_model(num_classes=len(class_to_idx))
    # Freeze backbone
    for name, pmt in model.named_parameters():
        if not name.startswith("fc."):
            pmt.requires_grad = False

    model.to(device)
    loss_fn = get_loss_fn(args.focal, args.label_smoothing, class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_acc = 0.0
    best_path = os.path.join(args.outdir, "best.pth")

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(args.epochs):
        # Unfreeze after freeze_epochs
        if epoch == args.freeze_epochs:
            for name, pmt in model.named_parameters():
                pmt.requires_grad = True
            # lower LR for backbone, higher for head
            head_params = [p for n,p in model.named_parameters() if n.startswith("fc.")]
            bb_params   = [p for n,p in model.named_parameters() if not n.startswith("fc.")]
            optimizer = torch.optim.AdamW([
                {"params": bb_params, "lr": args.lr * 0.25},
                {"params": head_params, "lr": args.lr}
            ], weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-epoch, eta_min=1e-6)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, scaler, args.mixup, args.cutmix)
        val_loss, val_acc = evaluate(model, val_loader, device, tta_flip=True)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"train_loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val_loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
              f"lr {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_acc": best_acc,
                "args": vars(args),
            }, best_path)
            print(f"[INFO] Saved new best to {best_path} ({best_acc*100:.2f}%)")

    print(f"[RESULT] Best val acc: {best_acc*100:.2f}%")

    if args.export_onnx and best_acc > 0:
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.randn(1,3,args.img_size,args.img_size)
        onnx_path = os.path.join(args.outdir, "model.onnx")
        torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}, opset_version=12)
        # write labels.json
        with open(os.path.join(args.outdir, "labels.json"), "w") as f:
            json.dump({v:k for k,v in ckpt["class_to_idx"].items()}, f, indent=2)
        print(f"[INFO] Exported ONNX to {onnx_path}")
        print("[INFO] Labels saved to labels.json")
    print("[DONE]")
if __name__ == "__main__":
    main()
