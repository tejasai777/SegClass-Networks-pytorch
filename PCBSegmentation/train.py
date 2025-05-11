import os
import cv2
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from augmentation import get_train_transform, get_val_transform
from network import build_seg_model
from segmentation_models_pytorch.losses import DiceLoss
from visualize import plot_samples, save_plots, plot_classwise_curves
from eval import evaluate_metrics

class PCBDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask.long()


def get_paths(img_dir, mask_dir):
    imgs = sorted([str(p) for p in Path(img_dir).glob("*.png")])
    masks = sorted([str(p) for p in Path(mask_dir).glob("*.png")])
    return imgs, masks


def train_fn(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_fn(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet3plus','unetpp','unet_vgg16','attunet','unet_mobilenet'])
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_imgs, train_masks = get_paths(os.path.join(args.data_dir,'train','images'),
                                        os.path.join(args.data_dir,'train','masks'))
    val_imgs, val_masks     = get_paths(os.path.join(args.data_dir,'val','images'),
                                        os.path.join(args.data_dir,'val','masks'))

    train_loader = DataLoader(PCBDataset(train_imgs, train_masks,
                                         transform=get_train_transform(args.img_size)),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(PCBDataset(val_imgs, val_masks,
                                         transform=get_val_transform(args.img_size)),
                              batch_size=1, shuffle=False, num_workers=2)

    model = build_seg_model(args.model, img_ch=3, num_classes=4).to(device)
    ce = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(mode='multiclass')
    criterion = lambda p, t: 0.5 * ce(p, t) + 0.5 * dice_loss(p, t)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    classwise_dice, classwise_iou = [], []

    for epoch in range(1, args.epochs + 1):
        # Training
        tr_loss = train_fn(train_loader, model, criterion, optimizer, device)
        vl_loss = eval_fn(val_loader, model, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}")

        # Save best and early stop
        out_dir = os.path.join(args.output_dir, args.model)
        os.makedirs(out_dir, exist_ok=True)
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_{args.model}.pth"))
            print(f"✔️ Best model saved at epoch {epoch}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("⏹️ Early stopping.")
                break

        # Evaluate per-class metrics
        metrics = evaluate_metrics(model, val_loader, device, num_classes=4)
        classwise_dice.append(metrics['dice_score'].tolist())
        classwise_iou.append(metrics['iou'].tolist())

    # After training, save and plot class-wise curves
    with open(os.path.join(out_dir, "classwise_dice.json"), "w") as f:
        json.dump(classwise_dice, f)
    with open(os.path.join(out_dir, "classwise_iou.json"), "w") as f:
        json.dump(classwise_iou, f)
    plot_classwise_curves(classwise_dice, classwise_iou, out_dir,
                          class_names=['BG', 'Cap', 'Res', 'IC'])

    # Visualize first 5 samples
    plot_samples(model, val_loader, device, os.path.join(out_dir, 'samples'), num_samples=5)

    # Final evaluation metrics
    final_metrics = evaluate_metrics(model, val_loader, device, num_classes=4)
    print("Evaluation Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    # Model params and size
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024**2)
    print(f"Total params: {total_params:,}")
    print(f"Model size: {size_mb:.2f} MB")

    # Save loss & accuracy plots
    save_plots(train_losses, val_losses, args.model, out_dir)

if __name__ == "__main__":
    main()