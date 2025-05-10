import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from augmentation import get_train_transforms, get_test_transforms
from network import build_model
from visualize import plot_metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, count = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        count += labels.size(0)
    return total_loss/len(loader), correct/count

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            count += labels.size(0)
    return total_loss/len(loader), correct/count

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_ds = ImageFolder(f"{args.data_dir}/train", transform=get_train_transforms(tuple(args.image_size)))
    test_ds  = ImageFolder(f"{args.data_dir}/test",  transform=get_test_transforms(tuple(args.image_size)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = build_model(args.model, args.num_classes, args.pretrained).to(device)
    model = build_model(
        args.model,
        args.num_classes,
        args.pretrained,
        config_name=args.config_name
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, test_loader, criterion, device)
        t1 = time.time()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | Time: {t1-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"➡️ Saved best model to {save_path}")    
    plot_metrics(history, args.model, args.output_dir)
    print(f"Saved loss/accuracy plots to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',     type=str,   default='data')
    parser.add_argument('--model',        type=str,   required=True)
    parser.add_argument('--num-classes',  type=int,   default=6)
    parser.add_argument('--pretrained',   action='store_true')
    parser.add_argument('--image-size',   type=int,   nargs=2, default=(64,64))
    parser.add_argument('--batch-size',   type=int,   default=16)
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--output-dir',   type=str,   default='outputs/models')
    parser.add_argument(
        '--config-name',
        type=str,
        default='L',
        choices=['S','M','L'],
        help='Which EfficientNetV2 variant to use (S, M or L)'
    )

    args = parser.parse_args()
    train(args)
