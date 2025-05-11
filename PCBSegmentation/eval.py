import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_metrics(model, loader, device, num_classes=4):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu().numpy().ravel()
            targ = masks.numpy().ravel()
            preds.append(pred)
            targets.append(targ)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    precision = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
    recall = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm) + 1e-8)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    dice = (2 * precision * recall) / (precision + recall + 1e-8)
    return {
        'pixel_accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice_score': dice
    }
