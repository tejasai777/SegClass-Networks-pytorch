import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Normalization constants 
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# class‐specific RGBA colors in [0,1]
# class 0 = background, classes 1–3 = your PCB components
CLASS_COLORS = [
    (0.0, 0.0, 0.0, 1.0),   # 0 = black (background)
    (1.0, 0.8, 0.0, 1.0),   # 1 = yellow (e.g. resistors)
    (0.0, 0.5, 1.0, 1.0),   # 2 = cyan   (e.g. capacitors)
    (1.0, 0.0, 1.0, 1.0),   # 3 = magenta(e.g. ICs)
]

# Create a Matplotlib colormap + norm so each label gets the exact color
cmap = ListedColormap(CLASS_COLORS)
norm = BoundaryNorm(boundaries=np.arange(len(CLASS_COLORS) + 1) - 0.5,
                    ncolors=len(CLASS_COLORS))

def unnormalize(img_tensor):
    """
    Converts a normalized tensor (C,H,W) float back to an HxWx3 RGB image in [0,1].
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * STD) + MEAN
    return np.clip(img, 0, 1)

def plot_samples(model, loader, device, save_dir, num_samples=5):
    """
    Saves num_samples examples, skipping the first 2*num_samples so that you get
    the 'third batch' (indices 10–14 if num_samples=5).

"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    skip = 2 * num_samples      # skip first 10 if num_samples=5
    count = 0                   # batches we've iterated through
    saved = 0                   # how many we've saved

    with torch.no_grad():
        for imgs, masks in loader:
            # skip until we've passed 'skip' samples
            if count < skip:
                count += 1
                continue

            imgs = imgs.to(device)
            outputs = model(imgs)
            # upsample to match mask resolution if needed
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )
            preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]

            # prepare data for plotting
            img = unnormalize(imgs[0])
            gt  = masks.numpy()[0]
            # mask out background for overlay
            gt_ov   = np.ma.masked_equal(gt,   0)
            pred_ov = np.ma.masked_equal(preds, 0)

            # plot
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))

            # Top row: original images & masks
            axs[0, 0].imshow(img);           axs[0, 0].set_title("Original Image"); axs[0, 0].axis("off")
            axs[0, 1].imshow(gt,   cmap=cmap, norm=norm); axs[0, 1].set_title("GT Mask");       axs[0, 1].axis("off")
            axs[0, 2].imshow(preds,cmap=cmap, norm=norm); axs[0, 2].set_title("Pred Mask");     axs[0, 2].axis("off")
            # Bottom row: overlays
            axs[1, 0].imshow(img); axs[1, 0].set_title("Original Image"); axs[1, 0].axis("off")
            axs[1, 1].imshow(img); axs[1, 1].imshow(gt_ov,   cmap=cmap, norm=norm, alpha=0.6)
            axs[1, 1].set_title("GT Overlay"); axs[1, 1].axis("off")
            axs[1, 2].imshow(img); axs[1, 2].imshow(pred_ov, cmap=cmap, norm=norm, alpha=0.6)
            axs[1, 2].set_title("Pred Overlay"); axs[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{count}.png"))
            plt.close(fig)

            saved += 1
            count += 1
            if saved >= num_samples:
                break


def save_plots(train_losses, val_losses, model_name, out_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'{model_name}_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, [1 - l for l in train_losses], label='Train Acc')
    plt.plot(epochs, [1 - l for l in val_losses], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'{model_name}_acc.png'))
    plt.close()


def plot_classwise_curves(dice_hist, iou_hist, out_dir, class_names):
    epochs = range(1, len(dice_hist) + 1)
    dice_arr = np.array(dice_hist)
    iou_arr = np.array(iou_hist)

    # Dice
    plt.figure()
    for idx, name in enumerate(class_names):
        plt.plot(epochs, dice_arr[:, idx], label=name)
    plt.title('Class-wise Dice over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'classwise_dice.png'))
    plt.close()

    # IoU
    plt.figure()
    for idx, name in enumerate(class_names):
        plt.plot(epochs, iou_arr[:, idx], label=name)
    plt.title('Class-wise IoU over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'classwise_iou.png'))
    plt.close()
