import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# These must match your A.Normalize() settings
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(img_tensor):
    """
    Convert a normalized tensor back to an HxWx3 RGB image in [0,1]
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * STD) + MEAN
    img = np.clip(img, 0, 1)
    return img


def plot_samples(model, loader, device, save_dir, num_samples=5):
    """
    Saves examples with two rows of three panels each:
      Top row:    Original | GT Mask    | Pred Mask
      Bottom row: Original | GT Overlay | Pred Overlay
    Uses discrete class colormap consistent with class-wise plots, with background always black.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]

            # Un-normalize for display
            img = unnormalize(imgs[0])
            gt = masks.numpy()[0]

            # Determine vmin/vmax for colormap
            vmin, vmax = 0, int(max(gt.max(), preds.max()))
            num_classes = vmax + 1
            # Create a discrete colormap with background=black
            cmap_base = plt.get_cmap('tab10', num_classes)
            colors = list(cmap_base.colors)
            # set class 0 (background) to black
            colors[0] = (0, 0, 0, 1)
            cmap = ListedColormap(colors)

            # Create overlays (mask out background)
            gt_overlay = np.ma.masked_where(gt == 0, gt)
            pred_overlay = np.ma.masked_where(preds == 0, preds)

            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            # Top row: raw masks
            axs[0, 0].imshow(img)
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis("off")

            axs[0, 1].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[0, 1].set_title("GT Mask")
            axs[0, 1].axis("off")

            axs[0, 2].imshow(preds, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[0, 2].set_title("Pred Mask")
            axs[0, 2].axis("off")

            # Bottom row: overlays
            axs[1, 0].imshow(img)
            axs[1, 0].set_title("Original Image")
            axs[1, 0].axis("off")

            axs[1, 1].imshow(img)
            axs[1, 1].imshow(gt_overlay, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
            axs[1, 1].set_title("GT Overlay")
            axs[1, 1].axis("off")

            axs[1, 2].imshow(img)
            axs[1, 2].imshow(pred_overlay, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
            axs[1, 2].set_title("Pred Overlay")
            axs[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{count}.png"))
            plt.close(fig)

            count += 1
            if count >= num_samples:
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
