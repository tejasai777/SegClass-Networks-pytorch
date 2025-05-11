import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_samples(model, loader, device, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0]
            img = imgs.cpu().numpy()[0].transpose(1,2,0)
            gt = masks.numpy()[0]
            fig, axs = plt.subplots(1,3,figsize=(12,4))
            axs[0].imshow(img); axs[0].set_title('Image'); axs[0].axis('off')
            axs[1].imshow(gt);  axs[1].set_title('GT Mask'); axs[1].axis('off')
            axs[2].imshow(pred);axs[2].set_title('Pred Mask'); axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{count}.png'))
            plt.close(fig)
            count += 1
            if count >= num_samples:
                break
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_classwise_curves(dice_hist, iou_hist, out_dir, class_names):
    """
    dice_hist: list of [C]-lists over epochs
    iou_hist:  list of [C]-lists over epochs
    class_names: list of length C
    """
    epochs = np.arange(1, len(dice_hist)+1)
    dice_arr = np.array(dice_hist)  # shape (E, C)
    iou_arr  = np.array(iou_hist)
    
    # Dice curves
    plt.figure()
    for c, name in enumerate(class_names):
        plt.plot(epochs, dice_arr[:,c], label=name)
    plt.title("Class-wise Dice over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Dice")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "classwise_dice.png"))
    plt.close()
    
    # IoU curves
    plt.figure()
    for c, name in enumerate(class_names):
        plt.plot(epochs, iou_arr[:,c], label=name)
    plt.title("Class-wise IoU over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("IoU")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "classwise_iou.png"))
    plt.close()

def save_plots(train_losses, val_losses, model_name, out_dir):
    epochs = range(1, len(train_losses)+1)
    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(out_dir, f'{model_name}_loss.png'))
    plt.close()
    # Accuracy plot (1 - loss as proxy)
    plt.figure()
    plt.plot(epochs, [1 - l for l in train_losses], label='Train Acc')
    plt.plot(epochs, [1 - l for l in val_losses], label='Val Acc')
    plt.title(f'{model_name} Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(out_dir, f'{model_name}_acc.png'))
    plt.close()
