# Saves loss and accuracy graghd from train history 

import os
import matplotlib.pyplot as plt

def plot_metrics(history, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy.png"))
    plt.close()
