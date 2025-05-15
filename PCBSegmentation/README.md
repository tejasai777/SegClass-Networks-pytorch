# PCB Segmentation Project

## Structure
```
project_root/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── models/
│   ├── __init__.py
│   ├── attunet.py
│   ├── unet_mobilenet.py
│   ├── unet_vgg16.py
|          .
|          .
|          .
│   ├── unetpp.py
│   └── unet3plus.py
├── augmentation.py
├── network.py
├── train.py
├── eval.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Prepare your data under `data/train/images`, `data/train/masks`, `data/val/images`, `data/val/masks`.
3. Run training:
   ```
   python train.py --model unet3plus --data-dir data --output-dir outputs
   ```
4. Find best model, samples, metrics, and plots under `outputs/{model}`.
