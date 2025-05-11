# PCB-Component-Classification-Networks-pytorch

## Structure

```
project_root/
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── dinpnet.py
│   ├── efficientnetv2.py
│   ├── mobilenetv3.py
│   ├── pcbclassifier.py
│   └── convnext_tiny.py
├── augmentation.py
├── network.py
├── train.py
├── visualize.py
├── requirements.txt
└── README.md
```

Install dependencies:

```
pip install -r requirements.txt
```

Train:

```
python train.py --model mobilenetv3
```
