# ViT Replication
This folder contains a replication of the vision transformer from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021).

The training pipeline currently uses `torchvision.datasets.CIFAR10`, resized to `224x224` for the ViT input.

## Run Training Loop
Starting from repo root:
```
$ cd vit
$ python loop.py
```

If the dataset is not found, it will be downloaded to `data/`.
