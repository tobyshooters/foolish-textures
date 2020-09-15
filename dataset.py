import os
import numpy as np
import torch
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2

targets = { "uv": "image" }

augment = albu.Compose([
    albu.HorizontalFlip(p=0.5),
], additional_targets=targets)

preprocess = albu.Compose([
    albu.PadIfNeeded(768, 768),
    albu.RandomSizedCrop((256, 768), 512, 512),
    ToTensorV2(),
], additional_targets=targets)


class Dataset(torch.utils.data.Dataset):
    """
    Gives an image, uv-map, and mask.
    """

    def __init__(self, path):
        self.augment = augment
        self.preprocess = preprocess
        return

    def __len__(self):
        return 0

    def __getitem__(self, idx):

        image = np.zeros((600, 800, 3))
        uv = np.zeros((600, 800, 2))
        mask = (uv >= -1.0) and (uv <= 1.0)

        s = augment(image=image, uv=uv, mask=mask)
        image, uv, mask = s["image"], s["uv"], s["mask"]

        s = preprocess(image=image, uv=uv, mask=mask)
        image, uv, mask = s["image"], s["uv"], s["mask"]

        return image, uv, mask
