import os
import numpy as np
import torch
import cv2
import pickle as pkl
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append("../detectron2/projects/DensePose/")


class Dataset(torch.utils.data.Dataset):
    """
    Gives an image, uv-map, and mask.
    """

    def __init__(self):
        self.fs = ["le_corbusier"]

        self.preprocess_frame = albu.Compose([
            albu.Normalize(), 
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        f = self.fs[idx]

        # Read in frame
        frame_path = os.path.join("./data/", f + ".jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = albu.Normalize(frame)
        frame = albu.ToTensorV2(frame)

        # Get IUV and bbox
        data = pkl.load(open(os.path.join("./data/", f + ".pkl"), "rb"))

        result = data[0]['pred_densepose'][0]
        iuv_bbox = torch.cat([result.labels.unsqueeze(0), result.uv])

        bbox = data[0]['pred_boxes_XYXY'][0]
        x0, y0, x1, y1 = [x.item() for x in bbox]
        x, dx = int(x0), int(x1 - x0)
        y, dy = int(y0), int(y1 - y0)

        # Fit IUV to frame shape
        _, H, W = frame.shape
        iuv = torch.zeros((3, H, W))
        iuv[:, y:y+dy, x:x+dx] = iuv_bbox

        print(frame.shape, iuv.shape)

        return frame, iuv


if __name__ == "__main__":
    dset = Dataset()
    dset[0]
