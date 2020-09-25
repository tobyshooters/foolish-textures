import os
import numpy as np
import torch
import cv2
import pickle as pkl
import albumentations as albu
from albumentations.pytorch import ToTensorV2

targets = { "uv": "image" }

class Dataset(torch.utils.data.Dataset):
    """
    Gives an image, uv-map, and mask.
    """

    def __init__(self):
        self.fs = [f.split(".")[0] for f in os.listdir("../data/moi_et_toi_frames")]

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        f = self.fs[idx]

        frame_path = os.path.join("../data/moi_et_toi_frames", f + ".jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        iuv_path = os.path.join("../data/moi_et_toi_iuvs", f + ".pkl")
        obj = pkl.load(open(iuv_path, "rb"))

        iuv = np.zeros(frame.shape)
        x1 = int(obj["bbox"][0]) 
        x2 = int(obj["bbox"][2]) 
        y1 = int(obj["bbox"][1]) 
        y2 = int(obj["bbox"][3])

        _iuv = cv2.resize(obj["iuv"].transpose(1,2,0), (x2-x1, y2-y1))
        iuv[y1:y2, x1:x2] = _iuv

        frame = frame.transpose(2,0,1)
        iuv = iuv.transpose(2,0,1)

        return frame, iuv
