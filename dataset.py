import os
import numpy as np
import torch
import cv2
import pickle as pkl
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class Dataset(torch.utils.data.Dataset):
    """
    Gives an image, uv-map, and mask.
    """

    def __init__(self):
        self.fs = [f.split(".")[0] for f in os.listdir("../data/moi_et_toi_frames")]

        self.preprocess_frame = albu.Compose([
            albu.Normalize(), 
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        f = self.fs[idx]

        # Read in frame
        frame_path = os.path.join("../data/moi_et_toi_frames", f + ".jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get IUV and bbox
        iuv_path = os.path.join("../data/moi_et_toi_iuvs", f + ".pkl")
        obj = pkl.load(open(iuv_path, "rb"))

        # IUV should be between [0, 1], with parts as first index and size of frame
        x1, y1, x2, y2 = (int(e) for e in obj["bbox"])
        _iuv = np.transpose(obj["iuv"], (1,2,0)) / np.array([1, 255, 255])
        _iuv = cv2.resize(_iuv, (x2-x1, y2-y1))
        _iuv = np.around(_iuv) # preserve indexes

        # Reformat into matrix the size of frame, with channel for each part
        H, W, _ = frame.shape
        iuv = np.zeros((H, W, 25, 2))

        for n in range(24):
            mask = (_iuv[:,:,0] == n+1)
            iuv[y1:y2, x1:x2, n][mask] = _iuv[:,:,1:][mask]

        iuv = np.transpose(iuv, (2,3,0,1))

        # Validates that IUV channel split worked
        # Need to make iuv have (n+1) channels so that argmax isolates background correctly
        # parts = np.argmax(iuv[:,:,:,0][y1:y2, x1:x2], axis=2)
        # _h, _w = parts.shape
        # for i in range(_h):
        #     for j in range(_w):
        #         if parts[i,j] != _iuv[i,j,0] and _iuv[i,j,1] > 0 and _iuv[i,j,2] > 0:
        #             print(parts[i,j], _iuv[i,j,0])

        # To BCHW format
        frame = self.preprocess_frame(image=frame)["image"]

        return frame, iuv
