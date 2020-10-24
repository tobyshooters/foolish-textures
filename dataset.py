import os
import numpy as np
import torch
import cv2
import pickle as pkl
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append("../detectron2/projects/DensePose/")
from densepose.data.structures import DensePoseResult

def process_iuv(obj):
    img_id, instance_id = 0, 0  # Look at the first image and the first detected instance
    bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
    result_encoded = data[img_id]['pred_densepose'].results[instance_id]
    iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
    print(iuv_arr.shape)


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
        print(frame.shape)

        # Get IUV and bbox
        iuv_path = os.path.join("./data/", f + ".pkl")
        obj = pkl.load(open(iuv_path, "rb"))

        process_iuv(obj)
        return

        # IUV should be between [0, 1], with parts as first index and size of frame
        x1, y1, x2, y2 = (int(e) for e in obj["bbox"])
        _iuv = np.transpose(obj["iuv"], (1,2,0)) / np.array([1, 255, 255])
        _iuv = cv2.resize(_iuv, (x2-x1, y2-y1))
        _iuv = np.around(_iuv) # preserve indexes after bilinear sampling

        # Reformat into matrix the size of frame, with channel for each part
        H, W, _ = frame.shape
        iuv = np.zeros((H, W, 25, 2))

        for n in range(24):
            mask = (_iuv[:,:,0] == n+1)
            iuv[y1:y2, x1:x2, n][mask] = _iuv[:,:,1:][mask]

        iuv = np.transpose(iuv, (2,3,0,1))

        # To BCHW format
        frame = self.preprocess_frame(image=frame)["image"]

        return frame, iuv


if __name__ == "__main__":
    dset = Dataset()
    dset[0]
