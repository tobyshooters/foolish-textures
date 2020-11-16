import os
import numpy as np
import torch
import cv2
import pickle as pkl

import sys
sys.path.append("../detectron2/projects/DensePose/")

from utils import bbox_area, bbox_iou, pad_by_divisor


class Dataset(torch.utils.data.Dataset):
    """
    Gives an image, uv-map, and mask.
    """

    def __init__(self, num_parts=24):
        self.default = -3
        self.num_parts = num_parts

        self.fs = sorted(os.listdir("data/breatheless"))
        self.data = {}

        # Sort by filename
        pkl_data = pkl.load(open("data/breatheless.pkl", "rb"))
        pkl_data = sorted(pkl_data, key=lambda d: d["file_name"])

        # Choose the object with largest bounding box
        best = {"i": 0, "area": 0}
        for i, bbox in enumerate(pkl_data[0]["pred_boxes_XYXY"]):
            x0, y0, x1, y1 = [e.item() for e in bbox]
            area = bbox_area(x0, y0, x1, y1)
            if area > best["area"]:
                best = {"i": i, "area": area}

        # Tracking using hungarian method
        prev_bbox = pkl_data[0]["pred_boxes_XYXY"][best["i"]]
        for d in pkl_data:

            best = {"i": 0, "iou": 0}
            for i, bbox in enumerate(d["pred_boxes_XYXY"]):
                iou = bbox_iou(prev_bbox, bbox)
                if iou > best["iou"]:
                    best = {"i": i, "iou": iou}

            fname = d["file_name"].split("/")[-1]
            self.data[fname] = {
                "pred": d["pred_densepose"][best["i"]],
                "bbox": d["pred_boxes_XYXY"][best["i"]]
            }

            prev_bbox = d["pred_boxes_XYXY"][best["i"]]


    def __len__(self):
        return len(self.fs)


    def __getitem__(self, idx):
        f = self.fs[idx]

        # Read in frame
        frame_path = os.path.join("./data/breatheless/", f)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.FloatTensor(frame)
        frame /= 255.
        frame = frame.permute(2, 0, 1)
        frame = pad_by_divisor(frame, div=32)
        frame = frame.to("cuda")

        # Get IUV and bbox
        result = self.data[f]['pred']
        uv = 2 * result.uv -1 # [0, 1] => [-1, 1]
        uv = uv.permute(1,2,0)
        uv = uv.to("cuda")

        bbox = self.data[f]['bbox']
        x0, y0, x1, y1 = [x.item() for x in bbox]
        x, dx = int(x0), int(x1 - x0)
        y, dy = int(y0), int(y1 - y0)

        # Fit IUV to frame shape, split into body parts
        _, H, W = frame.shape
        iuv = self.default * torch.ones((self.num_parts, H, W, 2)).to("cuda")

        for n in range(self.num_parts):
            part = (result.labels == (n+1))
            iuv[n, y:y+dy, x:x+dx][part] = uv[part]

        iuv = iuv.to("cuda")

        # Get IUV mask
        mask = torch.zeros((H, W)).to("cuda")
        mask[y:y+dy, x:x+dx] = (result.labels > 0)

        return frame, iuv, mask


if __name__ == "__main__":
    dset = Dataset()
    loader = torch.utils.data.DataLoader(dset, shuffle=False, batch_size=1)

    for t, (frame, iuv) in enumerate(loader):
        frame = frame[0].permute(1,2,0).cpu().numpy()
        iuv = iuv[0].cpu().numpy()

        for part in iuv:
            frame[part[:,:,0] >= -1] = np.array([0, 0, 1])

        cv2.imwrite(f"results/{t:05d}_data.jpg", (255 * frame).astype(np.uint8))

        # frame = frame[0].cpu().numpy().transpose(1,2,0)
        # iuv = iuv[0].cpu().numpy().transpose(1,2,0)
        # iuv[1:] *= 255
        # cv2.imwrite(f"{t}_img.jpg", (255 * frame).astype(np.uint8))
        # cv2.imwrite(f"{t}_iuv.jpg", iuv)
