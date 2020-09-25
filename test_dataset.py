import torch
import cv2
from dataset import Dataset

dset = Dataset()
loader = torch.utils.data.DataLoader(dset, shuffle=True, batch_size=1)

for t, (frame, iuv) in enumerate(loader):

    if False:
        frame = frame[0].numpy().transpose(1,2,0)
        iuv = iuv[0].numpy().transpose(1,2,0)
        cv2.imwrite("tmp_img.jpg", frame)
        cv2.imwrite("tmp_iuv.jpg", iuv)

    break
