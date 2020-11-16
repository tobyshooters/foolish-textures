import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

import lpips
from model import Pipeline
from dataset import Dataset
from utils import masked_l1

import wandb
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Dataset(num_parts=24)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = Pipeline(H=512, W=512, num_features=16, num_parts=24)
model.to(device)
# model.load_state_dict(torch.load("tmp.pth"))
model.train()

learning_rate = 1e-3
optimizer = torch.optim.Adam([
    # Apply increasing amount of regularization to finer layers
    {"params": model.atlas.layer1, 'weight_decay': 1e-2, 'lr': learning_rate},
    {"params": model.atlas.layer2, 'weight_decay': 1e-3, 'lr': learning_rate},
    {"params": model.atlas.layer3, 'weight_decay': 1e-4, 'lr': learning_rate},
    {"params": model.atlas.layer4, 'weight_decay': 0,    'lr': learning_rate},
])

lpips = lpips.LPIPS(net='alex').to("cuda")

wandb.init(project="foolish-textures", name="v1")

for e in range(30):
    print(f"EPOCH {e}")

    for i, sample in enumerate(tqdm(dataloader)):
        image, iuv, mask = sample

        # Inference
        tex, pred = model(iuv)

        # Composite
        pred = torch.where(mask > 0, pred, image)

        # Loss
        loss_rgb = masked_l1(tex[:, :3], image, mask.float())
        loss_l1 = masked_l1(pred, image, mask.float())
        loss_lpips = lpips(pred, image)
        loss = loss_rgb + loss_l1 + loss_lpips

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        output = pred[0].detach().cpu().numpy().transpose(1,2,0)
        cv2.imwrite(f"results/{e}_{i}.jpg", (255 * output).astype(np.uint8))

        wandb.log({
            "loss_rgb": loss_rgb.item(),
            "loss_l1": loss_l1.item(),
            "loss_lpips": loss_lpips.item(),
            "loss": loss.item(),
        })
        
torch.save(model.state_dict(), f"breatheless_v1.pth")
