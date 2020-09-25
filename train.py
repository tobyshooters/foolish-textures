import torch
import wandb
from tqdm import tqdm

from model import Pipeline
from dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Dataset("./data")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = Pipeline(W=1024, H=1024, num_features=16)
model.to(device)
model.train()

learning_rate = 1e-3
optimizer = torch.optim.Adam([
    # Apply increasing amount of regularization to finer layers
    {"params": model.texture.layer1, 'weight_decay': 1e-2, 'lr': learning_rate},
    {"params": model.texture.layer2, 'weight_decay': 1e-3, 'lr': learning_rate},
    {"params": model.texture.layer3, 'weight_decay': 1e-4, 'lr': learning_rate},
    {"params": model.texture.layer4, 'weight_decay': 0,    'lr': learning_rate},
])
criterion = nn.L1Loss()

wandb.init(project="foolish-textures")

for e in range(10):
    for sample in tqdm(dataloader):
        image, iuv = sample
        sampled_texture, pred = model(iuv)

        loss_rgb = criterion(sampled_texture[:,:3], image)
        loss_reconstruct = criterion(pred, image)
        loss = loss_rgb + loss_reconstruct

        wandb.log({
            "loss": loss.item(),
            "loss_rgb": loss_rgb.item(),
            "loss_reconstruct": loss_reconstruct.item(),
        })
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save(model, f"neural_textures_{e}.pth")
