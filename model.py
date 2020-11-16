# Based on: https://github.com/SSRSGJYD/NeuralTexture

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# Neural Texture
##############################################################################

class Texture(nn.Module):
    """
    Laplacian pyramid of textures, use forward to sample features
    """

    def __init__(self, N, H, W):
        super(Texture, self).__init__()
        self.layer1 = nn.Parameter(torch.zeros((1, N, H,    W   )))
        self.layer2 = nn.Parameter(torch.zeros((1, N, H//2, W//2)))
        self.layer3 = nn.Parameter(torch.zeros((1, N, H//4, W//4)))
        self.layer4 = nn.Parameter(torch.zeros((1, N, H//8, W//8)))

    def forward(self, uv):
        # Expects uv in [-1, 1]
        # Padding mode is zero, so numbers outside of (-1, 1) are set to 0
        B, _, H, W = uv.shape # (B, 2, H, W)

        # Repeat to do batches in parallel
        y1 = F.grid_sample(self.layer1.repeat(B, 1, 1, 1), uv, align_corners=True)
        y2 = F.grid_sample(self.layer2.repeat(B, 1, 1, 1), uv, align_corners=True)
        y3 = F.grid_sample(self.layer3.repeat(B, 1, 1, 1), uv, align_corners=True)
        y4 = F.grid_sample(self.layer4.repeat(B, 1, 1, 1), uv, align_corners=True)

        y = y1 + y2 + y3 + y4
        return y


class Atlas(nn.Module):

    def __init__(self, H, W, num_features, num_parts):
        super(Atlas, self).__init__()
        self.num_features = num_features
        self.num_parts = num_parts

        self.textures = nn.ModuleList([
            Texture(self.num_features, H, W) 
            for _ in range(num_parts)
        ])

        # Group layers together to apply same regularization
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()

        for i in range(num_parts):
            self.layer1.append(self.textures[i].layer1)
            self.layer2.append(self.textures[i].layer2)
            self.layer3.append(self.textures[i].layer3)
            self.layer4.append(self.textures[i].layer4)

    def forward(self, iuv):
        # Expects iuv of shape (B, num_parts, H, W, _)
        B, _, H, W, _ = iuv.shape
        features = torch.zeros((B, self.num_features, H, W)).to("cuda")

        for i in range(self.num_parts):
            samples = self.textures[i](iuv[:, i])
            features += samples

        return features


##############################################################################
# U-net: kept simple
##############################################################################

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, final=False):
        super(Up, self).__init__()
        self.concat = concat
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.nlin = nn.Tanh() if final else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2):
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat((x2, x1), dim=1)

        return self.nlin(self.norm(self.conv(x1)))


class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNet, self).__init__()
        self.down1 = Down(in_c, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(512, 512, output_pad=1, concat=False)
        self.up2 = Up(1024, 512)
        self.up3 = Up(768, 256)
        self.up4 = Up(384, 128)
        self.up5 = Up(192, out_c, final=True)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x


##############################################################################
# Pipeline
##############################################################################

class Pipeline(nn.Module):
    """
    Pass uv map through model.
    Currently ignores camera extrinsics.
    """

    def __init__(self, H, W, num_features, num_parts=24):
        super(Pipeline, self).__init__()
        self.atlas = Atlas(H, W, num_features, num_parts)
        self.unet = UNet(num_features, 3)

    def forward(self, iuv):
        x = self.atlas(iuv)
        y = self.unet(x)
        return x, y
