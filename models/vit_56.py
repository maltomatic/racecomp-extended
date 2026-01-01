import torch
import sys
import os
import models.apple_models
from torch import nn
import torchvision
from torchsummary import summary
from timm.models import create_model
from models.apple_models.modules.mobileone import reparameterize_model

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class up_block(nn.Module): # increase shape by factor of scale
    def __init__(self, in_ch, out_ch, scale = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * (scale ** 2), kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(scale)
        )
    def forward(self, x):
        return self.up(x)
    
class VitUpscaler(nn.Module):
    def __init__(self, base_model = "fastvit_ma36", factor = 2, input_shape = (3, 112, 112), out_shape = (3, 224, 224)):
        super().__init__()
        # Load the base FastViT model
        self.encoder = create_model(base_model)
        chkp = torch.load('models/apple_models/fastvit_ma36.pth.tar')
        state_dict = chkp['state_dict']
        self.encoder.load_state_dict(state_dict)

        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()

        self.prep = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1)
        self.large = False if input_shape == (3, 56, 56) else True
        # print("Large mode: ", self.large)

        self.entry = nn.Sequential(
            self.encoder.patch_embed,
            nn.Identity()
        )
        self.enc1 = nn.Sequential(
            self.encoder.network[0],
            nn.Identity()
        )
        self.enc2 = nn.Sequential(
            self.encoder.network[1],
            self.encoder.network[2],
            nn.Identity()
        )
        self.enc3 = nn.Sequential(
            self.encoder.network[3],
            self.encoder.network[4],
            nn.Identity()
        )
        self.enc4 = nn.Sequential(
            self.encoder.network[5],
            self.encoder.network[6],
            nn.Identity()
        )
        self.out = nn.Sequential(
            self.encoder.network[7],
            self.encoder.conv_exp,
            nn.Identity()
        )
        self.decHead = conv_block(1216, 608)

        self.decHeadConv = conv_block(1216, 1216)
        
        self.dec1 = up_block(1216, 304)
        self.d1conv = conv_block(608, 304)

        self.dec2 = up_block(304, 152)
        self.d2conv = conv_block(304, 152)

        self.dec3 = up_block(152, 76)
        self.d3conv = conv_block(152, 76)

        self.dec4 = conv_block(76, 76)
        self.d4conv = conv_block(152, 76)

        #76, 28, 28 --> 3, 112, 112
        self.dec5 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(19, 3*(2**2), kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1)
        )
        self.dec5conv = conv_block(6, 3)

        self.decout = up_block(3, 3)
        if(not self.large):
            self.large_decout = up_block(3, 3)
        self.out_conv = nn.Conv2d(3, out_shape[0], kernel_size=3, padding=1, stride=1)

        # self.dec5
    def center_crop(self, tensor, target_size):
        _, _, h, w = tensor.size()
        _, _, th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[:, :, i:i+th, j:j+tw]
    
    def forward(self, x):
        xp = self.prep(x)

        x1 = self.entry(x)
        # print("After entry: ", x1.shape)
        x2 = self.enc1(x1)
        # print("After enc1: ", x2.shape)
        x3 = self.enc2(x2)
        # print("After enc2: ", x3.shape)
        x4 = self.enc3(x3)
        # print("After enc3: ", x4.shape)

        if self.large:
            x5 = self.enc4(x4)
            # print("After enc4: ", x5.shape)
            x6 = self.out(x5)
        
            # print("Final: ", x6.shape)
            y1 = self.decHead(x6)
            y1 = torch.cat([y1, x5], dim=1)
            y1 = self.decHeadConv(y1)
            # print("After decHead: ", y1.shape)

            y2 = self.dec1(y1)
            y2 = self.center_crop(y2, x4.size())
            y2 = torch.cat([y2, x4], dim=1)
            y2 = self.d1conv(y2)
            # print("After dec1: ", y2.shape)
        else:
            y2 = x4

        y3 = self.dec2(y2)
        y3 = self.center_crop(y3, x3.size())
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.d2conv(y3)
        # print("After dec2: ", y3.shape)

        y4 = self.dec3(y3)
        y4 = torch.cat([y4, x2], dim=1)
        y4 = self.d3conv(y4)
        # print("After dec3: ", y4.shape)

        y5 = self.dec4(y4)
        y5 = torch.cat([y5, x1], dim=1)
        y5 = self.d4conv(y5)
        # print("After dec4: ", y5.shape)

        y6 = self.dec5(y5)
        y6 = torch.cat([y6, xp], dim=1)
        y6 = self.dec5conv(y6)
        # print("After dec5: ", y6.shape)

        y = self.decout(y6)
        if(not self.large):
            y = self.large_decout(y)
        y = self.out_conv(y)
        # print("Output: ", y.shape)

        return y

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Availability: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = VitUpscaler(input_shape = (3, 224, 224)).to(device)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x.to(device))
    print("Input:", x.shape, "Output:", y.shape)