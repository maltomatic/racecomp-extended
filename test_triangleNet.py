import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from timm.models import create_model
from models.apple_models.modules.mobileone import reparameterize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class downsample_maker(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor
    # shrink image by factor using bilinear interpolation
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=1/self.factor, mode='bilinear', align_corners=False) 

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            # nn.BatchNorm2d(out_ch,),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            # nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class interpolate_resize(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class learner_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class TriangleNet(nn.Module):
    def __init__(self, px_shuffle = False, px_shuffle_interpolate = True, extra_bridge = False,
                 training = False, base_model = "fastvit_ma36"):
        super().__init__()
        self.training = training
        self.extra_bridge = extra_bridge
        
        self.vit_learner = create_model(base_model)
        chkp = torch.load('models/apple_models/fastvit_ma36.pth.tar')
        state_dict = chkp['state_dict']
        self.vit_learner.load_state_dict(state_dict)
        
        self.fake_upscale = nn.Sequential(
            # bilinear interpolate to 224 224 and conv once
            interpolate_resize(scale_factor=2),
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1),
            interpolate_resize(scale_factor=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True)
        )
        self.false_input = downsample_maker(factor=4)
        self.le1 = self.vit_learner.patch_embed[0] #76, 112
        self.le1_c = learner_conv(76, 64)
        self.le2 = nn.Sequential(
            self.vit_learner.patch_embed[1], self.vit_learner.patch_embed[2],
            self.vit_learner.network[0],
            learner_conv(76, 256)
        ) #256, 56
        self.learner = nn.Sequential(self.false_input, self.fake_upscale, self.le1, self.le1_c, self.le2)

        self.resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True)
        )
        self.entry = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.enc1 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.enc2 = self.resnet.layer2
        self.enc3 = self.resnet.layer3
        self.enc4 = self.resnet.layer4
        self.teacher = nn.Sequential(
            self.prep_layer,
            self.entry,
            self.enc1
        )
        self.teacher_upscaler = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True),
            interpolate_resize(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True)
        )

        if(not px_shuffle):
            # NN interpolate to 2x then conv same size w/ channel redux
            self.dec4 = nn.Sequential(
                interpolate_resize(),
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.dec4 = nn.Sequential(
                nn.Conv2d(2048, 1024*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up4 = conv_block(2048, 1024)
        
        if(not px_shuffle and not px_shuffle_interpolate):
            self.dec3 = nn.Sequential(
                interpolate_resize(),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.dec3 = nn.Sequential(
                nn.Conv2d(1024, 512*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up3 = conv_block(1024, 512)
        
        if(not px_shuffle):
            self.dec2 = nn.Sequential(
                interpolate_resize(),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.dec2 = nn.Sequential(
                nn.Conv2d(512, 256*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up2 = conv_block(512, 256)
        
        if(not px_shuffle and not px_shuffle_interpolate):
            self.dec1 = nn.Sequential(
                interpolate_resize(),
                nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.dec1 = nn.Sequential(
                nn.Conv2d(256, 64*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up1 = conv_block(128, 64)

        if(not px_shuffle):
            self.dec0 = nn.Sequential(
                interpolate_resize(),
                nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.dec0 = nn.Sequential(
                nn.Conv2d(64, 16*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(2)
            )
        self.conv_up0 = conv_block(32, 16)

        self.bufferPX = nn.Sequential(
            # pixel_shuffle then downsize back to 112 112
            nn.Conv2d(16, 16*(2**2), kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=4, padding=1, stride=2),
        )
        # interpolation up to double size then bilinear back down
        self.up_down_exit = nn.Sequential(
            interpolate_resize(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1),
            downsample_maker(factor=2)
        )
        

    def center_crop(self, tensor, target_size):
        _, _, h, w = tensor.size()
        _, _, th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[:, :, i:i+th, j:j+tw]

    def forward(self, x, stage = 5):  # stage 1-4 for training, implicit stage 5 is eval forward pass
        #encoder
        if(stage != 5):
            #### TODO: proccess according to stages
            if(stage != 4):
                x_prep = self.prep_layer(x)
                # print("x_prepped shape:", x_prep.shape) # 16, 224 /// 56 --> concat d0 
                x1 = self.entry(x)
                # print("x1 shape:", x1.shape)    # 64, 112 /// 28 --> concat d1
                x2 = self.enc1(x1)
                # print("x2 shape:", x2.shape)    # 256, 56 /// 14 --> concat d2
            
            if(stage == 1):
                # fine tune teacher
                out = self.teacher_upscaler(x2)
                return out
            elif(stage == 2):
                # train learner convs
                x = self.false_input(x)
                lea224 = self.fake_upscale(x)
                _l1 = self.le1(interpolate_resize(scale_factor=4)(x))
                lea112 = self.le1_c(_l1)
                lea56 = self.le2(_l1)
                return {"x_prep": x_prep, "x1": x1, "x2": x2, "lea224": lea224, "lea112": lea112, "lea56": lea56}
            elif(stage == 3):
                # train the rest of decoder with teacher outputs
                pass
            elif(stage == 4):
                # fine tune decoder against learner
                x = self.false_input(x)
                x_prep = self.fake_upscale(x)
                _x1 = self.le1(interpolate_resize(scale_factor=4)(x))
                x1 = self.le1_c(_x1)
                x2 = self.le2(_x1)
        else:
            x_prep = self.fake_upscale(x)
            # print("x_prepped shape:", x_prep.shape) # 16, 224
            _x1 = self.le1(interpolate_resize(scale_factor=4)(x))
            x1 = self.le1_c(_x1)
            # print("x1 shape:", _x1.shape)    # 64, 112
            x2 = self.le2(_x1)
            # print("x2 shape:", x2.shape)    # 256, 56
        x3 = self.enc2(x2)
        # print("x3 shape:", x3.shape)    # 512, 28 /// 7 --> concat d3
        x4 = self.enc3(x3)
        # print("x4 shape:", x4.shape)    # 1024, 14 /// 4 --> bottleneck
        
        if(self.extra_bridge):
            x5 = self.enc4(x4)
            # print("x5 shape:", x5.shape)

        # print("---- Decoder shapes ----")

        #decode with encode outputs
            d4 = self.dec4(x5)
            d4 = self.center_crop(d4, x4.size())
            # print("d4 shape:", d4.shape)
            # print("x4 shape:", x4.shape)
            d4 = torch.cat([d4, x4], dim=1)
            # print("    d4 concatted shape:", d4.shape)
            d4 = self.conv_up4(d4)
        else:
            d4 = x4

        d3 = self.dec3(d4)
        d3 = self.center_crop(d3, x3.size())
        # print("d3 shape:", d3.shape)    # 512, 28 /// 7
        # print("x3 shape:", x3.shape)
        d3 = torch.cat([d3, x3], dim=1)
        # print("    d3 concatted shape:", d3.shape)  # 1024, 28 /// 7
        d3 = self.conv_up3(d3)
        # print("    d3 after conv shape:", d3.shape)  # 512, 28 /// 7

        d2 = self.dec2(d3)
        d2 = self.center_crop(d2, x2.size())
        # print("d2 shape:", d2.shape)    # 256, 56 /// 14
        # print("x2 shape:", x2.shape)
        d2 = torch.cat([d2, x2], dim=1)
        # print("    d2 concatted shape:", d2.shape)  # 512, 56 /// 14
        d2 = self.conv_up2(d2)
        # print("    d2 after conv shape:", d2.shape)  # 256, 56 /// 14
        
        d1 = self.dec1(d2)
        d1 = self.center_crop(d1, x1.size())
        # print("d1 shape:", d1.shape)    # 64, 112 /// 28
        # print("x1 shape:", x1.shape)
        d1 = torch.cat([d1, x1], dim=1)
        # print("    d1 concatted shape:", d1.shape)  # 128, 112 /// 28
        d1 = self.conv_up1(d1)
        # print("    d1 after conv shape:", d1.shape) # 64, 112 /// 28

        d0 = self.dec0(d1)
        # print("d0 shape:", d0.shape)    # 16, 224 /// 56
        d0 = torch.cat([d0, x_prep], dim=1)
        # print("    d0 concatted shape:", d0.shape)  # 32, 224 /// 56
        d0 = self.conv_up0(d0)
        # print("    d0 after conv shape:", d0.shape) # 16, 224 /// 56
        
        d0 = self.bufferPX(d0)
        # print("    d0 after bufferPX shape:", d0.shape) # 16, 224 /// 224
        out = self.up_down_exit(d0)
        # print("up_down_exit out shape:", out.shape) # 3, 448       
        return out
# Example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Availability: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = TriangleNet(px_shuffle=False, px_shuffle_interpolate=True, training = True).to(device)
    # summary(model, (3, 56, 56))
    for s in range(1, 5):
        x = torch.randn(1, 3, 224, 224)
        print(f"\n--- Stage {s} ---")
        y = model(x.to(device), stage=s)
        print("Input:", x.shape, "Output:", y.shape if s != 2 else {k: v.shape for k, v in y.items()})
    model.training = False
    print("\n--- Evaluation Pass ---")
    x = torch.randn(1, 3, 56, 56)
    y = model(x.to(device))
    print("Input:", x.shape, "Output:", y.shape)