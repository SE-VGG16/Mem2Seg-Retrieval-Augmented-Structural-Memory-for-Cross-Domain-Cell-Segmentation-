import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNetLite(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)

        self.mid = conv_block(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.out = nn.Conv2d(base, base, 1)

    def forward(self, x):
        e1 = self.enc1(x)                  # [B,base,H,W]
        e2 = self.enc2(self.pool(e1))      # [B,2b,H/2,W/2]
        e3 = self.enc3(self.pool(e2))      # [B,4b,H/4,W/4]
        m  = self.mid(self.pool(e3))       # [B,8b,H/8,W/8]

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        feat = self.out(d1)                # features for decoder head
        return feat