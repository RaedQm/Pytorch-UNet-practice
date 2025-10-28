""" Full assembly of the parts to form the complete network """

import torch
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, base_c: int = 16):
        """
        UNet with configurable base channels (base_c).
        Smaller base_c -> much faster and lower memory usage, at cost of accuracy.
        Default base_c=16 is aggressive for fast runs on consumer GPUs.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_c = base_c

        # compute channel widths from base
        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        c4 = base_c * 8
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, c1))
        self.down1 = (Down(c1, c2))
        self.down2 = (Down(c2, c3))
        self.down3 = (Down(c3, c4))
        self.down4 = (Down(c4, c4 * 2 // factor))
        self.up1 = (Up(c4 * 2, c4 // factor, bilinear))
        self.up2 = (Up(c4, c3 // factor, bilinear))
        self.up3 = (Up(c3, c2 // factor, bilinear))
        self.up4 = (Up(c2, c1, bilinear))
        self.outc = (OutConv(c1, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)