import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, pad):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, pad=0, out_pad=0):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, output_padding=out_pad, padding=pad),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)



