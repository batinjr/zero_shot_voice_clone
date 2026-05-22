import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.dropout(self.relu(self.bn(self.conv(x)))))

class SpeakerEncoderTier1(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)

        self.fc = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        mean = torch.mean(x, dim=[2, 3]) # Shape: (Batch, 128)
        std = torch.std(x, dim=[2, 3])   # Shape: (Batch, 128)
        
        # We combine them. This captures both average and variation.
        x = torch.cat((mean, std), dim=1) # Shape becomes (Batch, 256)
        
        x = self.fc(x)
        return x