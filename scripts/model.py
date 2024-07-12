import sys
import os

import torch
import torch.nn as nn
from MedViT.MedViT import MedViT_small as tiny

def get_pretrained_model():
    pretrained_model = tiny()
    checkpoint = torch.load('saved_models/MedViT_small_im1k.pth')
    pretrained_model.load_state_dict(checkpoint['model'])
    return pretrained_model

class MedViT_Segmentation(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(MedViT_Segmentation, self).__init__()
        self.encoder = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    pretrained_model = get_pretrained_model()
    model = MedViT_Segmentation(pretrained_model, num_classes=1)
    print("Model loaded successfully")