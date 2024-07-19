# python -m scripts.model

import os
import torch
import torch.nn as nn
from scripts.MedViT.MedViT import MedViT_small as small
from scripts.MedViT.MedViT import MedViT_large as large 

def get_pretrained_model(pretrained_model='small'):
    """
    Loads the pre-trained MedViT model.

    Args:
        pretrained_model (str): The type of pre-trained MedViT model to load ('small' or 'large').

    Returns:
        nn.Module: The pre-trained MedViT model.
    """
    if pretrained_model == 'small':
        model = small()
        checkpoint_path = 'saved_models/MedViT_small_im1k.pth'
    elif pretrained_model == 'large':
        model = large()
        checkpoint_path = 'saved_models/MedViT_large_im1k.pth'
    else:
        raise ValueError(f"Invalid pretrained model: {pretrained_model}")

    # Check if running inside Docker by checking the environment variable
    if os.getenv('IN_DOCKER') == '1':
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model'])
    return model

class MedViTSegmentation(nn.Module):
    """
    Segmentation model using a pre-trained MedViT as the encoder and a custom decoder.

    Args:
        pretrained_model (nn.Module): The pre-trained MedViT model.
        num_classes (int): The number of classes for the segmentation task.
    """
    def __init__(self, pretrained_model, num_classes):
        super(MedViTSegmentation, self).__init__()
        # Use all layers of the pretrained model except the last two
        self.encoder = nn.Sequential(*list(pretrained_model.children())[:-2])
        # Define the decoder with transposed convolutions to upsample the feature maps
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
            nn.Conv2d(32, num_classes, kernel_size=1)  # Final layer to get the desired number of output classes
        )

    def forward(self, x):
        """
        Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the encoder and decoder.
        """
        x = self.encoder(x)  # Apply the encoder
        x = self.decoder(x)  # Apply the decoder
        return x

if __name__ == "__main__":
    # Load the pretrained model and initialize the segmentation model
    pretrained_model = get_pretrained_model(pretrained_model='small')
    print("Pretrained Med-ViT model loaded successfully")
    model = MedViTSegmentation(pretrained_model, num_classes=1)
    print("Segmentation Model loaded successfully")
