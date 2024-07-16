import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation tasks.

    Dice Loss is a commonly used loss function for image segmentation problems.
    It measures the overlap between predicted and ground truth masks.

    Args:
        smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for DiceLoss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth binary masks.
            smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1.

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to the inputs to get probabilities
        inputs = inputs.view(-1)  # Flatten the inputs
        targets = targets.view(-1)  # Flatten the targets
        
        intersection = (inputs * targets).sum()  # Compute intersection
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  # Compute Dice coefficient
        return 1 - dice  # Return Dice loss

class BCEWithLogitsDiceLoss(nn.Module):
    """
    Combination of Binary Cross-Entropy (BCE) and Dice Loss.

    This loss function combines BCEWithLogitsLoss and DiceLoss, which is useful for
    binary segmentation tasks where both per-pixel classification accuracy and overlap
    between predicted and ground truth masks are important.

    Args:
        bce (nn.Module): BCEWithLogitsLoss module.
        dice (nn.Module): DiceLoss module.
    """
    def __init__(self):
        super(BCEWithLogitsDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Initialize BCEWithLogitsLoss
        self.dice = DiceLoss()  # Initialize DiceLoss

    def forward(self, inputs, targets):
        """
        Forward pass for BCEWithLogitsDiceLoss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Combined BCE and Dice loss.
        """
        bce_loss = self.bce(inputs, targets)  # Compute BCE loss
        dice_loss = self.dice(inputs, targets)  # Compute Dice loss
        return bce_loss + dice_loss  # Return the sum of BCE and Dice loss
