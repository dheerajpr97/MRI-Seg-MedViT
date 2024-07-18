import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def save_indices(indices, file_path):
    """
    Saves indices to a file.

    Args:
        indices (list): List of indices to save.
        file_path (str): Path to the file where indices will be saved.
    """
    with open(file_path, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

def load_indices(file_path):
    """
    Loads indices from a file.

    Args:
        file_path (str): Path to the file containing indices.

    Returns:
        list: List of indices as strings.
    """
    with open(file_path, 'r') as f:
        indices = [line.strip() for line in f]  # Read each line and strip any extra whitespace
    return indices

def unnormalize(img, mean, std):
    """
    Unnormalizes a tensor image using the given mean and standard deviation.

    Args:
        img (torch.Tensor): The image tensor to unnormalize.
        mean (list): The mean values used for normalization.
        std (list): The standard deviation values used for normalization.

    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    img = img.clone()
    for t, m, s in zip(img, mean, std): # For each channel in the tensor, t = img, m = mean, s = std 
        t.mul_(s).add_(m)
    return img

def overlay_masks(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Overlays a mask on the original image with a given transparency and color.

    Args:
        image (torch.Tensor): The original image tensor.
        mask (torch.Tensor): The mask tensor to overlay.
        alpha (float, optional): The transparency factor for the overlay. Default is 0.5.
        color (tuple, optional): The color to use for the mask overlay. Default is green (0, 255, 0).

    Returns:
        np.ndarray: The image with the overlayed mask.
    """
    mask = mask.squeeze().cpu().numpy()
    image = image.permute(1, 2, 0).cpu().numpy()
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    overlayed_image = image * (1 - alpha) + color_mask * alpha
    return overlayed_image

def highlight_contours(image, gt_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Highlights contours of the ground truth and predicted masks on the original image.

    Args:
        image (np.ndarray): The original image.
        gt_mask (np.ndarray): The ground truth mask.
        pred_mask (np.ndarray): The predicted mask.
        gt_color (tuple, optional): Color for the ground truth mask contours. Default is green (0, 255, 0).
        pred_color (tuple, optional): Color for the predicted mask contours. Default is red (255, 0, 0).

    Returns:
        np.ndarray: The image with highlighted contours.
    """
    gt_mask = gt_mask.astype(np.uint8)
    pred_mask = pred_mask.astype(np.uint8)
    image = (image * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)  # Ensure the image is contiguous

    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, gt_contours, -1, gt_color, 1)
    cv2.drawContours(image, pred_contours, -1, pred_color, 1)

    return image


def calculate_iou(gt_mask, pred_mask):
    """
    Calculates the Intersection over Union (IoU) for the given ground truth and predicted masks.

    Args:
        gt_mask (np.ndarray): The ground truth mask.
        pred_mask (np.ndarray): The predicted mask.

    Returns:
        float: The IoU value.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0
    else:
        return intersection / union
    
def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])