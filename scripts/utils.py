import cv2
import numpy as np
import torch

def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

# Function to overlay masks on the original image
def overlay_masks(image, mask, alpha=0.5, color=(0, 255, 0)):
    mask = mask.squeeze().cpu().numpy()
    image = image.permute(1, 2, 0).cpu().numpy()
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    overlayed_image = image * (1 - alpha) + color_mask * alpha
    return overlayed_image

# Function to highlight contours for both ground truth and predicted masks
def highlight_contours(image, gt_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    gt_mask = gt_mask.astype(np.uint8)
    pred_mask = pred_mask.astype(np.uint8)
    image = (image * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)  # Ensure the image is contiguous
    
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image, gt_contours, -1, gt_color, 2)
    cv2.drawContours(image, pred_contours, -1, pred_color, 2)
    
    return image

def load_indices(indices_path):
    with open(indices_path, 'r') as file:
        indices = [int(line.strip()) for line in file]
    return indices


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0
    else:
        return intersection / union
