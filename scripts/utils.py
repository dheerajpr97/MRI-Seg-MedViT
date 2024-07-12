import numpy as np
import cv2

def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def overlay_masks(image, mask, alpha=0.5, color=(0, 1, 0)):
    mask = mask.squeeze().cpu().numpy()
    image = image.permute(1, 2, 0).cpu().numpy()
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    overlayed_image = image * (1 - alpha) + color_mask * alpha
    return overlayed_image

def highlight_contours(image, mask, color=(1, 0, 0)):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2)
    return image