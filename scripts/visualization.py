# python scripts/visualization.py --data_dir 'data/lgg-mri-segmentation' --model_path 'saved_models/best_model_epoch_6.pth' --batch_size 4

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
from scripts.dataset import LGGSegmentationDataset, image_transform, mask_transform
from scripts.model import MedViT_Segmentation, get_pretrained_model
from torch.utils.data import DataLoader
from scripts.utils import unnormalize, overlay_masks, highlight_contours, calculate_iou, load_indices
import os
import random

def visualize_results(model, test_loader, device, mean, std, threshold=0.5):
    model.eval()
    positive_diagnosis_samples = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > threshold

            for idx in range(len(images)):
                if torch.sum(masks[idx]) > 0:
                    img = unnormalize(images[idx].cpu(), mean, std).permute(1, 2, 0).numpy()
                    gt_mask = masks[idx].cpu().squeeze().numpy()
                    pred_mask = preds[idx].cpu().squeeze().numpy().astype(np.uint8)
                    # iou_scores.append(calculate_iou(gt_mask, pred_mask))
                    positive_diagnosis_samples.append((img, gt_mask, pred_mask))
            
            # if len(positive_diagnosis_samples) >= 5:
            #     break

    positive_diagnosis_samples = positive_diagnosis_samples[:10]
    
    sample_imgs = [cv2.resize(img, (224, 224)) for img, _, _ in positive_diagnosis_samples]
    sample_gt_masks = [cv2.resize(gt_mask, (224, 224)) for _, gt_mask, _ in positive_diagnosis_samples]
    sample_pred_masks = [cv2.resize(pred_mask, (224, 224)) for _, _, pred_mask in positive_diagnosis_samples]

    sample_imgs_arr = np.hstack(sample_imgs)
    sample_gt_masks_arr = np.hstack(sample_gt_masks)
    sample_pred_masks_arr = np.hstack(sample_pred_masks)

    sample_overlayed_masks = []
    for img, gt_mask, pred_mask in positive_diagnosis_samples:
        iou_score = calculate_iou(gt_mask, pred_mask)
        overlayed_img = overlay_masks(torch.tensor(img).permute(2, 0, 1), torch.tensor(gt_mask), alpha=0.5, color=(0, 255, 0))
        overlayed_img = highlight_contours(overlayed_img, gt_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 0, 0))
         # Add IoU text to the overlayed image
        overlayed_img = cv2.putText(
            overlayed_img, f'IoU: {iou_score:.4f}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        sample_overlayed_masks.append(cv2.resize(overlayed_img, (224, 224)))
    sample_overlayed_masks_arr = np.hstack(sample_overlayed_masks)

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 1), axes_pad=0.5)

    titles = ["Images", "Ground Truth Masks", "Predicted Masks", "Overlayed Masks with Contours"]
    arrays = [sample_imgs_arr, sample_gt_masks_arr, sample_pred_masks_arr, sample_overlayed_masks_arr]

    for ax, arr, title in zip(grid, arrays, titles):
        ax.imshow(arr, cmap='gray' if 'Masks' in title else None)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
        ax.grid(False)

    plt.show()

def load_indices(indices_path):
    with open(indices_path, 'r') as file:
        indices = [int(line.strip()) for line in file]
    return indices

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--indices_dir", type=str, default=None, help="Path to the file containing test indices")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.indices_dir and os.path.exists(args.indices_dir):
        test_indices = load_indices(args.indices_dir)
    else:
        dataset_size = len(os.listdir(args.data_dir))
        test_indices = random.sample(range(dataset_size), int(0.3 * dataset_size))

    test_dataset = LGGSegmentationDataset(root_dir=args.data_dir, indices=test_indices, transform=image_transform, target_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    pretrained_model = get_pretrained_model()
    model = MedViT_Segmentation(pretrained_model, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    visualize_results(model, test_loader, device, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], threshold=0.5)
