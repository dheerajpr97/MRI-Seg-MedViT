# python -m main train --data_dir data/lgg-mri-segmentation --indices_dir data/splits --epochs 25 --batch_size 4 --lr 0.001 --save_dir saved_models
# python -m main visualize --data_dir data/lgg-mri-segmentation --model_type 'small' --model_path saved_models/best_model.pth --batch_size 4

import os
import random
import argparse
import torch
from scripts.train import main as train_main
from scripts.visualization import visualize_results
from scripts.dataset import LGGSegmentationDataset, image_transform, mask_transform
from scripts.model import MedViTSegmentation, get_pretrained_model
from scripts.utils import load_indices
from torch.utils.data import DataLoader

def parse_arguments():
    """
    Parse command line arguments for training and visualization.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Segmentation Fine-Tuning and Visualization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training and validation data")
    train_parser.add_argument("--model_type", type=str, default="small", help="Type of pre-trained model to train")
    train_parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    train_parser.add_argument("--indices_dir", type=str, default=None, help="Path to the file containing test indices")
    train_parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save model checkpoints")

    visualize_parser = subparsers.add_parser("visualize", help="Visualize segmentation results")
    visualize_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test data")
    visualize_parser.add_argument("--model_type", type=str, default="small", help="Type of pre-trained model to test")
    visualize_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    visualize_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    visualize_parser.add_argument("--indices_dir", type=str, default=None, help="Path to the file containing test indices")
    visualize_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for visualization")

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.command == "train":
        print("Starting training...")
        train_main(args.data_dir, args.indices_dir, args.model_type, args.epochs, args.batch_size, args.lr, args.save_dir)
        print("Training completed.")
    elif args.command == "visualize":
        print("Starting visualization...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.indices_dir and os.path.exists(args.indices_dir):
            test_indices = load_indices(args.indices_dir)
            print(f"Loaded test indices from {args.indices_dir}.")
        else:
            dataset_size = len(os.listdir(args.data_dir))
            test_indices = random.sample(range(dataset_size), int(0.3 * dataset_size))
            print("Generated random test indices.")

        print("Creating test dataset...")
        test_dataset = LGGSegmentationDataset(
            root_dir=args.data_dir, 
            indices=test_indices, 
            transform=image_transform, 
            target_transform=mask_transform
        )

        print("Creating DataLoader for test dataset...")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print("Loading pretrained model...")
        pretrained_model = get_pretrained_model(pretrained_model=args.model_type)

        print("Creating segmentation model...")
        model = MedViTSegmentation(pretrained_model, num_classes=1).to(device)

        print(f"Loading model weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        print("Visualizing results...")
        visualize_results(
            model, 
            test_loader, 
            device, 
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5], 
            threshold=args.threshold
        )
        print("Visualization completed.")

if __name__ == "__main__":
    main()
