# python main.py train --data_dir data/lgg-mri-segmentation --indices_dir data/splits --epochs 25 --batch_size 4 --lr 0.001 --save_dir saved_models
# python main.py visualize --data_dir data/lgg-mri-segmentation --model_path saved_models/best_model_epoch_6.pth --batch_size 4

import os
import random
import argparse
import torch
from scripts.train import main as train_main
from scripts.visualization import visualize_results
from scripts.dataset import LGGSegmentationDataset, image_transform, mask_transform
from scripts.model import MedViT_Segmentation, get_pretrained_model
from scripts.utils import load_indices
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Segmentation Fine-Tuning and Visualization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training and validation data")
    train_parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    train_parser.add_argument("--indices_dir", type=str, default=None, help="Path to the file containing test indices")
    train_parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save model checkpoints")


    visualize_parser = subparsers.add_parser("visualize", help="Visualize segmentation results")
    visualize_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test data")
    visualize_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    visualize_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    visualize_parser.add_argument("--indices_dir", type=str, default=None, help="Path to the file containing test indices")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.command == "train":
        train_main(args.data_dir, args.indices_dir, args.epochs, args.batch_size, args.lr, args.save_dir)
    elif args.command == "visualize":
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

if __name__ == "__main__":
    main()


# python main.py 