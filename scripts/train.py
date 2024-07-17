# python -m scripts.train --data_dir 'data/lgg-mri-segmentation' --indices_dir 'data/splits' --epochs 10 --batch_size 8 --lr 0.001 --save_dir 'saved_models'


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.dataset import LGGSegmentationDataset
from scripts.model import MedViTSegmentation, get_pretrained_model
from scripts.losses import BCEWithLogitsDiceLoss
from scripts.utils import load_indices
import torchvision.transforms as transforms 

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for the model parameters.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    batch_durations = []

    for batch_idx, (images, masks) in enumerate(train_loader):
        batch_start_time = time.time()

        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Calculate batch duration and update list
        batch_duration = time.time() - batch_start_time
        batch_durations.append(batch_duration)
        
        # Estimate remaining time for the epoch
        avg_batch_duration = sum(batch_durations) / len(batch_durations)
        remaining_batches = num_batches - (batch_idx + 1)
        estimated_remaining_time = remaining_batches * avg_batch_duration

        # Print progress and estimated remaining time
        print(f'Batch [{batch_idx + 1}/{num_batches}], '
              f'Loss: {loss.item():.3f}, '
              f'Estimated remaining time: {estimated_remaining_time:.2f} seconds', end='\r')

    avg_train_loss = running_loss / num_batches
    return avg_train_loss

def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def main(data_dir, indices_dir, model_type, epochs, batch_size, lr, save_dir):
    """
    Main function to set up data loaders, model, optimizer, and train/validate the model.

    Args:
        data_dir (str): Path to the dataset directory.
        indices_dir (str): Path to the directory containing split indices.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        save_dir (str): Directory to save model checkpoints.
    """
    # Load train, validation, and test indices
    train_indices = load_indices(os.path.join(indices_dir, 'train_indices.txt'))
    val_indices = load_indices(os.path.join(indices_dir, 'val_indices.txt'))
    test_indices = load_indices(os.path.join(indices_dir, 'test_indices.txt'))

    # Define image and mask transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create datasets and data loaders
    train_dataset = LGGSegmentationDataset(root_dir=data_dir, indices=train_indices, transform=image_transform, target_transform=mask_transform)
    val_dataset = LGGSegmentationDataset(root_dir=data_dir, indices=val_indices, transform=image_transform, target_transform=mask_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Set device to GPU if available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model = get_pretrained_model(pretrained_model=model_type)
    model = MedViTSegmentation(pretrained_model, num_classes=1).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # criterion = BCEWithLogitsDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'Epoch [{epoch + 1}/{epochs}]')
        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'\nEpoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model checkpoint if current validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model checkpoint at {checkpoint_path}')
        
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch duration: {epoch_duration:.2f} seconds')

    print('Training complete.')
    torch.save(model.state_dict(), os.path.join(save_dir, 'fine_tuned_model.pth'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train MedViT on LGG MRI Segmentation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--indices_dir', type=str, required=True, help='Path to the directory containing split indices')
    parser.add_argument('--model_type', type=str, default='small', help='Type of pre-trained model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.indices_dir, args.model_type, args.epochs, args.batch_size, args.lr, args.save_dir)

