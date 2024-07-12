# scripts/train.py

import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LGGSegmentationDataset, image_transform, mask_transform
from model import get_pretrained_model, MedViT_Segmentation
from sklearn.model_selection import train_test_split

def train(model, train_loader, criterion, optimizer, device):
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

def main(data_dir, epochs, batch_size, lr, save_dir):
    dataset = LGGSegmentationDataset(root_dir=data_dir, transform=image_transform, target_transform=mask_transform)

    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model = get_pretrained_model()
    model = MedViT_Segmentation(pretrained_model, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')  # Initialize best validation loss to infinity

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
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir)
