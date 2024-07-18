# python -m scripts.split_data --data_dir 'data/lgg-mri-segmentation' --output_dir 'data/splits'

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from scripts.utils import save_indices

def get_all_files(data_dir):
    """
    Recursively collects all files in the data directory and its subdirectories.
    
    Args:
        data_dir (str): Directory containing the dataset.
        
    Returns:
        List of file paths.
    """
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def main(data_dir, output_dir):
    """
    Splits the dataset files into train, validation, and test sets, saves the indices to files,
    and copies the test images and their masks to another directory in the output directory.

    Args:
        data_dir (str): Directory containing the dataset.
        output_dir (str): Directory where the split indices will be saved and test data will be copied.
    """
    # Get all file paths in the data directory
    all_files = get_all_files(data_dir)
    all_indices = list(range(len(all_files)))

    # Split indices into train, validation, and test sets
    train_indices, val_indices = train_test_split(all_indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.3, random_state=42)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the split indices to files
    save_indices([all_files[i] for i in train_indices], os.path.join(output_dir, 'train_indices.txt'))
    save_indices([all_files[i] for i in val_indices], os.path.join(output_dir, 'val_indices.txt'))
    save_indices([all_files[i] for i in test_indices], os.path.join(output_dir, 'test_indices.txt'))

    print("Split data into train, validation, and test indices and saved them to files.")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Split data into train, validation, and test sets')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split indices and test data')
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.data_dir, args.output_dir)
