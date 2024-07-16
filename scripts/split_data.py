# python -m scripts.split_data --data_dir 'data/lgg-mri-segmentation' --output_dir 'data/splits'

import os
import numpy as np
from sklearn.model_selection import train_test_split
from scripts.utils import save_indices

def main(data_dir, output_dir):
    """
    Splits the dataset indices into train, validation, and test sets and saves them to files.

    Args:
        data_dir (str): Directory containing the dataset.
        output_dir (str): Directory where the split indices will be saved.
    """
    # Get all indices based on the number of files in the data directory
    all_indices = list(range(len(os.listdir(data_dir))))

    # Split indices into train (80%), validation (16%), and test (4%) sets
    train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.3, random_state=42)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the split indices to files
    save_indices(train_indices, os.path.join(output_dir, 'train_indices.txt'))
    save_indices(val_indices, os.path.join(output_dir, 'val_indices.txt'))
    save_indices(test_indices, os.path.join(output_dir, 'test_indices.txt'))

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Split data into train, validation, and test sets')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split indices')
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.data_dir, args.output_dir)
    print("Split data into train, validation, and test indices and saved them to files.")    