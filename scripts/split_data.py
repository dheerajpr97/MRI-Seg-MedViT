# python scripts/split_data.py --data_dir 'data/lgg-mri-segmentation' --output_dir 'data/splits'

import os
import numpy as np
from sklearn.model_selection import train_test_split

def save_indices(indices, file_path):
    with open(file_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")

def main(data_dir, output_dir):
    all_indices = list(range(len(os.listdir(data_dir))))

    # Split into train, validation, and test indices
    train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.3, random_state=42)

    # Save the indices
    os.makedirs(output_dir, exist_ok=True)
    save_indices(train_indices, os.path.join(output_dir, 'train_indices.txt'))
    save_indices(val_indices, os.path.join(output_dir, 'val_indices.txt'))
    save_indices(test_indices, os.path.join(output_dir, 'test_indices.txt'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Split data into train, validation, and test sets')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split indices')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
