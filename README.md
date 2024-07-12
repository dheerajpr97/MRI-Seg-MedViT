# MRI-Seg-MedViT

This repository contains tools for fine-tuning and visualizing segmentation results using a pre-trained MedViT model on Brain MRI scans.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MRI-Seg-MedViT.git
   cd MRI-Seg-MedViT
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

1. Ensure your dataset is organized under `data/train` and `data/val`.
2. Run the training script:
   ```bash
   python main.py train --data_dir data --epochs 25 --batch_size 4 --lr 0.001
   ```

### Visualization

1. After training, visualize the results:
   ```bash
   python main.py visualize --data_dir data/test --model_path saved_models/best_model_epoch_6.pth --batch_size 4
   ```

## Credits

This project utilizes the MedViT model and related scripts from the [Original Repository](https://github.com/Omid-Nejati/MedViT). The scripts for datasets, engine, losses, and other utilities have been adapted from this repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

