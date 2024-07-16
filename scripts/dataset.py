# python -m scripts.dataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scripts.utils import load_indices

class LGGSegmentationDataset(Dataset):
    """
    Custom dataset class for LGG MRI segmentation dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        indices (list): List of indices representing the dataset.
        transform (optional): Transform to be applied to the images.
        target_transform (optional): Transform to be applied to the masks.
    """
    def __init__(self, root_dir, indices, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.mask_paths = []

        # Iterate over the indices and populate image and mask paths
        for idx in indices:
            case = os.listdir(root_dir)[idx]  # Get the case directory by index
            case_dir = os.path.join(root_dir, case)
            if os.path.isdir(case_dir):  # Ensure it's a directory
                for file in os.listdir(case_dir):
                    if file.endswith('.tif') and not file.endswith('_mask.tif'):
                        # Add image path
                        self.image_paths.append(os.path.join(case_dir, file))
                        # Add corresponding mask path
                        mask_path = os.path.join(case_dir, file.replace('.tif', '_mask.tif'))
                        self.mask_paths.append(mask_path)

    def __len__(self):
        """
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image and mask to retrieve.

        Returns:
            tuple: Tuple containing the image and mask tensors.
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Load and convert image to RGB
        mask = Image.open(mask_path).convert('L')   # Load and convert mask to grayscale

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

# Define image transformation pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Define mask transformation pipeline
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize masks to 224x224
    transforms.ToTensor()           # Convert masks to tensor
])

if __name__ == "__main__":
    # Test the dataset script
    indices_path = 'data/splits/train_indices.txt'  # Path to the file containing indices
    indices = load_indices(indices_path)  # Load indices from the file
    dataset = LGGSegmentationDataset(root_dir='data/lgg-mri-segmentation', indices=indices, 
                                     transform=image_transform, target_transform=mask_transform)
    image, mask = dataset[0]  # Get the first image and mask
    print(image.shape, mask.shape)  # Print their shapes
