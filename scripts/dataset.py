# scripts/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LGGSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.mask_paths = []

        for case in os.listdir(root_dir):
            case_dir = os.path.join(root_dir, case)
            if os.path.isdir(case_dir):
                for file in os.listdir(case_dir):
                    if file.endswith('.tif') and not file.endswith('_mask.tif'):
                        self.image_paths.append(os.path.join(case_dir, file))
                        mask_path = os.path.join(case_dir, file.replace('.tif', '_mask.tif'))
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if __name__ == "__main__":
    # Test the dataset script
    dataset = LGGSegmentationDataset(root_dir='data\lgg-mri-segmentation', transform=image_transform, target_transform=mask_transform)
    image, mask = dataset[0]
    print(image.shape, mask.shape)
