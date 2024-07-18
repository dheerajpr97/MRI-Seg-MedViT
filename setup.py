import os
import gdown

# Directory and file paths
save_dir = 'saved_models'
checkpoint_file_name = 'checkpoint.pth'
checkpoint_file_path = os.path.join(save_dir, checkpoint_file_name)
pretrained_file_name = 'MedViT_small_im1k.pth'
pretrained_file_path = os.path.join(save_dir, pretrained_file_name)

# Create the saved_models directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory {save_dir} created.")
else:
    print(f"Directory {save_dir} already exists.")

# Check if the checkpoint file already exists
if not os.path.exists(checkpoint_file_path):
    # Google Drive file ID for the checkpoint file
    checkpoint_file_id = '1OVU7711Ycl4bCPHnR597E5q02eb8PhMK'

    # URL to the checkpoint file
    checkpoint_url = f'https://drive.google.com/uc?id={checkpoint_file_id}'

    # Download the checkpoint file
    gdown.download(checkpoint_url, checkpoint_file_path, quiet=False)
    print(f"Checkpoint downloaded to {checkpoint_file_path}.")
else:
    print(f"File {checkpoint_file_path} already exists. Skipping download.")

# Check if the pre-trained model file already exists
if not os.path.exists(pretrained_file_path):
    # Google Drive file ID for the pre-trained model
    pretrained_file_id = '14wcH5cm8P63cMZAUHA1lhhJgMVOw_5VQ'  # Replace with the actual file ID

    # URL to the pre-trained model file
    pretrained_url = f'https://drive.google.com/uc?id={pretrained_file_id}'

    # Download the pre-trained model file
    gdown.download(pretrained_url, pretrained_file_path, quiet=False)
    print(f"Pre-trained model downloaded to {pretrained_file_path}.")
else:
    print(f"File {pretrained_file_path} already exists. Skipping download.")

print("Setup completed.")
