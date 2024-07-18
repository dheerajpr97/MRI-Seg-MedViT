import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.model import MedViTSegmentation, get_pretrained_model
from scripts.utils import unnormalize, overlay_masks, highlight_contours, image_transform, calculate_iou

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MedViTSegmentation(get_pretrained_model(), num_classes=1).to(device)
model.load_state_dict(torch.load('saved_models/checkpoint.pth', map_location=device))
model.eval()

def load_image(image_stream):
    print(f"Loading image from stream")
    image = Image.open(image_stream).convert('RGB')
    transform = image_transform()
    print(f"Applying transformation: {transform}")
    image = transform(image)
    print(f"Image transformed: {image.shape}")
    return image.unsqueeze(0)

def load_mask(mask_stream, size):
    print(f"Loading mask from stream")
    mask = Image.open(mask_stream).convert('L')
    mask = mask.resize(size, Image.NEAREST)
    mask = np.array(mask)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    mask = (mask > 127).float()
    return mask

def convert_jpeg_to_tif(jpeg_stream):
    image = Image.open(jpeg_stream).convert('RGB')
    tif_stream = io.BytesIO()
    image.save(tif_stream, format='TIFF')
    tif_stream.seek(0)
    return tif_stream

def visualize_single_image(model, image_stream, mask_stream, device, mean, std, threshold=0.5):
    image = load_image(image_stream).to(device)
    print(f"Image loaded and moved to device: {image.shape}")

    img_size = image.shape[2], image.shape[3]
    gt_mask = load_mask(mask_stream, img_size).to(device) if mask_stream else None
    if gt_mask is not None:
        print(f"Ground truth mask loaded: {gt_mask.shape}")

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)
        pred_mask = prob > threshold
    print(f"Model prediction completed")

    img = unnormalize(image.squeeze(), mean, std).permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    print(f"Unnormalized image and predicted mask ready")

    img_tensor = torch.tensor(img).permute(2, 0, 1)
    overlayed_img = overlay_masks(img_tensor, torch.tensor(pred_mask), alpha=0.5, color=(0, 255, 0))

    iou_score = None
    if gt_mask is not None:
        gt_mask = gt_mask.squeeze().cpu().numpy().astype(np.uint8)
        overlayed_img = highlight_contours(overlayed_img, gt_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 0, 0))
        iou_score = calculate_iou(pred_mask, gt_mask)
        print(f"IoU score calculated: {iou_score}")
    else:
        overlayed_img = highlight_contours(overlayed_img, pred_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 0, 0))

    return img, overlayed_img, iou_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part for image'
    file = request.files['file']
    mask = request.files.get('mask')
    if file.filename == '':
        return 'No selected file'
    
    try:
        if file.filename.endswith('.jpeg') or file.filename.endswith('.jpg'):
            img_stream = convert_jpeg_to_tif(io.BytesIO(file.read()))
        elif file.filename.endswith('.tif'):
            img_stream = io.BytesIO(file.read())
        else:
            return 'Invalid file type. Please upload a .tif or .jpeg file.'

        mask_stream = io.BytesIO(mask.read()) if mask and mask.filename != '' else None
        
        img, overlayed_img, iou_score = visualize_single_image(model, img_stream, mask_stream, device, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], threshold=0.5)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis("off")
        
        title = 'Overlayed Image with Mask'
        if iou_score is not None:
            title += f'\nIoU Score: {iou_score:.2f}'
        ax[1].imshow(overlayed_img)
        ax[1].set_title(title)
        ax[1].axis("off")
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='jpeg')
        img_bytes.seek(0)
        plt.close(fig)
        print(f"Output prepared for response")

        return send_file(img_bytes, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return str(e)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
