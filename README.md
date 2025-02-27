# Brain MRI Segmentation using MedViT

This project focuses on fine-tuning the MedViT (Medical Vision Transformer) model for MRI segmentation tasks. The goal is to accurately segment regions of interest in MRI scans using a pre-trained MedViT model, followed by evaluating the performance using various metrics such as Intersection over Union (IoU).

## Getting Started

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/dheerajpr97/MRI-Seg-MedViT.git
   cd MRI-Seg-MedViT
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
3. To avoid training and to use the pre-trained checkpoints for both the segmentation and MedViT models: 
   ```sh
   python setup.py
   ```

### Dataset Preparation
The dataset [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) is used for obtaining Brain MRI scans. The data corresponds to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Ensure your dataset is organized as follows:
```
data/
   lgg-mri-segmentation/
      TCGA_CS_4941_19960909/
      image1.tif
      image1_mask.tif
      ...
      
   splits/
      train_indices.txt
      val_indices.txt
      test_indices.txt
```

### Training the Model
To train the model, run the following command:
   ```sh
   python main.py train --data_dir data/lgg-mri-segmentation --indices_dir data/splits --epochs 25 --batch_size 4 --lr 0.001 --save_dir saved_models
   ```

### Visualizing the Results
To visualize the segmentation results, use the following command:

   ```sh 
   python -m scripts.visualization --data_dir 'data/lgg-mri-segmentation' --indices_dir 'data/splits/test_indices.txt' --model_path 'saved_models/checkpoint.pth' --batch_size 4
   ```
The visualization will show the original images, ground truth masks, predicted masks, and overlayed masks with contours along with their IoU scores.

### Results
 ![Output](output_5.jpeg)
### Interpretation
- Images: Original MRI scans.
- Ground Truth Masks: Manually annotated masks.
- Predicted Masks: Masks predicted by the MedViT model.
- Overlayed Masks with Contours: Original images with ground truth and predicted mask contours. The IoU score for each sample is displayed above the overlayed image.

### Discussion
The segmentation results demonstrate that the MedViT model can accurately identify regions of interest in MRI scans. The predicted masks show that finer edges are smoothened, resulting in a neat and cohesive segmentation. However, there is room for improvement. Further tuning and fine-tuning of the model parameters could enhance the segmentation accuracy and edge details. Overall, the current results provide a solid foundation and indicate that the MedViT model is a promising tool for MRI segmentation tasks.

### Using Docker
To containerize and run the project using Docker, follow these steps:

1. Build the Docker image:

```sh
docker build -t mri-medvit-app .
```
2. Run the Docker container:
```sh
docker run -p 5000:80 mri-medvit-app
```
This will start the application and map port 5000 (for example) on your host to port 80 in the Docker container. You can access the application at http://localhost:5000.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
This project utilizes the LGG MRI Segmentation Dataset obtained from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
This project utilizes the MedViT model and related scripts from the [Original Repository](https://github.com/Omid-Nejati/MedViT). 

### TO DO

-  [x] Refactor code in all the scripts
-  [x] Add an UI script and expose it locally as an application
-  [x] Encapsulate and containerize the project
-  [x] Add a CI/CD pipeline
- [] Add Cloud functionality
