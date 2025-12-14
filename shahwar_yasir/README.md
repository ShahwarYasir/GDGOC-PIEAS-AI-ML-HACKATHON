# Diabetic Retinopathy Detection – DRResNet

## Overview
This project implements a deep learning model for detecting Diabetic Retinopathy (DR) from retinal images. The goal is to classify retinal images into five severity levels:

0 – No DR

1 – Mild DR

2 – Moderate DR

3 – Severe DR

4 – Proliferative DR

The model is trained from scratch using a custom DRResNet architecture with advanced preprocessing, augmentation, and explainability via Grad-CAM.

## Repository Structure
shahwar_yasir/

├── model/

│   └── drresnet_enhanced.pth  

├── notebooks/

│   └── training_gradcam.ipynb  

├── README.md

├── requirements.txt    

└── report.pdf                      

## Dataset
Source: Kaggle – Diabetic Retinopathy Balanced Dataset
Structure: Train / Validation / Test directories, each containing subfolders for 5 classes
Samples:
Train: 34,792
Validation: 9,940
Test: 4,971

## Installation
Clone this repository and install dependencies:

git clone https://github.com/ShahwarYasir/GDGOC-PIEAS-AI-ML-HACKATHON.git
cd shahwar_yasir
pip install -r requirements.txt


## Recommended libraries:
torch, torchvision, torchmetrics
numpy, pandas, matplotlib
opencv-python
tqdm

## Usage
## 1. Prepare Dataset
Download and unzip the dataset to the folder structure:
dataset/content/Diabetic_Balanced_Data/
    
    ├── train/
   
    ├── val/
    
    └── test/

## 2. Training
Run the notebook for training:
# In Jupyter/Colab
!jupyter notebook notebooks/training_gradcam.ipynb
Model: DRResNet (from scratch)
Optimizer: Adam
Scheduler: CosineAnnealingLR
Loss: Weighted CrossEntropy
Mixed precision training enabled

## 3. Evaluation
After training, load the best model weights:
model.load_state_dict(torch.load("model/drresnet_enhanced.pth"))
model.eval()
Evaluate on the test set for accuracy, F1-score, precision, recall

## 4. Grad-CAM Visualization
Visualize important regions of retinal images:
visualize_gradcam(model, test_loader, target_layer=model.layer4[1].conv2, device=device, num_images=5)
Results
### Test Accuracy: ~0.70
Grad-CAM visualizations show the model focuses on relevant retinal regions for DR classification.
## Strengths
Fully original model trained from scratch
Handles class imbalance via weighted loss
Explainable predictions with Grad-CAM
GPU-optimized training with mixed precision

## Limitations
Training on full dataset takes time
Accuracy can improve with ensemble or larger models

## References
Kaggle Diabetic Retinopathy Dataset
He, K. et al. “Deep Residual Learning for Image Recognition,” CVPR 2016
Grad-CAM: Selvaraju et al., 2017


