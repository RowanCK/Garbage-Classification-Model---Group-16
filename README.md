# Garbage-Classification-Model - Group-16

A PyTorch-based multimodal garbage classification model using image and textual data.

## Team Members
- Rowan (Yi-Kai) Chen  
- Das (Shih Ting) Tai  
- Ryan Lau  
- Zain Jelani  

## Overview

This project implements a multimodal deep learning system for classifying garbage items into four categories:

- Black
- Blue
- Green
- TTR (Other)

The model integrates visual features extracted from images and semantic features derived from text descriptions. A late fusion approach is used to combine both modalities.


## Model Architecture

For this project, we make use of a Late Fusion Approach. This consists of three main components:
- Image Branch:
  - Pre-trained on ResNet50.
  - Outputs 2048 dimensional feature vector from the input images.
  - Backbone frozen to prevent overfitting.
- Text Branch:
  - Pre-trained on BERT.
  - Model processes text descriptions to capture semantic meaning.
- Fusion Head:
  - Image and text features are concatenated into a 2816 dimensional vector.
  - Passed through fully connected layers for final final multimodal classification:
    - Linear layer (512 units)
    - ReLU activation
    - Dropout (0.3)
    - Final linear layer with 4 output classes

## Requirements

The following Python packages are required:

- torch
- torchvision
- transformers
- scikit-learn
- matplotlib
- seaborn
- Pillow
- tqdm
- numpy

To install run `pip install torch torchvision transformers scikit-learn matplotlib seaborn Pillow tqdm numpy`

GPU acceleration is used automatically if available.

## Running Evaluation

To evaluate the trained model:

- Ensure the trained model file exists:
   - `multimodal_garbage_model_251638.pth`

- Ensure the dataset path is correctly set in the script.

- Run the evaluation script to perform inference on the test set.

## Evaluation Outputs

The evaluation pipeline produces:

- Classification report
    - Precision
  - Recall
  - F1-score
  - Overall accuracy
- Confusion matrix
    - Visualized as a heatmap
- Incorrect classification analysis
  - Displays up to 8 misclassified samples
  - Shows the input image, true label, and predicted label
