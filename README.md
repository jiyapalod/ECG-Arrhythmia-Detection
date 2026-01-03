# ECG Arrhythmia Classification (MATLAB)

## Overview
This project focuses on automated ECG arrhythmia classification using supervised machine learning. Pre-segmented ECG beats from standard public datasets are used to train and evaluate a neural network–based classifier implemented entirely in **MATLAB**.

The goal of the project is to explore ECG signal classification using a reproducible and interpretable pipeline grounded in digital signal processing and machine learning principles.

## Tools and Technologies
- **MATLAB**
- Deep Learning Toolbox
- Signal Processing concepts

## Datasets
The project utilizes publicly available ECG datasets commonly used in biomedical signal processing research:
- **MIT-BIH Arrhythmia Dataset** – provides labeled arrhythmia classes
- **PTB Diagnostic ECG Database (PTBDB)** – used in binary normal vs abnormal form

Pre-segmented ECG beats provided in CSV format are used for supervised learning.  
The datasets are combined during training to encourage generalization across different ECG sources, while evaluation is performed exclusively on the MIT-BIH test set.

> Note: Raw datasets are not included in this repository. Please place the required CSV files inside a `data/` directory to reproduce results.

## Methodology
- Dataset loading and label preparation
- Per-sample ECG signal normalization
- Train / validation / test split
- Fully connected neural network training
- Performance evaluation using accuracy, confusion matrix, and macro F1-score

## Model Architecture
The classifier is a fully connected neural network comprising:
- Feature input layer (ECG beat samples)
- Two hidden layers with ReLU activation and dropout
- Softmax output layer for classification

This architecture serves as a baseline deep learning approach for ECG beat classification.

## Results
The model achieves consistent classification performance on the MIT-BIH test set. Confusion matrix analysis highlights class-wise behavior and common misclassification patterns, while macro-averaged F1-score is used to account for class imbalance.

## Limitations
- No explicit ECG-specific filtering (e.g., baseline wander or noise removal)
- Limited architectural exploration beyond fully connected networks
- Mixed dataset label semantics

## Future Work
- Incorporation of ECG-specific preprocessing techniques
- CNN-based or temporal models for improved feature learning
- More refined multi-class arrhythmia analysis

## Repository Structure
