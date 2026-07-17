# Model Architecture

## Overview

NeuroVision-AI is a deep learning framework developed for Alzheimer's disease classification using structural MRI scans and optional clinical features.

The system follows a modular pipeline consisting of image preprocessing, feature extraction, feature fusion, and disease classification.

---

## Architecture

Input MRI Scan
        │
        ▼
MRI Preprocessing
        │
        ▼
CNN Backbone
(ResNet18 / ResNet50 / EfficientNet-B0)
        │
        ▼
Deep Feature Extraction
        │
        ▼
(Optional)
Clinical Feature Encoder
        │
        ▼
Feature Fusion
        │
        ▼
Fully Connected Layers
        │
        ▼
Softmax Classifier
        │
        ▼
Prediction

---

## MRI Branch

The MRI branch extracts visual features from structural brain scans using transfer learning.

Supported backbones include:

- ResNet18
- ResNet50
- EfficientNet-B0

Transfer learning enables faster convergence while reducing overfitting on relatively small medical datasets.

---

## Clinical Branch

Clinical metadata such as

- Age
- Gender
- Education
- MMSE
- CDR

can optionally be encoded using a small multilayer perceptron (MLP).

---

## Feature Fusion

MRI image features and clinical embeddings are concatenated before classification.

Fusion allows the model to combine anatomical information with patient metadata for improved decision making.

---

## Classification

The final classifier predicts one of three classes:

- CN (Cognitively Normal)
- MCI (Mild Cognitive Impairment)
- AD (Alzheimer's Disease)

---

## Explainability

Grad-CAM is integrated to visualize regions of the MRI contributing most to the model's prediction, improving interpretability for medical analysis.