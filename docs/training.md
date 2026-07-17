# Training Strategy

## Objective

The objective is to classify MRI scans into

- CN
- MCI
- AD

using deep learning.

---

## Model Training

The framework supports multiple CNN backbones including

- ResNet18
- ResNet50
- EfficientNet-B0

The backbone can be selected through the configuration file.

---

## Loss Function

CrossEntropyLoss is used for multi-class classification.

---

## Optimizer

Adam optimizer

Learning Rate: configurable

Weight Decay: configurable

---

## Regularization

To reduce overfitting the project supports

- Dropout
- Early Stopping
- Learning Rate Scheduling
- Data Augmentation

---

## Evaluation Metrics

Performance is evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## Explainability

Grad-CAM is used to highlight important brain regions contributing to each prediction.

This increases model transparency and assists in qualitative evaluation.

---

## Reproducibility

Random seeds are fixed during training to improve experiment reproducibility.

Training parameters are stored inside the configuration file.

---

## Running Training

```bash
python train.py
```

---

## Running Evaluation

```bash
python evaluate.py
```

---

## Running Inference

```bash
python predict.py
```

---

## Launching Streamlit

```bash
streamlit run app.py
```