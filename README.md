# 🧠 NeuroVision-AI
### Deep Learning Framework for Alzheimer's Disease Detection using Structural MRI

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Medical Imaging](https://img.shields.io/badge/Medical-Imaging-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</p>

---

## 📖 Overview

NeuroVision-AI is a modular deep learning framework for Alzheimer's Disease classification using structural MRI scans. The project integrates MRI preprocessing, convolutional neural networks, explainable AI, and an interactive Streamlit interface to demonstrate an end-to-end medical imaging workflow.

The repository is designed for research, experimentation, and educational purposes, with an emphasis on reproducible preprocessing and interpretable predictions.

---

## ✨ Features

- 🧠 Structural MRI preprocessing pipeline
- 📂 DICOM → NIfTI conversion
- 🖼 MRI slice extraction and normalization
- 🔥 Deep Learning using PyTorch
- 🏗 Configurable CNN backbones
- 📊 Training and evaluation pipeline
- 📈 TensorBoard logging
- 🎯 Grad-CAM explainability
- 🌐 Streamlit web application
- ⚙️ Modular project structure
- 🔬 Research-oriented workflow

---

# 🏗 Project Architecture

<p align="center">

<img src="assets/architecture.png" width="900">

</p>

---

# 🔄 End-to-End Pipeline

<p align="center">

<img src="assets/pipeline.png" width="950">

</p>

The complete workflow consists of:

1. MRI acquisition
2. DICOM → NIfTI conversion
3. MRI preprocessing
4. Slice extraction
5. Data augmentation
6. CNN training
7. Evaluation
8. Explainability using Grad-CAM
9. Interactive prediction using Streamlit

---

# 📂 Repository Structure

```text
NeuroVision-AI
│
├── assets/
├── checkpoints/
├── configs/
├── data/
├── docs/
├── notebooks/
├── results/
├── src/
│
├── preprocessing.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── gradcam.py
├── app.py
├── config.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🧬 Dataset

The project is designed to work with the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset.

Due to licensing restrictions, the dataset is **not included** in this repository.

Official website:

https://adni.loni.usc.edu/

Expected directory structure:

```text
data/

    raw/

        CN/

        MCI/

        AD/

    processed/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/NeuroVision-AI.git

cd NeuroVision-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

```bash
python train.py
```

---

# 📊 Evaluation

```bash
python evaluate.py
```

---

# 🔍 Prediction

```bash
python predict.py
```

---

# 🌐 Launch Streamlit App

```bash
streamlit run app.py
```

---

# 🧠 Explainability

Grad-CAM is integrated to visualize the regions of the MRI that contribute most to the model's predictions.

<p align="center">

<img src="assets/gradcam_example.png" width="700">

</p>

---

# 💻 Streamlit Demo

<p align="center">

<img src="assets/prediction.png" width="700">

</p>

The web application allows users to:

- Upload MRI scans
- Perform inference
- Visualize prediction confidence
- Inspect Grad-CAM heatmaps

---

# 📚 Documentation

Detailed documentation is available in the `docs/` folder.

- architecture.md
- dataset.md
- training.md

---

# 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Nibabel
- Streamlit
- Matplotlib

---

# 🔮 Future Work

- 3D CNN models
- Vision Transformers (ViT)
- MONAI integration
- Docker deployment
- REST API
- Cloud inference
- Clinical feature fusion
- Model quantization

---

# 📜 License

This project is licensed under the MIT License.

---

# ⚠️ Disclaimer

This repository is intended for research and educational purposes only.

It is **not** a certified medical diagnostic system and should not be used for clinical decision-making.

---

# 🙋 Author

**Gauri Kailash Soni**

B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)

Interested in Artificial Intelligence, Medical Imaging, Computer Vision, and Applied Machine Learning.

- GitHub: https://github.com/gaurisoonii
- LinkedIn: https://www.linkedin.com/in/gauri-soni/

---

## ⭐ If you find this project useful

Please consider giving the repository a ⭐ on GitHub.