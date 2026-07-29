# Brain Tumor MRI Classification

<p align="center">
  <img src="assets/banner.png" width="100%" alt="Brain MRI Classification Banner">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

## Overview

An end-to-end deep learning project for **4-class brain MRI classification** using PyTorch. The project follows a production-oriented machine learning workflow from data validation and model development to explainability, FastAPI deployment, and Docker packaging.

### Highlights

- End-to-end PyTorch training pipeline
- Two custom CNN baselines
- ResNet18 transfer learning and fine-tuning
- Grad-CAM explainability
- FastAPI inference API
- Docker-ready deployment
- Reproducible experiment tracking

---

# Results

| Model | Accuracy | Precision | Recall | F1 | Loss |
|------|------:|------:|------:|------:|------:|
| CNN Baseline V1 | 81.12% | 82.73% | 81.21% | 81.15% | 0.528 |
| CNN Baseline V2 | 96.41% | 96.43% | 96.80% | 96.55% | 0.128 |
| ResNet18 Feature Extraction | 81.80% | 82.05% | 81.89% | 81.88% | 0.458 |
| **ResNet18 Fine-Tuned** | **97.78%** | **98.04%** | **98.00%** | **98.01%** | **0.066** |

---

# Dataset

Four-class Brain MRI dataset.

Classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

Dataset split:

| Split | Images |
|------|------:|
| Train | 9,401 |
| Validation | 2,351 |
| Test | 1,308 |

---

# Project Pipeline

<p align="center">
<img src="assets/readme/pipeline.png" width="95%">
</p>

---

# Model Architectures

## CNN Baseline V2

<p align="center">
<img src="assets/readme/cnn_v2.png" width="85%">
</p>

Features:

- Four convolution blocks
- Batch Normalization
- ReLU activations
- Max Pooling
- Adaptive Average Pooling
- Dropout regularization
- Fully-connected classifier

---

## ResNet18 Fine-Tuned

<p align="center">
<img src="assets/readme/resnet18.png" width="85%">
</p>

Training strategy:

1. Feature extraction
2. Progressive fine-tuning
3. Transfer learning with ImageNet weights
4. Adam optimizer
5. Learning-rate scheduling
6. Checkpointing

---

# Training Curves

<p align="center">
<img src="assets/readme/training_curve.png" width="90%">
</p>

---

# Model Comparison

<p align="center">
<img src="assets/readme/model_comparison.png" width="100%">
</p>

---

# Confusion Matrix

<p align="center">
<img src="assets/readme/confusion_matrix.png" width="75%">
</p>

---

# Explainability (Grad-CAM)

<p align="center">
<img src="assets/readme/grad_cam.png" width="100%">
</p>

Grad-CAM demonstrates that the fine-tuned ResNet18 focuses on anatomically relevant tumor regions rather than unrelated image background, increasing confidence in the model's decision-making process.

---

# API

FastAPI provides REST inference.

## Health

```bash
GET /health
```

## Prediction

```bash
POST /predict
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
-F "file=@sample.png"
```

Swagger UI

<p align="center">
<img src="assets/readme/swagger.png" width="95%">
</p>

---

# Project Structure

```text
brain-tumor-classification/
├── api/
├── artifacts/
├── configs/
├── experiments/
├── inference/
├── notebooks/
├── outputs/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── Dockerfile
├── requirements.txt
├── train.py
├── train_resnet18.py
├── train_resnet18_finetune.py
└── README.md
```

---

# Installation

```bash
git clone <repository-url>
cd brain-tumor-classification

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

# Training

CNN Baseline

```bash
python train.py
```

Transfer Learning

```bash
python train_resnet18.py
```

Fine-Tuning

```bash
python train_resnet18_finetune.py
```

---

# Evaluation

```bash
python evaluate.py

python resnet_evaluate.py
```

---

# Run API

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# Docker

```bash
docker build -t brain-tumor-api .

docker run -p 8000:8000 brain-tumor-api
```

---

# Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- FastAPI
- Uvicorn
- Docker

---

# Future Improvements

- EfficientNet-B0/B3
- Vision Transformer (ViT)
- Mixed precision training
- Hyperparameter optimization
- MLflow experiment tracking
- CI/CD pipeline
- Cloud deployment

---

# Acknowledgements

Thanks to the maintainers of the Brain MRI dataset and the PyTorch ecosystem.

---

# License

This project is released under the MIT License.
