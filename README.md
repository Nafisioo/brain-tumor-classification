# 🧠 Brain Tumor MRI Classification

<p align="center">
  <!-- Note: Ensure you have a banner.png in your assets folder, or remove this line -->
  <img src="assets/banner.png" width="100%" alt="Brain MRI Classification Banner">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **Author:** [Nafise Bahoosh](https://github.com/Nafisioo) | **Portfolio Project**

## 📑 Overview

An end-to-end deep learning pipeline for **4-class brain MRI classification** built with PyTorch. This project bridges the gap between research and production by implementing a complete machine learning lifecycle: starting from custom baseline CNNs, advancing to fine-tuned ResNet18 architectures, validating predictions with Grad-CAM explainability, and deploying the final model via a robust, Dockerized FastAPI microservice.

### 🚀 Key Highlights
- **Model Development:** Designed custom CNN baselines and utilized transfer learning (ResNet18) to achieve **97.78% accuracy**.
- **Model Interpretability:** Integrated **Grad-CAM** to visualize spatial attention, ensuring the model focuses on clinically relevant tumor regions rather than background noise.
- **Production-Ready Deployment:** Built a highly concurrent REST API using **FastAPI** and packaged the environment via **Docker** for seamless deployment.
- **Reproducibility:** Structured following MLOps best practices with modularized code, strict environment management, and comprehensive evaluation scripts.

---

## 📊 Results & Performance

The fine-tuned ResNet18 significantly outperformed the custom baselines, demonstrating the power of transfer learning on medical imaging data.

| Model | Accuracy | Precision | Recall | F1 Score | Test Loss |
|:---|---:|---:|---:|---:|---:|
| CNN Baseline V1 | 81.12% | 82.73% | 81.21% | 81.15% | 0.528 |
| CNN Baseline V2 | 96.41% | 96.43% | 96.80% | 96.55% | 0.128 |
| ResNet18 (Feature Extraction) | 81.80% | 82.05% | 81.89% | 81.88% | 0.458 |
| **ResNet18 (Fine-Tuned)** | **97.78%** | **98.04%** | **98.00%** | **98.01%** | **0.066** |

<p align="center">
  <img src="assets/model_comparison.png" width="100%" alt="Model Performance Comparison">
</p>

---

## 🏗️ System Architecture

<p align="center">
  <img src="assets/architecture.png" width="60%" alt="System Architecture">
</p>

---

## 💾 Dataset

The model is trained on a multi-class Brain MRI dataset containing four distinct categories.

- **Classes:** `Glioma`, `Meningioma`, `Pituitary Tumor`, `No Tumor`
- **Total Images:** 13,060

| Split | Image Count | Percentage |
|:---|---:|---:|
| **Train** | 9,401 | ~72% |
| **Validation** | 2,351 | ~18% |
| **Test** | 1,308 | ~10% |

---

## 🧠 Model Architectures

### CNN Baseline V2
A custom-built Convolutional Neural Network designed to establish a performance baseline.
* **Features:** 4 Convolutional blocks, Batch Normalization, ReLU activations, Max Pooling, Adaptive Average Pooling, and Dropout regularization for overfitting prevention.

<p align="center">
  <img src="assets/cnn_v2.png" width="40%" alt="CNN Baseline V2 Architecture">
</p>

### ResNet18 Fine-Tuned (Production Model)
Leveraging ImageNet weights, the model was adapted for medical imaging.
* **Strategy:** Initial feature extraction followed by progressive fine-tuning of deeper layers using the Adam optimizer and dynamic learning-rate scheduling.

<p align="center">
  <img src="assets/resnet18.png" width="40%" alt="ResNet18 Architecture">
</p>

---

## 📈 Evaluation & Interpretability

### Training Convergence
<p align="center">
  <img src="assets/training_curve.png" width="85%" alt="Training Curves">
</p>

### Confusion Matrix
The model exhibits high confidence across all classes, with minimal false positives between tumor types.
<p align="center">
  <img src="assets/confusion_matrix.png" width="65%" alt="Confusion Matrix">
</p>

### Explainable AI (Grad-CAM)
In medical AI, trust is paramount. Grad-CAM visualizes the gradients flowing into the final convolutional layer, highlighting the specific regions of the MRI scan that triggered the model's prediction. 

<p align="center">
  <img src="assets/grad_cam.png" width="85%" alt="Grad-CAM Visualization">
</p>
*Observation: The heatmap clearly localizes the tumor mass, confirming the model relies on pathological features rather than dataset artifacts.*

---

## 🌐 API & Deployment

Inference is served via a high-performance **FastAPI** application.

### API Endpoints
* `GET /health`: Returns API status and loaded model details.
* `POST /predict`: Accepts a multipart form data image and returns JSON with class predictions and confidence scores.

### Local Testing with cURL
```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_mri.png;type=image/png"