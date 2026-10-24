# 🩺 Multimodal Ensemble Skin Cancer Detection Pipeline

A production-ready, containerized Streamlit web application for automated skin lesion classification and risk assessment using a multi-model deep learning ensemble (MobileNetV2, ResNet50, and EfficientNetB0) combined with tabular clinical features.

---

## 📂 Project Architecture & Setup Guide

To run or containerize this application from scratch, the model weights must first be generated through the training experiments and correctly mapped.

### 1. Data Source (ISIC 2024 Dataset)
The models in this repository are designed to leverage data aligned with the **ISIC 2024 - Skin Cancer Detection with 3D-TBP** challenge. 
* You can access and download the dataset files via the official Kaggle competition page: [ISIC 2024 Challenge on Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge) or via mirrored community resources like the [All ISIC Data 2024 Repository](https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629/data).

### 2. Running Training Experiments & Organizing Artifacts
Before launching the app or container, execute your model experiment notebooks/scripts. Once trained, organize your output files so that each model sits cleanly in its respective subdirectory under a root `clean_models/` folder:

```text
skin-cancer-detection/
├── clean_models/
│   ├── baseline_mobilenetv2/
│   │   └── saved_model.pb (and associated variables/)
│   ├── resnet50_multimodal/
│   │   └── saved_model.pb (and associated variables/)
│   ├── research_efficientnetb0/
│   │   └── saved_model.pb (and associated variables/)
│   └── scaler.pkl