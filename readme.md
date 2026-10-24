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

Once your models are positioned in clean_models/, you can successfully run the app, test scripts, or the model comparison file.

🚀 Running Locally
Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

🐳 Running with Docker
Build the Docker image:
docker build -t skin-cancer-detector .

Run the container:
docker run -p 8501:8501 skin-cancer-detector

Open your browser and navigate to http://localhost:8501


📚References & Research Background
This project draws upon established deep learning architectures and dermatological research:

Skin Cancer Detection & Deep Learning Overview:

Explorations into deep learning architectures for automated skin lesion classification, such as frameworks outlined in literature reviewing computer-aided dermatological diagnosis.

ResNet Architecture Foundation:

The residual learning framework utilized by the ResNet50 backbone is based on Kaiming He et al., Deep Residual Learning for Image Recognition (CVPR 2016). Available at IEEE Xplore.

ISIC Benchmarks:

Dataset context originates from the International Skin Imaging Collaboration (ISIC) archives.