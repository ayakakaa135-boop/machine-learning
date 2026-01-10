# 🧠 Machine Learning Engine: Hybrid Skin Lesion Classification

This directory contains the end-to-end pipeline for training the DermAI diagnostic engine. The project focuses on solving the challenge of skin lesion classification using a multi-modal approach.

## 🏗️ Architecture: The Hybrid Approach
Unlike standard models that rely solely on images, this engine uses a **Functional API** to process two distinct types of data simultaneously:

1.  **Computer Vision Branch:** * **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
    * **Input:** 128x128x3 RGB Images.
    * **Role:** Extracts deep spatial features from dermoscopy images.
2.  **Metadata Branch:**
    * **Structure:** Multi-layer Perceptron (MLP).
    * **Input:** 19-dimensional vector (Age, Sex, Anatomical Site).
    * **Role:** Integrates clinical context to refine the final prediction.

## 🧪 Data Engineering
* **Dataset:** HAM10000 (Human Against Machine).
* **Pre-processing:** * Image normalization (1./255).
    * Metadata encoding (One-Hot Encoding for categorical variables).
    * Handling Data Imbalance using **Class Weights** to ensure high sensitivity for critical classes like *Melanoma*.

## 📉 Training Details
* **Loss Function:** Categorical Crossentropy.
* **Optimizer:** Adam ($\alpha = 0.001$).
* **Callbacks:** `ModelCheckpoint` to save the best performing weights based on validation loss.

## 📈 Key Achievements
* Successfully integrated metadata to reduce false negatives in malignant cases.
* Optimized for mobile-ready inference using the MobileNetV2 architecture.
