# 🩺 Chest X-Ray Pneumonia Detection System

## 📋 Project Overview

A comprehensive deep learning pipeline for automated detection of pneumonia from chest X-ray images.  
This project follows clean machine learning and software engineering practices, combining **Object-Oriented Programming (OOP)**, **unit testing with PyTest**, and **Explainable AI (XAI)**, along with a **Streamlit web application** for real-time predictions.

The system covers:
- Exploratory Data Analysis (EDA)
- Image preprocessing & data augmentation
- CNN-based DenseNet121 model
- Explainability using Grad-CAM & SHAP
- Modular, testable, and maintainable codebase
- Interactive web deployment using Streamlit

------------------------------------------------------------------------

## 🩺 Medical Summary

-   **Dataset Size:** 5,863 chest X-ray images (JPEG)
-   **Age Group:** Pediatric patients (1–5 years)
-   **Classes:**
    -   Normal: 1,349
    -   Pneumonia: 3,876
-   **Class Imbalance Ratio:** 2.87 : 1
-   **Source:** Guangzhou Women and Children's Medical Center
-   **License:** Licensed under the **MIT License**.
-   **Use Case:** AI-assisted pneumonia detection

------------------------------------------------------------------------

## 🖥️ Frontend & User Interface

-   **Web App:** Built using Streamlit for interactive real time predictions
-   **Design & Prototyping:** UI/UX designed in Figma
-   **Styling:** Custom CSS for layout, colors, and responsiveness

------------------------------------------------------------------------

## 📄 Report & Documentation

-   **Report Format:** LaTeX, created and compiled on Overleaf
-   Includes: project overview, methodology, model explanation, metrics, visualizations, and results

------------------------------------------------------------------------

## 💻 Tech Stack

-   **Backend / Model:** Python, TensorFlow, Keras, NumPy, OpenCV, Matplotlib
-   **Frontend / Visualization:** Streamlit, CSS, Figma
-   **Explainable AI:** Grad-CAM, SHAP
-   **Documentation:** LaTeX (Overleaf)

## 🚀 Quick Start

### 🔧 Prerequisites

-   Python **3.8+**
-   Git
-   Virtual environment support
-   Kaggle API (Not mandatory, for dataset download)

------------------------------------------------------------------------

### 📥 Clone Repository

``` bash
git clone https://github.com/SeyedRumaiz/pneumonia-image-classification.git
cd pneumonia-image-classification
```

### Install dependencies

```         
pip install -r requirements.txt
```
