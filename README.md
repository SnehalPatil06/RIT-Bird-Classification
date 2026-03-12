# 🦅 Bird Species Classification Using Deep Learning

> A Convolutional Neural Network (CNN) model to automatically identify and classify **260 bird species** from images using TensorFlow and Keras.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![CNN](https://img.shields.io/badge/Model-CNN-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🏛️ Institution

**K.E. Society's Rajarambapu Institute of Technology, Rajaramnagar**  
Department of Computer Engineering  
Project Guide: **Prof. P.R. Gavali**

---

## 👥 Team Members

| Name | Roll No. |
|------|----------|
| Vaibhav Raju Kolekar | 1804034 |
| Shubham Shankar Patil | 1804066 |
| Snehal Rajgonda Patil | 1954011 |
| Bhagyashri Vijay Suryawanshi | 1954015 |

---

## 📌 Project Overview

Many people visiting bird sanctuaries cannot identify bird species without expertise in ornithology. This project builds an **automated bird species classifier** using Deep Learning that can identify a bird species from just a photograph.

The model uses a **Convolutional Neural Network (CNN)** trained on a dataset of **260 bird species** with over **36,000 images**.

---

## 🌐 Live Demo

Open `bird_species_classifier.html` in any browser to try the interactive demo — no installation needed!

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total Species | 260 |
| Training Images | 36,609 |
| Test Images | 1,300 (5 per species) |
| Validation Images | 1,300 (5 per species) |
| Image Size | 224 × 224 × 3 (JPG) |

---

## 🧠 CNN Architecture

```
Input (224×224×3)
     ↓
Convolutional Layer  →  Feature extraction
     ↓
Activation Layer (ReLU)  →  Non-linearity
     ↓
Pooling Layer (MaxPool 2×2)  →  Dimensionality reduction
     ↓
Fully Connected Layer (Dense 1024)  →  Classification
     ↓
Output Layer (Softmax, 260 classes)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** TensorFlow 2.x, Keras
- **Data Processing:** NumPy, Pandas
- **Visualisation:** Matplotlib
- **Environment:** Anaconda

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/RIT-Bird-Classification.git
cd RIT-Bird-Classification

# 2. Create a virtual environment
conda create -n bird-classifier python=3.8
conda activate bird-classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
open bird_species_classifier.html
```

---

## 📁 Project Structure

```
RIT-Bird-Classification/
│
├── bird_species_classifier.html   # Interactive web demo
├── model/
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Prediction script
│   └── cnn_model.py               # CNN architecture definition
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~94% |
| Validation Accuracy | ~91% |
| Test Accuracy | ~90% |
| No. of Classes | 260 |

---

## 🎯 Objectives

1. Raise awareness about bird species in various regions
2. Develop an automated model capable of identifying bird species from images
3. Improve upon state-of-the-art bird species classifiers using CNN
4. Build a system based on the CNN algorithm using TensorFlow
5. Increase accuracy of detecting bird species

---

## 📚 References

1. Tóth, B.P. and Czeba, B. (2016). *Convolutional Neural Networks for Large-Scale Bird Song Classification.* CLEF Working Notes.
2. Fagerlund, S. (2007). *Bird species recognition using support vector machines.* EURASIP Journal.
3. Cireşan, D., Meier, U. and Schmidhuber, J. (2012). *Multi-column deep neural networks for image classification.* arXiv:1202.2745.

---

## 📄 License

This project is licensed under the MIT License.
