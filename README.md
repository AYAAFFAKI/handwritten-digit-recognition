# Handwritten Digit Recognition

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) using the MNIST dataset. The model is built and trained using Python and TensorFlow/Keras.

---

## 🔍 Project Overview

- **Dataset:** MNIST — a large database of handwritten digits.
- **Model:** CNN with two convolutional layers, max pooling, dense layers, and dropout for regularization.
- **Goal:** Achieve accurate digit classification from grayscale images.
- **Usage:** Predict handwritten digits from input images and output predicted digit with confidence score.

---

## 🚀 How to Use

Clone the repository:
git clone https://github.com/AYAAFFAKI/handwritten-digit-recognition.git
cd handwritten-digit-recognition

Install dependencies:
pip install -r requirements.txt

Train the model:
python model.py

Predict digits on sample images:
python predict.py


📂 Repository Structure
bash
Copier
Modifier
handwritten-digit-recognition/
│
├── model.py            # Model definition and training script
├── predict.py          # Script for loading model and predicting digits from images
├── requirements.txt    # Project dependencies
├── images/             # Sample digit images for prediction
├── mnist.h5            # Saved trained model weights
└── README.md           # This file


📈 Results
Achieved over 90% accuracy on MNIST test dataset.
Supports prediction on custom images with preprocessing.

🤝 Contributions
Feel free to fork the repository and submit pull requests for improvements or fixes.

📞 Contact
Created by Aya Affaki
GitHub: AYAAFFAKI

Thank you for checking out this project!
