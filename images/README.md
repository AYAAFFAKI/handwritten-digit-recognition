# Handwritten Digit Recognition

This project implements a convolutional neural network (CNN) to recognize handwritten digits (0-9) using the MNIST dataset. The model is built with TensorFlow and Keras.

---

## Project Structure

- `model.py` — Script to build, train, and save the CNN model.
- `predict.py` — Script to load the trained model and predict digits from images.
- `requirements.txt` — Required Python libraries.
- `images/` — Contains sample images of handwritten digits used to test and visualize the model's predictions.

---

## About the `images/` Folder

The `images/` directory includes various digit images (e.g., `0.jpg`, `1.jpg`, ..., `9.jpg`) that you can use to verify how well the model performs on handwritten digit samples.

You can run the `predict.py` script to classify these images and observe the predicted digits along with their confidence scores.

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

Train the model (optional if you already have mnist.h5):
python model.py

Predict digits on sample images:
python predict.py


Example Output
6:
6 0.62
----
8:
8 0.47
----

Feel free to add your own handwritten digit images into the images/ folder and test the model's predictions!

Author
AYA AFFAKI

