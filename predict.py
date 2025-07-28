from keras.models import load_model
from PIL import ImageGrab, Image, ImageOps, ImageChops
import numpy as np

# Load the pre-trained CNN model
model = load_model('mnist.h5')


def predict_digital(img):
    """
    Preprocess the image and predict the handwritten digit using the loaded model.

    Parameters:
    img (PIL.Image): Input image containing a handwritten digit.

    Returns:
    tuple: Predicted digit and its confidence score.
    """
    # Resize image to 28x28 pixels as required by the model
    img = img.resize((28, 28))

    # Convert image to grayscale (1 channel)
    img = img.convert('L')

    # Invert the image colors: white background → black, black digit → white
    img = ImageOps.invert(img)

    # Convert image to numpy array
    img = np.array(img)

    # Reshape image to match model input: (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Perform prediction
    res = model.predict([img])[0]

    # Return the predicted digit and confidence score
    return np.argmax(res), max(res)


# Predict multiple digits and display their predictions and confidence

number, conf = predict_digital(Image.open('6.jpg'))
print('6:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('8.jpg'))
print('8:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('2.jpg'))
print('2:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('3.jpg'))
print('3:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('0.jpg'))
print('0:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('5.jpg'))
print('5:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('9.jpg'))
print('9:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('4.png'))
print('4:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('1.jpg'))
print('1:')
print(number, conf)
print('----')

number, conf = predict_digital(Image.open('7.jpg'))
print('7:')
print(number, conf)
