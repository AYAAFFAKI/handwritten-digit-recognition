# Import necessary libraries from Keras and TensorFlow
from tensorflow.keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Load MNIST dataset: 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data to fit the input format of CNN (batch_size, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vectors to one-hot encoded format (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalize image data to float32 and scale to [0, 1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Print dataset shapes for verification
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Set training parameters
batch_size = 128
num_class = 10
epochs = 160

# Build the CNN model
model = Sequential()

# First convolutional layer: 32 filters, 5x5 kernel, ReLU activation
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

# First max-pooling layer: 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer: 64 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Second max-pooling layer: 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten feature maps into 1D vector before dense layers
model.add(Flatten())

# First fully connected layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Dropout layer to prevent overfitting (30% dropout rate)
model.add(Dropout(0.3))

# Second fully connected layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu'))

# Dropout layer (50% dropout rate)
model.add(Dropout(0.5))

# Output layer with 10 neurons (one for each digit) and softmax activation
model.add(Dense(num_class, activation='softmax'))

# Compile the model with categorical crossentropy loss and Adadelta optimizer
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

# Add EarlyStopping to prevent overfitting if validation loss doesn't improve
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with validation on test set
hist = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

print("The model has successfully trained")

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the trained model to disk
model.save('mnist.h5')
print("Saving the model as mnist.h5")
