# Handwritten-Digit-Recognition

This project utilizes a Convolutional Neural Network (CNN) model to recognize handwritten digits. The model is trained on a dataset of handwritten digits and can predict the digit in a given image with high accuracy.

## Usage

To use this project, you will need to run the `App.py` using command `streamlit run app.py`.This script utilizes the trained model to predict handwritten digits from images.
You can also explore the `testing.ipynb` and `training.ipynb` notebooks for testing and training the model, respectively.

## Model Architecture

The model is a sequential CNN with the following layers:
- An input layer designed to accept images of size 28x28.
- A flatten layer to convert the 2D image data into a 1D array.
- Dense layers with ReLU activation for feature extraction.
- A final dense layer with softmax activation to classify the digits into one of the ten categories (0-9).

For more details on the model architecture, refer to the `sheep_one.model/keras_metadata.pb` file.
