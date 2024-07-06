import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('sheep_one.model')

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to match the model's expected sizing
    image = cv2.resize(image, (28, 28))
    # Invert colors (make it white on black)
    image = cv2.bitwise_not(image)
    # Reshape and normalize the image
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image

# Function to make prediction and display result
def predict_and_display(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    display_results(image, prediction)

# Function to display result
def display_results(img, result):
    st.image(img, caption="Drawn Digit", use_column_width=True)
    result_lbl = np.argmax(result)
    st.write(f"Predicted Digit: {result_lbl}")

# Main Streamlit app
def main():
    st.title("Handwritten Digit Recognition")

    # Create a canvas for drawing
    st.sidebar.write("Draw a digit on the canvas:")
    uploaded_file = st.file_uploader("Choose a file")

    # When the user clicks on the "Predict" button
    if st.button("Predict"):
        if uploaded_file is not None:
            # Convert uploaded file to NumPy array
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            # Make prediction and display result
            predict_and_display(img_array)

# Run the Streamlit app
if __name__ == "__main__":
    main()
