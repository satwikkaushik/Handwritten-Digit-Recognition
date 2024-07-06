import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf

# Pre-trained model
model = tf.keras.models.load_model('sheep_one.model')

def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    
    # Invert colors
    image = cv2.bitwise_not(image)
    
    # Resize
    image = image.reshape(1, 28, 28, 1)
    
    # Normalize pixels
    image = image / 255.0
    return image

def predict_and_display(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    display_results(prediction)

# This function displays the output on streamlit app
def display_results(result):
    result_lbl = np.argmax(result)
    st.write(f"**Predicted Digit:** {result_lbl}", font="HelveticaNeue-Bold, sans-serif")

# Main
def main():
    st.title("Handwritten Digit Recognition")

    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.8)",
        stroke_width=10,
        stroke_color="black",
        background_color="#eee",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Handling Recognise Button
    if st.button("Recognize"):
        # Convert drawn to array
        img_array = np.array(canvas_result.image_data)
        img_array = cv2.resize(img_array, (28, 28))

        predict_and_display(img_array)

# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
