import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# Load your trained model
custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
model = tf.keras.models.load_model('ResNet152V2.h5')


# Define class labels of the animals
class_labels = ['Butterfly', 'Cat', 'Cow', 'Dog', 'Hen']

# Streamlit App
st.title("Image Classification App")

# Upload image through Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the bytes of the uploaded file
    image_bytes = uploaded_file.read()

    # Convert the bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    image = image.resize((256, 256))  # Adjust size as needed
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize the pixel values to be between 0 and 1

    # Make predictions
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display the predicted class and confidence
    st.write("Prediction:")
    st.write(f"Class: {class_labels[predicted_class]}, Confidence: {confidence:.2f}")
