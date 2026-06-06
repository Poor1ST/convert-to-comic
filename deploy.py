import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import custom_object_scope
from tensorflow_addons.layers import InstanceNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Register custom layers before loading the model
with custom_object_scope({'InstanceNormalization': InstanceNormalization}):
    gan_model = tf.keras.models.load_model("./models/FaceGenTrain_45.h5")

# Define the preprocess functions
def load(face):
    # face = tf.io.read_file(face_path)
    # face = tf.convert_to_tensor(face)
    # face = tf.image.decode_jpeg(face, channels=3)
    face = tf.image.resize(face, size=[128, 128])

    return face

def normalize(face):
    face = (tf.cast(face, tf.float32) / 255.0 * 2) - 1
    return face

def preprocess(face):
    face = load(face)
    face = normalize(face)
    return face

def translate_to_comic(image_path, IMAGE_SIZE):
    # Load and preprocess the image
    input_image = preprocess(image_path)

    # Expand dimensions to match the model's expected input shape
    input_image = tf.expand_dims(input_image, 0)

    # Use your GAN model to translate the image
    translated_image = gan_model.predict(input_image)

    # Remove the batch dimension
    translated_image = tf.squeeze(translated_image, axis=0)

    # Denormalize the image
    translated_image = (translated_image + 1) / 2.0 * 255.0

    translated_image = tf.image.resize(translated_image, size=[IMAGE_SIZE[1], IMAGE_SIZE[0]]) 

    return translated_image.numpy().astype(np.uint8)


def main():
    st.title("Real-to-Comic Image Translator")

    # Choose the mode
    mode = st.sidebar.selectbox("Choose the mode", ("Upload", "Camera"))
    uploaded_file = None

    if mode == "Upload":
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    elif mode == "Camera":
        st.subheader("Take a photo")
        uploaded_file = st.camera_input('Take a photo')

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PNG to JPG
        if image.format == "PNG":
            image = image.convert("RGB")

        IMAGE_SIZE = image.size

        # Translate the image to comic style
        image = np.array(image)
        translated_image = translate_to_comic(image, IMAGE_SIZE)

        # Display the original and translated images side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("Translated Image")
            st.image(translated_image, caption="Comic Style", use_column_width=True)

if __name__ == "__main__":
    main()
