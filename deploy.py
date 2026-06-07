import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import custom_object_scope

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Define the custom InstanceNormalization layer directly
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, gamma_initializer='ones', **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        # Calculate mean and variance along spatial dimensions (height and width)
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# 2. Cache the model loading
@st.cache_resource
def load_gan_model():
    # Register both the custom layer AND the initializer class
    with custom_object_scope({
        'InstanceNormalization': InstanceNormalization,
        'RandomNormal': tf.keras.initializers.RandomNormal
    }):
        return tf.keras.models.load_model("./models/AnimeGenTrain_55.h5")

# Load it once, use it forever
gan_model = load_gan_model()

# Define the preprocess functions
def load(face):
    face = tf.image.resize(face, size=[128, 128])
    return face

def normalize(face):
    face = (tf.cast(face, tf.float32) / 255.0 * 2) - 1
    return face

def preprocess(face):
    face = load(face)
    face = normalize(face)
    return face

def translate_to_comic(image_array, IMAGE_SIZE):
    # Load and preprocess the image
    input_image = preprocess(image_array)

    # Expand dimensions to match the model's expected input shape
    input_image = tf.expand_dims(input_image, 0)

    # Use your GAN model to translate the image
    translated_image = gan_model.predict(input_image)

    # Remove the batch dimension
    translated_image = tf.squeeze(translated_image, axis=0)

    # Denormalize the image back to [0, 255]
    translated_image = (translated_image + 1) / 2.0 * 255.0

    # Resize back to original image dimensions
    translated_image = tf.image.resize(translated_image, size=[IMAGE_SIZE[1], IMAGE_SIZE[0]]) 

    return translated_image.numpy().astype(np.uint8)

def main():
    st.set_page_config(page_title="Real-to-Comic", layout="wide")
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
        # Open and format the image
        image = Image.open(uploaded_file)
        if image.format == "PNG":
            image = image.convert("RGB")

        IMAGE_SIZE = image.size
        image_array = np.array(image)
        
        # Add a loading spinner while the model predicts
        with st.spinner("Translating to comic style..."):
            translated_image = translate_to_comic(image_array, IMAGE_SIZE)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Translated Image")
            st.image(translated_image, use_column_width=True)

if __name__ == "__main__":
    main()