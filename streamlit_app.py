import streamlit as st
import tensorflow as tf
import pathlib
import os
from tensorflow import keras
import numpy as np
import pickle
import random
import time
from PIL import Image
import gdown

st.set_page_config(
    page_title="Preform Classifier",
    page_icon="Logo/machine-learning.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "This app is a demonstration of the preform classification model trained using a Convolutional Neural Network"
    }
)

with st.sidebar:
    st.page_link('streamlit_app.py', label='Preform Classifier', icon='🤖')

st.logo(image='Logo/machine-learning.png', 
        icon_image='Logo/machine-learning.png')


def pad_image_to_size(image_id, target_size=(512, 512)):
    img = tf.keras.preprocessing.image.load_img(image_id)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    height, width, channels = img_array.shape  # Get channels

    # Calculate padding
    pad_height = max(0, target_size[1] - height)
    pad_width = max(0, target_size[0] - width)

    # Pad the image
    padded_img = tf.image.pad_to_bounding_box(
        img_array,
        pad_height // 2,  # Top padding
        pad_width // 2,   # Left padding
        target_size[1],   # Target height
        target_size[0]    # Target width
    )

    # Normalize the image
    normalized_img = padded_img / 255.0

    return tf.keras.preprocessing.image.array_to_img(normalized_img)

def predict_class(image):
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = array / 255.0  # Normalize pixel values
    image = np.expand_dims(array, axis=0)
    return label_mapping[np.argmax(model.predict(image))]

def random_image(directory = 'test_img'):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    random_image = random.choice(image_files)
    return os.path.join(directory, random_image)


def get_class(image):
    return image.split('-')[0].split("/")[-1]

@st.cache_resource
def load_model(model_path = 'Model/my_model.keras'):
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1oL_JMifBpYckyvpUBzN6wZzG2iTDqHhK"
    output = "CNN_model.keras"
    gdown.download(url=url, output=output)


download_model()

model = load_model(model_path='CNN_model.keras')

label_mapping = {
    0: 'Blackspot',
    1: 'Degrade',
    2: 'Good',
    3: 'Haziness',
    4: 'Watermark'
}


#Get random image
if 'random_image' not in st.session_state:
    st.session_state['random_image'] = random_image()

#procsess image for prediction using model
processed_img = pad_image_to_size(st.session_state.random_image)



st.title("Preform Classifier :camera:")
 
left, right = st.columns(2)


#Display random image with caption
with st.container():
    with left:
        shuffle = st.button("New image", icon = "🔀")
        if shuffle:
            st.session_state['random_image'] = random_image()
        st.image(st.session_state.random_image, caption=get_class(st.session_state.random_image))


    with right:
        predict = st.button("Predict", icon="🤖")
        st.header("Prediction:")
        if predict:
            with st.spinner("Predicting..."):
                time.sleep(1)
            st.header(f"_**{predict_class(processed_img)}**_")
