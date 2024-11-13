import streamlit as st
import tensorflow as tf
import pathlib
import os
from tensorflow import keras
import numpy as np
import pickle
import random

st.logo(image='Logo/machine-learning.png', 
        icon_image='Logo/machine-learning.png')

with st.container():
    st.header("Sample Collection")

    st.image("images/Samples.png")

    watermark, haziness, blackspot, degrade, good = st.columns(5)
    with watermark:
        st.image("images/watermark.jpg", caption="Watermark")
    with haziness:
        st.image("images/Haziness.jpg", caption="Haziness")
    with blackspot:
        st.image("images/Blackspot.jpg", caption="Blackspot")
    with degrade:
        st.image("images/Degrade.jpg", caption="Degrade")
    with good:
        st.image("images/Good.jpg", caption="Good")

with st.container():
    st.header("Training :chart_with_upwards_trend: :chart_with_downwards_trend:")
    accuracy, loss = st.columns(2)
    with accuracy:
        st.image("images/Accuracy.png")
    with loss:
        st.image("images/Loss.png")

with st.container():
    st.header("Testing")

    st.image("images/Confusion_Matrix.png")