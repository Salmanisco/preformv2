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

left, right = st.columns(2)
with left:
    st.title("""Special thanks:""")
with right:
    st.image("images/Genoa-GPI-Official-Logo-mod.png")

st.header("General Manager: Mr. Abdullah Alfares")
st.header("IT Manager: Mr. Ahmad Adel")
