import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

#Load the trained model
model = load_model("mnist_model.keras")

st.title("MNIST Digit Classifier")

#Upload image
uploaded_file = st.file_uploader("Upload a MNIST image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")
    st.write("Prediction:", digit)