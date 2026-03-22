import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (28X28 grayscale).")

#Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("L").resize((28,28))
    img_array = np.array(image) / 255.0

    img_array = img_array.reshape(1,28,28,1).astype(np.float32)

    #Make prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    st.image(image, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {prediction}")