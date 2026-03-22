import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.title("MNIST Digit Classifier")

#Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

#Build Model
model = keras.Sequential([
    
    #Block 1
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    #Block 2
    keras.layers.Conv2D(64, (3,3), activation='relu'), 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    #Block 3
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    #Fully connected
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(10, activation='softmax')
])

#Compile
model.compile(
    optimizer="adam",

loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

#Train
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

#Upload image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")
    st.write("Prediction:", digit)