import streamlit as st
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Helper function to load images
def load_images_from_folder(folder, image_size=(150, 150)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path)
                img = img.resize(image_size)
                img = np.array(img)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load the trained model
model = load_model('dog_breed_classifier_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
data_dir = r'C:\Users\arsha\Downloads\archive (9)\dataset'  # Update the path to your dataset
_, y = load_images_from_folder(data_dir, (150, 150))
label_encoder.fit(y)

# Function to predict the breed of a given image
def predict_breed(image, model, label_encoder):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = image.reshape(1, 150, 150, 3)
    y_pred = model.predict(image)
    predicted_class = label_encoder.inverse_transform([np.argmax(y_pred)])
    return predicted_class[0]

# Streamlit app
st.title("Dog Breed Classifier")
st.write("Upload an image of a dog and the model will predict its breed.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the breed
    breed = predict_breed(image, model, label_encoder)

    # Display the prediction with a button
    btn = st.button("See Results!!")
    if btn:
        st.info(f"Predicted breed: {breed}")
        st.balloons()
