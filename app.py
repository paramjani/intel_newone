import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random

# Disable scientific notation
np.set_printoptions(suppress=True)

# Set Streamlit page
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")
st.title("InspectorsAlly")
st.caption("Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App")
st.write("Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly.")

# Sidebar - Show random sample images from dataset
with st.sidebar:
    st.subheader("Sample Images from Dataset")

    good_folder = "./good"
    bad_folder = "./bad"

    good_images = os.listdir(good_folder) if os.path.exists(good_folder) else []
    bad_images = os.listdir(bad_folder) if os.path.exists(bad_folder) else []

    if good_images and bad_images:
        good_sample = Image.open(os.path.join(good_folder, random.choice(good_images)))
        bad_sample = Image.open(os.path.join(bad_folder, random.choice(bad_images)))

        col1, col2 = st.columns(2)
        with col1:
            st.image(good_sample, caption="✅ Good", use_column_width=True)
        with col2:
            st.image(bad_sample, caption="❌ Bad", use_column_width=True)
    else:
        st.warning("Sample images not found in 'good' or 'bad' folder.")

    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections."
    )
    st.write(
        "This advanced inspection app uses deep learning models to inspect wooden products for defects like cracks, warping, and discoloration."
    )

# Load the model and labels
model = tf.keras.models.load_model("keras_model.h5")

try:
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("Error: 'labels.txt' file not found in the project directory.")
    st.stop()

# Preprocess function
def preprocess_image(image_pil):
    # Convert to RGB if image is RGBA or L (grayscale)
    if image_pil.mode in ["RGBA", "L"]:
        image_pil = image_pil.convert("RGB")

    image = image_pil.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return np.expand_dims(image, axis=0)

# Prediction logic
def Anomaly_Detection_Keras(image_pil):
    img_array = preprocess_image(image_pil)
    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = labels[predicted_class_idx]

    if predicted_class.lower() == "good":
        return "✅ Congratulations! Your wooden product has been classified as 'Good' with no anomalies detected."
    else:
        return "⚠️ Our AI inspection system has detected an anomaly in your wooden product."

# Image Input Method
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = Image.open(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        camera_file_img = Image.open(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Submit Button
if st.button("Submit a Wooden Product Image"):
    st.subheader("Output")

    if input_method == "File Uploader" and uploaded_file_img:
        image = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img:
        image = camera_file_img
    else:
        image = None

    if image:
        with st.spinner("This may take a moment..."):
            result = Anomaly_Detection_Keras(image)
            st.write(result)
    else:
        st.warning("No image provided.")
