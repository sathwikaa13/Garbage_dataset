import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

# Define paths
MODEL_PATH = "garbage_classifier_model.h5"
DATASET_PATH = "dataset"
CLASS_LABELS = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Load model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("🚫 Model file not found!")
    st.stop()

# Sidebar: App mode
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section:", ["Predict Waste Category", "Dataset Analysis"])

# ----------------------------- Prediction Section -----------------------------
if app_mode == "Predict Waste Category":
    st.title("🗑 Garbage Image Classifier")
    st.write("Upload an image and the model will classify it into one of the 10 garbage categories.")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🧠 Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Show all class probabilities
        st.subheader("Class Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{CLASS_LABELS[i]}: {prob * 100:.2f}%")

# ----------------------------- Dataset Analysis Section -----------------------------
elif app_mode == "Dataset Analysis":
    st.title("📊 Garbage Dataset Analysis")

    if not os.path.exists(DATASET_PATH):
        st.error(f"🚫 Dataset folder '{DATASET_PATH}' not found!")
        st.stop()

    class_dirs = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    class_counts = {}
    image_formats = []

    for cls in class_dirs:
        folder = os.path.join(DATASET_PATH, cls)
        images = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        class_counts[cls] = len(images)

        for img_file in images[:3]:  # Sample a few to avoid slowdown
            try:
                img = Image.open(os.path.join(folder, img_file))
                image_formats.append((cls, img.size, img.mode))
            except:
                continue

    df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image Count"])

    st.subheader("🔢 Image Count Per Class")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df_counts.set_index("Class"))

    with col2:
        fig, ax = plt.subplots()
        ax.pie(df_counts["Image Count"], labels=df_counts["Class"], autopct="%1.1f%%", startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

    # Show sample images
    st.subheader("🖼 Sample Images Per Class")
    cols = st.columns(5)
    for i, cls in enumerate(class_dirs):
        try:
            folder = os.path.join(DATASET_PATH, cls)
            sample_img = random.choice(os.listdir(folder))
            img_path = os.path.join(folder, sample_img)
            img = Image.open(img_path).convert("RGB")
            with cols[i % 5]:
                st.image(img, caption=cls, use_container_width=True)
        except:
            continue

    # Show format info
    st.subheader("📏 Sample Image Properties")
    if image_formats:
        df_formats = pd.DataFrame(image_formats, columns=["Class", "Size (WxH)", "Mode"])
        st.dataframe(df_formats)
    else:
        st.info("No image format info available.")