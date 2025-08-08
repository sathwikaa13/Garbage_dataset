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
