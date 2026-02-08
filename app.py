import streamlit as st
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ========================
# Page config
# ========================
st.set_page_config(page_title="Butterfly CNN", layout="centered")

st.title("🦋 Butterfly Image Classification using CNN")
st.write(
    "Upload a butterfly image and the trained CNN model will predict its class."
)

# ========================
# Constants
# ========================
MODEL_PATH = "butterfly_cnn_model.h5"
FILE_ID = "1xXzpnqrkh2cXe6AhkBq9sslk85l3vtMG"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ========================
# Download + Load Model (CACHED)
# ========================
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# ========================
# Load Class Labels (CACHED)
# ========================
@st.cache_data
def load_labels():
    df = pd.read_csv("Training_set.csv")
    return sorted(df["label"].unique())

labels = load_labels()

# ========================
# Image Upload
# ========================
uploaded_file = st.file_uploader(
    "📸 Upload a butterfly image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (IN MEMORY)
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = labels[pred_index]
    confidence = preds[0][pred_index] * 100

    st.markdown(f"### 🦋 Predicted Class: **{pred_class}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")

    # Top 3 predictions
    st.markdown("#### 🔝 Top 3 Predictions")
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top3_classes = [labels[i] for i in top3_idx]
    top3_scores = preds[0][top3_idx] * 100

    df = pd.DataFrame({
        "Class": top3_classes,
        "Confidence (%)": top3_scores.round(2)
    })
    st.dataframe(df, use_container_width=True)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(top3_classes, top3_scores)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 3 Predictions")
    st.pyplot(fig)
