import streamlit as st
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ========================
# 🦋 Title and Description
# ========================
st.title("🦋 Butterfly Image Classification using CNN")
st.write("""
Upload a butterfly image and the trained CNN model will predict its class.  
If the model is missing, it will automatically download from Google Drive.
""")

# ========================
# Load Model (with auto-download)
# ========================
MODEL_PATH = "butterfly_cnn_model.h5"
FILE_ID = "1xXzpnqrkh2cXe6AhkBq9sslk85l3vtMG"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found locally. Downloading from Google Drive... ⏳")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ========================
# Load Class Labels
# ========================
if os.path.exists("Training_set.csv"):
    train_csv = pd.read_csv("Training_set.csv")
    labels = sorted(train_csv['label'].unique())
else:
    st.error("❌ Training_set.csv not found. Please keep it in the same directory.")
    st.stop()

# ========================
# Image Upload Section
# ========================
uploaded_file = st.file_uploader("📸 Upload a butterfly image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join("temp_image.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # ========================
    # Preprocess and Predict
    # ========================
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = labels[pred_index]
    confidence = preds[0][pred_index] * 100

    # ========================
    # Display Results
    # ========================
    st.markdown(f"### 🦋 Predicted Class: **{pred_class}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")

    # Top 3 Predictions
    st.markdown("#### 🔝 Top 3 Predictions:")
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top3_classes = [labels[i] for i in top3_idx]
    top3_scores = preds[0][top3_idx] * 100

    result_df = pd.DataFrame({
        'Class': top3_classes,
        'Confidence (%)': top3_scores.round(2)
    })
    st.dataframe(result_df)

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.barh(top3_classes, top3_scores, color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top 3 Predicted Classes')
    st.pyplot(fig)
