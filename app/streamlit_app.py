import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import streamlit as st
st.write("Streamlit is working!")
# allow importing project modules
sys.path.append("..")

from src.models.vit_model import FairViTMed
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="FairViT-Med",
    layout="centered"
)

st.title("FairViT-Med")
st.subheader("Fairness-Aware Vision Transformer for Medical Diagnosis")

st.write(
"""
Upload a **Chest X-ray image** to get a model prediction.

This model is trained with fairness evaluation across demographic groups.
"""
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FairViTMed(num_classes=1)

    model.load_state_dict(
        torch.load("../models/baseline_model.pth")
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "Pneumonia" if prob > 0.5 else "Normal"

    st.markdown("### Prediction")

    st.write(f"**Diagnosis:** {prediction}")
    st.write(f"**Confidence:** {prob:.3f}")

# -----------------------------
# Fairness Metrics Display
# -----------------------------
st.markdown("---")
st.markdown("### Model Fairness Metrics")

st.write("These metrics were computed during validation.")

col1, col2, col3 = st.columns(3)

col1.metric("Male AUC", "0.63")
col2.metric("Female AUC", "0.61")
col3.metric("Fairness Gap", "0.02")

st.markdown("---")

st.write(
"""
**Fairness Gap** measures performance difference between demographic groups.
Lower values indicate more equitable predictions.
"""
)