"""Simple web interface for an exported detector checkpoint."""
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_image_detector.model import create_model

st.set_page_config(page_title="AI Image Detector", page_icon="🔎")
st.title("AI-Generated Image Detector")
st.caption("Research prototype — scores are not proof of image authenticity.")

checkpoint_path = Path("models/baseline/best_model.pt")

@st.cache_resource
def load_detector(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["architecture"], pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not checkpoint_path.exists():
        st.warning("Train a model first. Expected checkpoint: models/baseline/best_model.pt")
    else:
        model, checkpoint = load_detector(str(checkpoint_path))
        preprocess = transforms.Compose([transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            fake_probability = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0, 0].item()
        st.metric("AI-generated probability", f"{fake_probability:.1%}")
        st.info("Likely AI-generated" if fake_probability >= 0.5 else "Likely real")
