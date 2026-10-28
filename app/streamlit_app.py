"""Interactive Streamlit interface for an exported detector checkpoint."""
import hashlib
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_image_detector.model import create_model

st.set_page_config(page_title="Verity | AI image detector", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#f6f7fb}[data-testid="stSidebar"]{background:#111827}[data-testid="stSidebar"] *{color:#f9fafb}
.hero{padding:1.4rem 0 .6rem}.eyebrow{color:#6d5dfc;font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.hero h1{color:#161b2a;font-size:clamp(2rem,4vw,3.3rem);margin:.2rem 0;letter-spacing:-.045em}.hero p{color:#5f6678;font-size:1.05rem;max-width:42rem}
.result-card{background:white;border:1px solid #e7e9f1;border-radius:18px;padding:1.35rem;box-shadow:0 8px 30px rgba(33,39,66,.06)}
.verdict{font-weight:750;font-size:1.55rem;color:#161b2a;margin-bottom:.25rem}.subtle{color:#687085;font-size:.9rem}
.stButton>button{border-radius:10px;font-weight:650;min-height:2.7rem}[data-testid="stMetric"]{background:white;border:1px solid #e7e9f1;border-radius:14px;padding:.8rem}
</style>""", unsafe_allow_html=True)

CHECKPOINT_PATH = Path("models/cifake_baseline/best_model.pt")
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

@st.cache_resource(show_spinner=False)
def load_detector(path: str):
    """Load the trained classifier once per active app session."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["architecture"], pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint

def classify(image: Image.Image):
    model, checkpoint = load_detector(str(CHECKPOINT_PATH))
    preprocess = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        probabilities = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]
    return probabilities[0].item(), checkpoint  # CIFAKE: class 0 is FAKE.

if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

with st.sidebar:
    st.markdown("## ✦ VerifiAI")
    st.caption("Image provenance explorer")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Upload an image\n\n2. Run an analysis\n\n3. Review the confidence signal")
    st.caption("Scores are a model signal, not proof of origin.")
    if st.session_state.history and st.button("Clear session history", use_container_width=True):
        st.session_state.history, st.session_state.result = [], None
        st.rerun()

st.markdown("""<div class="hero"><div class="eyebrow">Visual authenticity research</div><h1>Understand what your image may be.</h1><p>Upload a JPEG, PNG, or WebP image to estimate whether it resembles the AI-generated examples used to train this detector.</p></div>""", unsafe_allow_html=True)
upload_col, analysis_col = st.columns([1.05, 1], gap="large")
with upload_col:
    uploaded = st.file_uploader("Drop an image here", type=SUPPORTED_TYPES)
    if uploaded:
        raw_bytes = uploaded.getvalue()
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)
        info_left, info_right, info_size = st.columns(3)
        info_left.metric("Dimensions", f"{image.width} × {image.height}")
        info_right.metric("Format", uploaded.type.split("/")[-1].upper())
        info_size.metric("File size", f"{len(raw_bytes) / 1024:.0f} KB")
        if not CHECKPOINT_PATH.exists():
            st.warning("No trained detector found. Train a model to enable analysis.")
        elif st.button("Analyze image", type="primary", use_container_width=True):
            with st.spinner("Inspecting visual patterns…"):
                ai_probability, checkpoint = classify(image)
            fingerprint = hashlib.sha1(raw_bytes).hexdigest()[:8]
            st.session_state.result = {"name": uploaded.name, "fingerprint": fingerprint, "ai_probability": ai_probability, "architecture": checkpoint["architecture"]}
            st.session_state.history = [st.session_state.result] + [item for item in st.session_state.history if item["fingerprint"] != fingerprint]
            st.session_state.history = st.session_state.history[:5]
    else:
        st.info("Choose an image to start an analysis. Your upload stays in this browser session.")

with analysis_col:
    result = st.session_state.result
    if result:
        ai_probability = result["ai_probability"]
        real_probability = 1 - ai_probability
        verdict = "Likely AI-generated" if ai_probability >= 0.5 else "Likely photographic"
        signal = ai_probability if ai_probability >= .5 else real_probability
        confidence_label = "strong" if signal >= .8 else "moderate" if signal >= .6 else "low"
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="verdict">{verdict}</div><div class="subtle">{confidence_label.title()} confidence · analyzed as {result["architecture"]}</div>', unsafe_allow_html=True)
        st.progress(ai_probability, text=f"AI-generated signal  {ai_probability:.1%}")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("AI-generated", f"{ai_probability:.1%}")
        metric_b.metric("Photographic", f"{real_probability:.1%}")
        st.caption("Interpret the result alongside source, context, and corroborating evidence.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card"><div class="verdict">Awaiting an image</div><div class="subtle">Your result will appear here after you run an analysis.</div></div>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("### Recent analyses")
    history_columns = st.columns(min(len(st.session_state.history), 5))
    for column, item in zip(history_columns, st.session_state.history):
        with column:
            st.caption(item["name"][:22])
            st.metric("AI-like" if item["ai_probability"] >= .5 else "Photo-like", f"{item['ai_probability']:.0%}")
