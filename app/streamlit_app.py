import streamlit as st
import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

st.set_page_config(
    page_title="FairViT-Med",
    page_icon="🔬",
    layout="wide",
)

st.markdown('\n<style>\n.stApp{background:#f4f7fb;color:#16243a}\n[data-testid="stHeader"]{background:rgba(244,247,251,.96)}\n.block-container{padding-top:2.4rem;padding-bottom:2rem;max-width:1440px}\nh1,h2,h3{letter-spacing:-.035em;color:#16243a}\nh3{font-size:1.35rem!important}\n[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2e8f0}\n[data-testid="stSidebar"] .block-container{padding-top:2rem}\n[data-testid="stMetric"]{background:#fff;border:1px solid #e0e8f1;border-radius:16px;padding:19px 20px;box-shadow:0 4px 16px #172b4d05;min-height:118px}\n[data-testid="stMetricLabel"]{color:#5b6b82;font-size:.8rem}\n[data-testid="stMetricValue"]{font-size:clamp(1.35rem,2vw,2rem);font-weight:650;letter-spacing:-.04em}\n[data-testid="stCaptionContainer"]{color:#607087}\n[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:16px}\n[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]{background:white;border:1px solid #e0e8f1;border-radius:16px;overflow:hidden}\n[data-baseweb="tab-list"]{gap:8px;background:#e9eef5;border-radius:12px;padding:5px;margin-bottom:22px}\nbutton[data-baseweb="tab"]{padding:10px 20px;border-radius:9px;height:44px}\nbutton[data-baseweb="tab"][aria-selected="true"]{background:white;color:#2968a9;box-shadow:0 2px 6px #172b4d10}\n[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none}\n.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#2968a9;border:0;border-radius:10px;min-height:46px;font-weight:650}\n.stButton>button,.stDownloadButton>button{border-radius:10px}\n[data-testid="stFileUploaderDropzone"]{border:1px dashed #91abc6;border-radius:16px;background:#edf4fb;padding:28px}\n[data-testid="stAlert"]{border-radius:12px}\n[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}\n[data-testid="stSidebar"] [role="radiogroup"] label{padding:10px 12px;border-radius:10px}\n[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#e8f3f2;font-weight:600}\n.brand{display:flex;align-items:center;gap:10px;font-weight:750;font-size:21px;letter-spacing:-.7px;margin-bottom:22px}\n.brand-mark{background:#2968a9;color:white;display:inline-flex;align-items:center;justify-content:center;width:35px;height:35px;border-radius:10px;font-size:17px}\n.hero{background:linear-gradient(110deg,#112339,#203e57);color:white;border-radius:22px;padding:32px 36px;margin:8px 0 28px;position:relative;overflow:hidden;border:1px solid #314d66}\n.hero h1{color:white!important;font-size:clamp(2rem,3.5vw,3rem);font-weight:720;margin:8px 0 12px;padding:0;max-width:850px}\n.hero p{color:#c6d6e7;max-width:750px;line-height:1.6;margin:0}\n.eyebrow{color:#8de0cf;font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase}\n.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}\n.chip{font-size:11px;padding:6px 11px;border:1px solid #536b80;border-radius:6px;color:#e2edf7}\n.empty{border:1px dashed #c6d3e0;background:#f9fbfd;border-radius:16px;padding:48px 26px;text-align:center;color:#66788f;min-height:235px}\n.empty strong{display:block;color:#243c55;font-size:19px;margin:14px 0 9px}\n.section-label{font-size:11px;font-weight:750;letter-spacing:1.5px;color:#2968a9;text-transform:uppercase;margin:4px 0 10px}\n.footer{border-top:1px solid #dce5ef;padding-top:18px;margin-top:36px;font-size:11px;color:#6a7b90;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}\n@media(max-width:768px){.block-container{padding:1.5rem 1rem}.hero{padding:25px 22px}.hero h1{font-size:2rem} [data-testid="stMetric"]{padding:15px;min-height:100px}[data-testid="stMetricValue"]{font-size:1.6rem}button[data-baseweb="tab"]{padding:8px 12px;font-size:12px}}\n</style>\n', unsafe_allow_html=True)
st.markdown('<div class="brand"><span class="brand-mark">F</span>FairViT / Research Studio</div><div class="hero"><div class="eyebrow">Vision transformer research</div><h1>A closer look at model behaviour.</h1><p>Upload a chest X-ray sample and explore the baseline classifier’s output in a focused research workspace.</p><div class="chips"><span class="chip">ViT-B/16</span><span class="chip">BASELINE CHECKPOINT</span><span class="chip">CPU INFERENCE</span></div></div>', unsafe_allow_html=True)
st.caption("Research demonstration only. Not validated for clinical use; cannot diagnose or rule out disease.")


class FairViTMed(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=None)
        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, 1
        )

    def forward(self, x):
        return self.vit(x)


@st.cache_resource
def load_model():
    torch.set_num_threads(1)

    checkpoint_path = hf_hub_download(
        repo_id="Visveswaranvisves/fairvit-medical-ai",
        filename="baseline_model.pth",
        revision="c90fbaa4612fff2db348514eef9f074fcc479636",
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    # Support a plain state dictionary or a wrapped checkpoint.
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    with torch.device("meta"):
        model = FairViTMed()

    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model


# Matches the current training dataloader in your repository.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

preview, results = st.columns([1.1, 1], gap="large")
with preview:
    st.markdown('<div class="section-label">01 / Image workspace</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a sample chest X-ray", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        except Exception:
            st.error("Could not read this image. Try another PNG or JPEG.")
            st.stop()
        st.image(image, caption="Uploaded sample · RGB input", use_container_width=True)
    else:
        st.markdown('<div class="empty"><span style="font-size:40px">⊞</span><strong>Your image workspace</strong>Upload a PNG or JPEG to preview the sample before analysis.</div>', unsafe_allow_html=True)
with results:
    st.markdown('<div class="section-label">02 / Inference result</div>', unsafe_allow_html=True)
    st.subheader("Explore the model output")
    st.write("The baseline model returns a score for the pneumonia class. Interpret it alongside the model’s research limitations.")
    run = st.button("Analyse sample", type="primary", use_container_width=True, disabled=uploaded is None)
    if run and uploaded is not None:
        try:
            with st.spinner("Preparing the model and analysing your sample. The first run may take longer…"):
                model = load_model()
                tensor = preprocess(image).unsqueeze(0)
                with torch.inference_mode():
                    score = torch.sigmoid(model(tensor)).item()
            st.metric("Pneumonia-class model score", f"{score:.3f}")
            st.progress(float(score))
            st.write("**Model output:** " + ("Above the pneumonia-class threshold" if score > 0.5 else "Below the pneumonia-class threshold"))
            st.caption("Decision threshold: 0.5 · Score range: 0–1")
            st.info("This is not a calibrated clinical probability. A low score does not mean the X-ray is normal.")
        except Exception as exc:
            st.error("Model loading or inference failed.")
            with st.expander("Technical details"):
                st.exception(exc)
    else:
        st.markdown('<div class="empty"><span style="font-size:36px">◉</span><strong>Ready when you are</strong>Your model score and threshold comparison will appear after analysis.</div>', unsafe_allow_html=True)
    with st.expander("About this research model"):
        st.write("Architecture: Vision Transformer B/16 with a single-output classification head. Input: RGB image resized to 224 × 224. Uses the baseline checkpoint; no fairness-adjusted model is selected.")
        st.write("The app does not verify whether an uploaded image is a chest X-ray. Use appropriate research samples. This demonstration is not a medical device.")
st.markdown('<div class="footer"><span>FairViT-Med · Built by Visveswaran V</span><span>Research use only · Baseline model</span></div>', unsafe_allow_html=True)
